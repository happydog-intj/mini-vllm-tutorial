# adv14 — Multi-LoRA 动态切换

## 1. 教学目标

- 理解 LoRA(Low-Rank Adaptation)低秩适配器的原理与参数结构
- 掌握如何在同一 base 模型上挂载多个独立的 LoRA adapter
- 学习推理时按请求动态切换 adapter 的工程模式(无需重载模型)
- 对比教学版简化实现与真实推理框架(vLLM/PEFT)的差距

---

## 2. 问题

**场景:多任务共享同一 base 模型**

大型语言模型(如 LLaMA-3、Qwen3)参数量达数十亿,为每个下游任务维护
一份完整模型副本代价极高。LoRA 微调只训练少量低秩矩阵(参数量 < 1%),
但推理时如何高效处理来自不同任务/用户的请求?

**核心矛盾:**

```
请求 A (客服任务)  → 需要 adapter-0
请求 B (代码生成)  → 需要 adapter-1
请求 C (翻译任务)  → 需要 adapter-2
         ↓
如何在不重载模型的前提下,同时/快速切换服务这三类请求?
```

朴素方案"每请求重载模型"会导致:
- GPU 显存反复换入换出(带宽瓶颈)
- 吞吐量暴降,无法支持高并发推理服务

---

## 3. 原理

**LoRA 低秩分解 + 动态 adapter 切换:**

```
输入 x [*, in_features]
         │
         ├─────────────────────────────────────────────┐
         │                                             │
         ▼                                             ▼
  ┌─────────────┐                          ┌──────────────────────┐
  │  base(x)    │                          │  LoRA 旁路 (adapter) │
  │  W0 · x     │                          │                      │
  │  [*, out]   │                          │  A: [in → r]         │
  └──────┬──────┘                          │  B: [r  → out]       │
         │                                 │  delta = B(A(x))     │
         │                                 └──────────┬───────────┘
         │                                            │
         └──────────────── + ─────────────────────────┘
                           │
                           ▼
                    y = W0·x + B·A·x
                    [*, out_features]

  adapter-0: A0, B0  ──┐
  adapter-1: A1, B1  ──┤── 按请求 idx 切换 active adapter
  adapter-2: A2, B2  ──┘
```

**关键等式:**  W' = W0 + B·A,其中 r << min(in, out)

base 模型 W0 **冻结不动**,只有 A、B 矩阵随任务不同而切换。
切换成本:仅需更改指针/索引,无需任何显存拷贝。

---

## 4. 实现细节

### 4.1 LoRALinear

```
LoRALinear(base, r=4, num_adapters=2)
  ├── base: nn.Linear(in, out)    ← 冻结 (requires_grad=False)
  ├── adapters['0']:
  │     A: nn.Linear(in, r)       ← 降维投影
  │     B: nn.Linear(r, out)      ← 升维投影
  ├── adapters['1']:
  │     A: nn.Linear(in, r)
  │     B: nn.Linear(r, out)
  └── active: '0'                 ← 当前激活的 adapter
```

**形状链验证(r=2, in=out=8):**

```
x: [1, 8]
  → A: [8→2] → [1, 2]
  → B: [2→8] → [1, 8]   # 与 base(x) 形状一致,可直接相加
```

**forward:**
```python
def forward(self, x):
    out = self.base(x)          # W0·x  [*, out]
    A, B = self.adapters[self.active]
    return out + B(A(x))        # + B·A·x [*, out]
```

### 4.2 set_adapter

```python
def set_adapter(self, idx: int):
    self.active = str(idx)      # O(1),仅更新字符串键
```

推理时每个请求调用一次 `set_adapter`,无显存移动,开销可忽略。

### 4.3 MultiLoRAEngine

```python
class MultiLoRAEngine:
    def generate(self, token_ids, adapter_idx, steps=4):
        # 遍历所有 LoRALinear,统一切换到 adapter_idx
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.set_adapter(adapter_idx)
        return model(token_ids)
```

---

## 5. 教学版 vs 真实框架

| 维度 | 教学版 (本目录) | 真实框架 (vLLM/PEFT) |
|------|----------------|----------------------|
| adapter 存储 | `nn.ModuleDict` in-memory | 按需从磁盘/显存池加载 |
| 批量调度 | 逐请求串行切换 | SGMV/BGMV CUDA kernel:同批次多 LoRA 并行 |
| 切换粒度 | 整个模型统一切换 | per-layer 细粒度,支持混合 batch |
| 显存管理 | 所有 adapter 常驻 | LRU 缓存,动态换入换出 |
| 精度 | FP32 | FP16/BF16 + 可选量化(QLoRA) |
| 吞吐量 | 教学演示 | 数百 adapter × 数千 req/s |

**SGMV/BGMV Kernel 原理(简述):**

vLLM 使用 `punica` / `lora-sgmv` 库,在单次 CUDA kernel 调用中:
1. 输入 batch 中每条请求携带 adapter_idx
2. Kernel 按 idx 查表,对不同请求使用不同的 A/B 矩阵做分段矩阵乘
3. 结果合并,吞吐量接近单 adapter 的线性 batch 推理

---

## 6. 运行

```bash
cd advanced/adv14_multi_lora

# 核心演示 + 断言验证
python run.py

# 批量调度演示(多请求)
python multi_lora_engine.py
```

**预期输出(run.py):**

```
✓ 验证 1: base 权重 requires_grad=False (已冻结)

adapter=0 输出: [...]
adapter=1 输出: [...]
✓ 验证 2: adapter=0 与 adapter=1 产生不同输出
✓ 验证 3: 切换回 adapter=0 后输出与首次一致
✓ 验证 4: 形状链正确 — x[1,8] → A → [2] → B → [1,8]

adapter 输出差值 (L∞): ...
adapter 输出差值 (L2): ...

✅ adv14_multi_lora 通过
```

---

## 7. 下一步

本节展示了 **多 adapter 静态预载 + 动态切换** 的教学模式。

下一节 **adv15 — Guided Decoder(约束解码)** 将在解码层面施加约束:
- JSON/正则表达式约束采样
- 状态机驱动的 token mask
- 适用于结构化输出场景(工具调用、Schema 生成等)

→ 参见 `advanced/adv15_guided_decoder/`
