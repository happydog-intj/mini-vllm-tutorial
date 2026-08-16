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

### 3.0 数学直觉：为什么"低秩"能捕捉任务差异

> 以下用 3Blue1Brown 式的"从几何图像建立直觉"的方式来理解 LoRA 的数学本质。

**从线性变换的几何意义开始。**

一个 d×d 的权重矩阵 W 代表一个线性变换——它把输入向量"旋转、缩放、投影"到输出空间。全量微调意味着修改这个变换的**每一个方向**，需要调整 d² 个参数：

```
全量微调：W' = W₀ + ΔW

  ΔW 是一个完整的 d×d 矩阵
  d=4096 → ΔW 有 16,777,216 个自由度
  它可以在任意方向上修改变换
```

**LoRA 的核心假设：任务差异是"低维"的。**

想象一个 4096 维空间里的线性变换。全量微调可以在所有 4096 个方向上调整。但 LoRA 假设：从"通用语言模型"适配到"客服模型"，实际只需要调整**少数几个关键方向**——绝大多数方向上，通用模型已经够好了。

```
几何直觉（想象 3D 空间，实际是 4096D）：

  全量微调 ΔW：可以在 x、y、z 三个方向上任意调整
  → 3×3 = 9 个参数

  低秩微调 ΔW = B·A（rank=1）：只能在一个"平面"上调整
  → 压缩到某个特定方向
  → 参数量: 3+3 = 6（而非 9）

  rank=2：可以在两个方向上调整
  → 参数量: 3×2 + 2×3 = 12?  不，是 d×r + r×d = 3×2 + 2×3 = 12
  → 但当 d=4096, r=8 时：4096×8 + 8×4096 = 65,536 vs d²=16,777,216
  → 节省 256 倍参数！
```

**为什么 B·A 的乘积形式天然是低秩的？**

这是线性代数的基本事实：一个 [d, d] 矩阵如果能写成 [d, r] × [r, d] 的乘积，它的秩至多为 r。几何上，A 先把 d 维输入"压缩"到 r 维瓶颈，B 再把这 r 维"展开"回 d 维输出：

```
输入空间 (4096 维)              瓶颈 (r=8 维)              输出空间 (4096 维)
    ●───────────────→ A ──→ ●───────────→ B ──→ ●
    所有信息                  只保留 8 个                再映射回
                             最关键的方向               完整空间

    这 8 个方向就是 LoRA 认为"任务差异"所在的子空间
```

**一个直观的类比。**

想象你有一张 4096×4096 像素的照片（全量权重），要告诉别人"这张照片和原图有什么不同"：

```
全量微调 = 发送完整的差异图（16M 像素全发）
LoRA     = 说"沿着这 8 个方向做了调整"（只发 65K 个数字）

就像 PCA/SVD 能用少数主成分近似一张图片，LoRA 假设
任务微调的 ΔW 本身就是低秩的——它只在少数方向上有实质性变化。
```

**为什么这个假设是合理的？**

实证研究（Aghajanyan et al., 2020）发现：预训练语言模型的微调权重变化 ΔW 确实集中在一个低维子空间里。直觉上，预训练已经学会了"语言的通用结构"，微调只需要在"语气/领域/格式"等少数维度上做调整：

```
预训练 → 通用语言能力（需要 d² 参数来表达）
微调   → "请用客服语气回答"（只需要调整少数几个方向）

ΔW 的奇异值分布:
  σ₁ ████████████████  ← 前几个奇异值很大
  σ₂ ██████████
  σ₃ █████
  σ₄ ██
  σ₅ █
  ...
  σ_d ▎                 ← 大部分奇异值接近 0

  rank r=8 就能捕捉 >95% 的微调信息
```

**从"加法"看 Multi-LoRA 的优雅。**

既然 LoRA 的修改是 `y = W₀·x + B·A·x`，那切换任务只需要换一对 (A, B)：

```
base 输出:  W₀·x = [固定的通用表示]

任务 0:  W₀·x + B₀·A₀·x  ← 往"客服方向"偏移
任务 1:  W₀·x + B₁·A₁·x  ← 往"代码方向"偏移
任务 2:  W₀·x + B₂·A₂·x  ← 往"翻译方向"偏移

各任务的偏移彼此独立，base 模型纹丝不动
切换 = 换一对指针，零拷贝
```

---

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

### ❓ Q1：LoRA 的 rank r 怎么选？

**答案**：经验法则：

```
r=1~4:  极小适配，适合"风格迁移"类任务
r=4~8:  通用选择，指令微调、领域适配
r=8~16: 复杂任务，代码生成、数学推理

参数量对比（d=4096, r=8）:
  全量: d² = 16,777,216
  LoRA: 2×d×r = 65,536（约 0.4%）
```

### ❓ Q2：B·A·x 为什么不直接算 (B·A)·x？

**答案**：**显存和计算都亏**：

```
链式: x → A([in→r]) → B([r→out]) → 中间显存 [r]
合并: W_lora = B·A ([in,out]) → W_lora·x → 中间显存 [in,out] = d²

当 d=4096: 链式中间显存 8 vs 合并显存 64 MB！
LoRA 的核心价值就是避免构造完整的 [in,out] 矩阵。
```

### ❓ Q3：SGMV kernel 为什么比逐请求切换快？

**答案**：SGMV 的魔法在于**一个 kernel 内处理多个 adapter**：

```
逐请求: set_adapter → kernel launch → set_adapter → kernel launch（每次切换）
SGMV:   一个 kernel 调用，每行根据 adapter_idx 查表用不同 A/B
        → kernel launch 从 O(batch) 降到 O(1)
```

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
