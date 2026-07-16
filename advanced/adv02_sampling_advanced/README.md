# 采样进阶：MinP / 惩罚项 / Beam Search

## 1. 教学目标

- 理解 **MinP** 采样的过滤逻辑，掌握它与 TopK/TopP 的本质区别
- 掌握三种惩罚项（频率惩罚 / 存在惩罚 / 重复惩罚）的含义与差异
- 理解 **Beam Search** 的搜索树结构，以及它与贪心解码的区别
- 能区分教学实现与生产框架（vLLM / transformers）的差异

---

## 2. 问题：主系列 step06 缺了什么？

`step06_sampler` 覆盖了 Greedy / Temperature / TopK / TopP / Gumbel-Max，
但生产中常用的以下三类机制尚未涉及：

| 缺失机制 | 实际影响 |
|----------|---------|
| **MinP** 采样 | 比 TopP 更自适应，已被 llama.cpp / Mistral 等采用 |
| **重复惩罚** (三种) | 不加惩罚，模型容易陷入重复循环 |
| **Beam Search** | 翻译、摘要等精确任务中仍是主流选择 |

---

## 3. 原理

### 3.1 TopK vs TopP vs MinP 候选集对比

```
示例 logits（6 个 token，T=1.0）：
  token:  A      B      C      D      E      F
  prob:  0.60   0.25   0.08   0.04   0.02   0.01
         ↑ max_p = 0.60

TopK (k=3) — 固定保留 3 个：
  ┌────┬────┬────┬────┬────┬────┐
  │ A  │ B  │ C  │ ×  │ ×  │ ×  │   候选集固定为 3，不管分布形状
  └────┴────┴────┴────┴────┴────┘

TopP (p=0.90) — 累积概率截断：
  累积：  0.60  0.85  0.93  ...
  ┌────┬────┬────┬────┬────┬────┐
  │ A  │ B  │ C  │ ×  │ ×  │ ×  │   本例与 TopK=3 相同，但平坦分布时会纳入更多
  └────┴────┴────┴────┴────┴────┘

MinP (min_p=0.05) — 绝对概率阈值（相对最大值）：
  阈值 = max_p × min_p = 0.60 × 0.05 = 0.030
  保留 prob >= 0.030 的 token：A(0.60) B(0.25) C(0.08) D(0.04)
  ┌────┬────┬────┬────┬────┬────┐
  │ A  │ B  │ C  │ D  │ ×  │ ×  │   候选集随最大概率动态调整
  └────┴────┴────┴────┴────┴────┘

分布越尖锐（max_p 越高），阈值越高，候选集越小（自适应收紧）
分布越平坦（max_p 越低），阈值越低，候选集越大（自适应放宽）
```

### 3.2 三种惩罚项

```
已生成序列：[A, B, A, C, A, B]
           token A 出现 3 次，B 出现 2 次，C 出现 1 次

频率惩罚 (frequency_penalty = 0.5):
  logit[A] -= 0.5 × 3 = 1.5    ← 出现越多，惩罚越重（线性）
  logit[B] -= 0.5 × 2 = 1.0
  logit[C] -= 0.5 × 1 = 0.5

存在惩罚 (presence_penalty = 0.5):
  logit[A] -= 0.5 × 1 = 0.5    ← 出现过就减固定值，不管次数
  logit[B] -= 0.5 × 1 = 0.5
  logit[C] -= 0.5 × 1 = 0.5

重复惩罚 (repetition_penalty = 1.3):
  logit[A] /= 1.3               ← 乘法惩罚，对正值 logit 效果更强
  logit[B] /= 1.3
  logit[C] /= 1.3

结论：
  多次出现的 token → 用频率惩罚（次数越多，压制越重）
  只要出现过就惩罚 → 用存在惩罚（鼓励话题多样性）
  乘法型强压制     → 用重复惩罚
```

### 3.3 Beam Search 搜索树（宽度=2，max_new=3）

```
起点: [SOS]

Step 1 — 展开 Top-2：
  ┌─────────────────────────────────────┐
  │           [SOS]                     │
  │          /     \                    │
  │        A(-0.5) B(-1.2)              │  保留 2 个候选
  └─────────────────────────────────────┘

Step 2 — 每个候选再展开 Top-2，共 4 个，保留最优 2 个：
  ┌─────────────────────────────────────┐
  │   [SOS,A]           [SOS,B]         │
  │   /     \           /     \         │
  │ A,C     A,D       B,C     B,A       │
  │(-0.5-0.3) (-0.5-0.8) (-1.2-0.2) (-1.2-0.9)  │
  │= -0.8    = -1.3    = -1.4    = -2.1 │
  │ ✓ 保留   ✓ 保留   × 淘汰   × 淘汰  │
  └─────────────────────────────────────┘

Step 3 — 再展开，最终保留最优序列：
  最优: [SOS, A, C, X]  （累积 logprob 最高）

Greedy 只保留每步最优 1 条路径，可能错过全局最优。
Beam 保留 beam_width 条，在搜索质量和计算量之间权衡。
```

---

## 4. 实现细节

### 4.1 `min_p_sample`

```python
probs = torch.softmax(logits / temperature, dim=-1)
max_p = probs.max()
mask = probs >= max_p * min_p
filtered = torch.where(mask, probs, torch.zeros_like(probs))
return torch.multinomial(filtered, num_samples=1).squeeze(-1)
```

- 先计算 softmax 概率（含温度缩放）
- 找到最大概率 `max_p`，乘以 `min_p` 得到阈值
- 低于阈值的 token 置零（等价于过滤掉）
- `torch.multinomial` 按相对权重采样，自动等价于对过滤后分布重归一化

### 4.2 `apply_frequency_penalty`

```python
freq = torch.bincount(token_ids, minlength=logits.size(-1)).float()
return logits - penalty * freq
```

`bincount` 统计每个 token 的出现次数，线性叠加惩罚。

### 4.3 `apply_presence_penalty`

```python
appeared = torch.bincount(token_ids, minlength=logits.size(-1)).clamp(0, 1).float()
return logits - penalty * appeared
```

`.clamp(0, 1)` 将次数截断为 0/1，使惩罚与次数无关。

### 4.4 `apply_repetition_penalty`

```python
appeared = torch.bincount(token_ids, minlength=logits.size(-1)).clamp(0, 1).bool()
logits = logits.clone()
logits[appeared] = logits[appeared] / penalty
```

用布尔 mask 定位出现过的 token，除以 penalty 实现乘法型惩罚。
注意 `.clone()` 避免 in-place 修改原 tensor。

### 4.5 `beam_search`

```python
beams = [(prompt_ids, 0.0)]
for _ in range(max_new):
    all_cands = []
    for ids, score in beams:
        logits, _ = model(ids[-1:], past_key_values=None)
        logp = torch.log_softmax(logits[-1], dim=-1)
        topk = torch.topk(logp, beam_width)
        for v, idx in zip(topk.values, topk.indices):
            all_cands.append((torch.cat([ids, idx.view(1)]), score + v.item()))
    beams = sorted(all_cands, key=lambda c: c[1])[-beam_width:]
return beams[-1][0]
```

- `all_cands` 收集当前所有候选的所有展开（最多 `beam_width²` 个）
- 按累积 log 概率排序，保留最后 `beam_width` 个（最高分）
- 教学简化：每步传入最后一个 token，不使用 KV Cache（生产中会用）

---

## 5. 教学版 vs 真实框架

| 特性 | 本教学实现 | vLLM / transformers |
|------|-----------|-------------------|
| **MinP** | 过滤 + multinomial | 同样逻辑，已内置参数 `min_p` |
| **频率惩罚** | `logits -= penalty * count` | OpenAI API `frequency_penalty` 同逻辑 |
| **存在惩罚** | `logits -= penalty * (count > 0)` | OpenAI API `presence_penalty` 同逻辑 |
| **重复惩罚** | 正/负 logit 统一除以 penalty | transformers 区分正负：正值除，负值乘 |
| **Beam Search** | 全量重算，无 KV Cache | 配合 KV Cache，每步只计算新 token |
| **Beam 长度惩罚** | 无 | 引入 `length_penalty` 平衡长短序列 |
| **惩罚叠加** | 单独函数 | SamplingParams 支持多惩罚同时生效 |

**重复惩罚的符号约定差异**（教学 vs 生产）：

```python
# 教学版（简化）：统一除以 penalty
logits[appeared] = logits[appeared] / penalty

# transformers/vLLM 生产版（更符合直觉）：
# 正 logit 除以 penalty → 降低
# 负 logit 乘以 penalty → 更负，进一步降低
for i in appeared:
    if logits[i] > 0:
        logits[i] /= penalty
    else:
        logits[i] *= penalty
```

---

## 6. 运行

```bash
cd advanced/adv02_sampling_advanced
python run.py
```

预期输出：

```
==================================================
1. MinP 采样
   vocab_size=20, min_p=0.1
   过滤后候选 token 数: 14
   ...
   [PASS] MinP 采样 50 次均在候选集内

==================================================
2. 惩罚项验证
   [频率惩罚] token 3 (出现3次): logit 1.1483 → -3.3517  下降 4.5000
   [存在惩罚] token 3: logit 1.1483 → -0.3517  下降 1.5000
   [重复惩罚] token 3: 1.1483 → 0.7656  ✓ (正 logit 降低)
   ...
   [PASS] 三种惩罚项验证通过

==================================================
3. Beam Search 长度验证
   prompt_len=5, max_new=10, beam_width=3
   期望输出长度: 15, 实际输出长度: 15
   [PASS] Beam Search 长度验证通过

✅ adv02_sampling_advanced 通过
```

---

## 7. 下一步

下一节 **adv03 投机解码（Speculative Decoding）**：

用一个小草稿模型（draft model）快速生成多个候选 token，再由大目标模型一次性并行验证。
验证通过的 token 直接接受，拒绝时回退。在保持输出分布等价的前提下，
将吞吐量提升 2~3 倍——这是当前生产推理系统的主要加速手段之一。

相关概念：拒绝采样（rejection sampling）、接受率（acceptance rate）、
草稿模型与目标模型对齐。
