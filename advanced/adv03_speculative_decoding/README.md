# adv03 — 投机解码 Speculative Decoding

## 1. 教学目标

- 理解 **投机解码 (Speculative Decoding)** 的核心思想:用一个小模型批量预测,再用大模型一次并行验证
- 掌握 `draft_speculate` / `target_verify` 两个核心函数的实现细节
- 通过断言验证:投机解码(贪婪模式)与纯自回归贪婪生成结果**完全一致**
- 直观感受 target model forward 次数的减少带来的潜在加速

---

## 2. 问题

### 自回归解码的瓶颈

标准自回归生成每步只产生 **1 个 token**,每步都需要完整地跑一次 target model forward:

```
token 1 → [Target Forward] → token 2
token 2 → [Target Forward] → token 3
token 3 → [Target Forward] → token 4
...
生成 N 个 token = N 次 Target Forward
```

**问题根源:** GPU 的算力远超单 token 所需,每步 forward 都严重**欠载** (memory-bound)。
大模型推理瓶颈不在算力,而在将权重从 HBM 搬到计算核心的**带宽**,每步只生成 1 token 浪费严重。

### 投机解码的思路

- 用一个**小而快**的草稿模型批量预测 k 个候选 token (计算成本低)
- 让大的目标模型**一次前向**并行验证所有 k 个 token (仍是 1 次 forward)
- 接受的 token 直接保留,拒绝时用目标模型的正确 token 替代

---

## 3. 原理

### 核心加速思想：批处理（Batching）

投机解码的加速本质是**把 decode 阶段的工作"攒一批"再做**，利用了和 prefill 相同的批处理思想。

GPU 推理的瓶颈是**显存带宽**（memory-bound），不是算力。每次 forward 都要把全部权重从 HBM 搬到计算核心，但标准 decode 每步只生成 1 个 token —— 算力严重浪费。

```
标准自回归：  生成 4 token = 4 次 target forward（每次搬一遍权重）
投机解码：    生成 4 token ≈ 1 次 target forward（一次搬权重，验证 4 个位置）
```

目标模型的验证步骤本质就是一次 **batch prefill**：

```
输入: [context, d0, d1, d2, d3]   ← 把 k 个候选拼成一个序列
输出: logits[0..L+k-1]           ← 因果 mask 保证每个位置只看到前面的 token
```

一次 forward 处理 k 个位置，和 prefill 阶段处理一整段 prompt 是同一个操作 —— 都是利用因果注意力在一次矩阵乘法中并行计算多个 token 的 logits。

| 操作 | 搬权重次数 | 计算的 token 数 |
|------|-----------|----------------|
| 自回归 4 步 | 4 次 | 每次 1 个 |
| 投机验证 1 次 | 1 次 | 一次 4 个 |

搬权重的代价几乎不随序列长度增加（memory-bound 下），所以验证 4 个 token 的耗时 ≈ 生成 1 个 token 的耗时。

**一句话总结**：投机解码 = 用小模型的串行成本换取大模型的批处理收益。

### 整体流程 (ASCII)

```
Context: [p0, p1, ..., pL-1]
              │
              ▼
    ┌─────────────────────────┐
    │    Draft Model (小)     │  ← 串行生成 k 个候选
    │   d_model=2, 速度快     │
    └─────────────────────────┘
              │  k 次 autoregressive forward
              ▼
    draft_tokens = [d0, d1, d2, d3]   (k=4)
              │
              ▼
    ┌─────────────────────────┐
    │    Target Model (大)    │  ← 一次并行验证
    │   d_model=4, 更精准     │
    └─────────────────────────┘
              │  1 次 forward,输入长度 = L + k
              ▼
    logits[L-1+j] = target 对第 j 草稿位的预测
              │
              ▼
    验收逐 token 对比:
      d0 == target_argmax[0]? → ✓ 接受
      d1 == target_argmax[1]? → ✓ 接受
      d2 == target_argmax[2]? → ✗ 拒绝 → 插入 target token, 丢弃后续
              │
              ▼
    accepted = [d0, d1, target_t2]   本轮生成 3 个 token, 仅 1 次 target forward
```

### 经典 Speculative Sampling 概率比公式

教学版使用 **argmax 对比**,真实框架采用概率比进行随机采样:

```
对草稿 token d_i:
  r = min(1,  p_target(d_i) / p_draft(d_i) )

以概率 r 接受 d_i:
  接受 → 继续验证 d_{i+1}
  拒绝 → 从修正分布 p_target - r * p_draft (归一化) 采样,本轮结束

这样保证:最终输出的 token 分布 == 纯目标模型输出分布 (无偏!)
```

### ❓ Q1：概率比公式为什么能保证"无偏"？

**问题**：`r = min(1, p_target(d_i) / p_draft(d_i))` 这个公式怎么就能保证最终分布等于目标模型分布？

**答案**：核心直觉是**"草稿模型提议，目标模型裁决"**。让我们用直觉（非严格证明）理解：

```
设草稿模型说 token "A" 的概率是 p_d(A) = 0.4
目标模型说 token "A" 的概率是 p_t(A) = 0.6

r = min(1, 0.6/0.4) = 1.0 → 总是接受！
解释：目标模型比草稿模型更确信 "A" 是好的，所以放心接受。

反过来：
p_d(A) = 0.6, p_t(A) = 0.2
r = min(1, 0.2/0.6) = 0.33 → 只有 33% 概率接受
解释：草稿模型过度自信了，目标模型不太认同。
```

**被拒绝时怎么办？** 不是简单地跳过，而是从修正分布 `p_target - r × p_draft` 采样。这个修正分布恰好补偿了"草稿过采样但被拒绝"的那些概率质量。数学上可以证明：接受概率 × 保留概率 + 拒绝概率 × 修正采样 = p_target。严格证明见 Leviathan et al. (2023) 的 Speculative Decoding 原论文。

### ❓ Q2：草稿模型和目标模型共享 KV Cache 吗？

**问题**：草稿生成 k 个 token 时，目标模型的 KV Cache 要不要也维护一份？

**答案**：教学版每轮重算（"简化"），但**真实框架必须维护两套 KV Cache**：

```
草稿模型 KV：[context + d0, d1, d2, d3]  ← 草稿逐步生成，KV 逐步累积
目标模型 KV：[context + d0, d1, d2, d3]  ← 一次前向计算全部

草稿 KV 在验证后就被丢弃了（因为目标模型已经算出来了）
```

为什么草稿不能直接用目标的 KV？因为**模型不同，权重不同**，算出来的 K/V 值不同。即使是 self-drafting（同一模型不同层），KV 也不同（草稿用浅层的投影，目标用完整的）。EAGLE 等方案让草稿头复用目标的 hidden states，可以省掉草稿的 KV 计算。

### ❓ Q3：什么时候投机解码不如直接自回归？

**问题**：教学版说"d_model=2 随机权重接受率极低"。那什么场景下投机解码反而更慢？

**答案**：投机解码的**净加速条件**是：

```
加速 = (接受的 token 数 + 1) / (1 + 草稿生成开销 / 目标单次开销)

接受率低时（比如 10%）：
  每轮平均接受 1.1 个 token
  但草稿花了 k+1=5 次 forward
  目标花了 1 次 forward
  总 forward 等效 = 5(草稿) + 1(目标) vs 直接 1.1(目标)
  → 如果草稿不够快，反而更慢！
```

**投机解码生效的三个前提**：
1. 草稿模型必须比目标模型**快很多**（至少 5×）
2. 草稿必须**准确**（接受率 > 50%，同族模型通常 70-90%）
3. 验证时的 batch size 不能太大（否则目标模型的单次 forward 也很慢）

当接受率 < 30% 时，投机解码几乎一定亏。这就是为什么草稿模型不能随便选——需要和目标模型在同一个"知识家族"里。

---

## 4. 实现细节

### `draft_speculate(draft_model, context_ids, k)`

```
context_ids (prefill) → draft_model → logits[-1] → d0
[d0] (decode)         → draft_model → logits[-1] → d1
[d1] (decode)         → draft_model → logits[-1] → d2
...共 1 prefill + k decode = k+1 draft forwards
```

- 首次 prefill 复用 KV Cache,后续每步只输入 1 个新 token
- 返回 `draft_tokens` (list[int]) 和 `draft_probs` (list[Tensor])

### `target_verify(target_model, context_ids, draft_tokens, draft_probs)`

```
输入:  [p0, p1, …, pL-1, d0, d1, d2, d3]   长度 L+k
输出:  logits[0..L+k-1]  (因果 mask 已内置)

验收:  j = 0, 1, 2, …
  target_token[j] = argmax(logits[L-1+j])
  if target_token[j] == draft_tokens[j]: 接受
  else:                                   拒绝,插入 target_token[j], break
```

**关键**: 整个验收过程只调用 1 次 `target_model.forward`,这正是投机解码的收益所在。

---

## 5. 教学版 vs 真实框架

| 维度 | 教学版 (adv03) | 真实框架 (vLLM/SGLang) |
|------|--------------|------------------------|
| 验收策略 | argmax 直接对比 | 概率比 r = p_t/p_d 采样 (无偏) |
| 输出分布 | 与 target argmax 等价 | 严格等于 target 分布 |
| 草稿模型 | 独立小模型 (d_model=2) | EAGLE/Medusa 共享骨干层 |
| KV Cache 管理 | 简化 (每轮重新 prefill) | 持续维护 draft+target 两套 KV |
| 接受率 | 随机权重极低;同族 ~70-90% | 调优后 80-95% |

### vLLM 投机解码

- [EAGLE](https://arxiv.org/abs/2401.15077): 轻量草稿头复用 target 特征层,接受率极高
- Medusa: 多个并行草稿头,同时预测多步
- vLLM `speculative_config` 一行开启

### SGLang 投机解码

- 内置 draft model + target model 双引擎协调
- 与 RadixAttention KV Cache 深度结合,减少冗余前向

---

## 6. 运行

```bash
cd advanced/adv03_speculative_decoding
python run.py
```

**预期输出:**

```
构建模型 ...
自回归生成 16 个 token ...
投机解码生成 16 个 token (k=4) ...

==========================================================
  adv03 投机解码 — Forward 次数对比 (d_model=2 草稿)
==========================================================
  生成 tokens 数           : 16
  草稿步数 k               : 4
  Target forward (AR)      : 16
  Target forward (Spec)    : 15 (随机权重,接受率低属正常)

  ✓ 投机解码结果 == 纯自回归贪婪结果 (正确性断言通过)

──────────────────────────────────────────────────────────
  Self-drafting 演示 (draft=target, 模拟 100% 接受率)
──────────────────────────────────────────────────────────
  Target forward (AR)          : 16
  Target forward (Self-draft)  : 4
  实测加速比                   : 4.0×  (理论上限 4×)

✅ adv03_speculative_decoding 通过
```

- `d_model=2` 草稿随机权重与 target 无关,接受率极低,forward 次数接近 AR
- **Self-drafting** 演示(draft = target)展示接受率 100% 时的理想加速:4×
- 正式场景中草稿来自同族小模型,接受率 70-90%,target forward 可减少 2-3×

---

## 7. 下一步

→ **adv04 Flash-Decoding**

Flash-Decoding 针对长上下文推理场景,将 KV Cache 的注意力计算拆分为多个并行块,
最后归并,使长序列下 GPU 利用率进一步提升。配合投机解码可叠加收益。

```
本章                        下一章
投机解码 (减少 forward 次数) → Flash-Decoding (加速单次 forward 的注意力计算)
```
