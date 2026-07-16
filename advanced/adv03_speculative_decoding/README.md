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
