# adv04: Flash-Decoding（长序列 Decode 切分 + FlashInfer 简介）

## 1. 教学目标

理解 **Flash-Decoding（split-K）** 的核心思路：

- 为什么 decode 阶段单 token 对超长 KV 是显存带宽瓶颈
- 如何把长 KV 按序列方向切成多段、分配给不同 SM 并行处理
- **online softmax 归约**如何正确合并各段结果，保证数值等价
- 教学版（串行 CPU 模拟）与真实框架（FlashAttention-2 / FlashInfer）的区别

---

## 2. 问题：Decode 时单 Q 对超长 KV，显存带宽吃满

Prefill 阶段 Q/K/V 等长，GPU 的 SM 可以沿 Q 的行方向并行。
但 **decode 阶段每步只有 1 个新 token**，Q 退化为单行：

```
Q:   [1,   heads, d_head]   ← 1 个新 token
K/V: [seq, heads, d_head]   ← seq 可达几千甚至几万（长 KV Cache）
```

此时 GPU 的计算图退化为：

```
标准 decode attention（朴素实现）：

  单行 Q  ──────────────────────────────────────────→  output [1, heads, d_head]
            ↓ 顺序遍历全部 KV（seq 个 token）
          [token_0, token_1, ..., token_{seq-1}]
            ↑
            单个 SM 线性扫描，显存带宽成瓶颈
            seq 越长，耗时线性增长
```

**问题根源**：
1. 只有 1 行 Q，无法沿 Q 方向并行，大量 SM 空闲
2. KV Cache 很长，读取它的流量完全受限于显存带宽（HBM bandwidth）
3. 计算量（FLOPS）极小，GPU 不是算力瓶颈而是带宽瓶颈

---

## 3. 原理：长 KV 切多段，各段并行，online softmax 合并

### split-K 分段并行

```
长 KV 序列（seq = 4096）切成 4 段，分配给 4 组 SM 并行：

  KV Cache [0  .. 1023]  →  SM Group 0  →  局部输出 o_0, 局部最大 m_0, 局部 sum s_0
  KV Cache [1024..2047]  →  SM Group 1  →  局部输出 o_1, 局部最大 m_1, 局部 sum s_1
  KV Cache [2048..3071]  →  SM Group 2  →  局部输出 o_2, 局部最大 m_2, 局部 sum s_2
  KV Cache [3072..4095]  →  SM Group 3  →  局部输出 o_3, 局部最大 m_3, 局部 sum s_3
                                                      ↓
                                           全局 online softmax 归约
                                                      ↓
                                              最终输出 [heads, d_head]
```

### online softmax 全局归约

每段独立做局部 softmax，不知道全局最大值。归约时用以下公式对齐：

```
设各段局部结果：
  段 k：局部最大值 m_k，局部 sum s_k，局部输出 o_k（未归一化）

全局归约（以两段 A、B 为例，推广到 N 段同理）：

  Gm = max(m_A, m_B)                    ← 全局最大值

  w_A = exp(m_A - Gm)                   ← 对齐权重
  w_B = exp(m_B - Gm)

  全局分子 = o_A * w_A + o_B * w_B      ← 加权 value 输出
  全局分母 = s_A * w_A + s_B * w_B      ← 加权归一化项

  最终输出 = 全局分子 / 全局分母        ← 等价于全局 softmax
```

ASCII 图解：

```
各段独立计算（并行）              全局归约（串行，开销 O(splits)）
┌─────────────────────┐          ┌──────────────────────────────┐
│ 段0: scores_0       │          │                              │
│   → m_0, s_0, o_0  │──────→   │  Gm = max(m_0,..,m_{K-1})   │
├─────────────────────┤          │                              │
│ 段1: scores_1       │          │  for each split k:           │
│   → m_1, s_1, o_1  │──────→   │    w_k = exp(m_k - Gm)       │
├─────────────────────┤          │    Go  += o_k * w_k          │
│ ...                 │          │    Gs  += s_k * w_k          │
├─────────────────────┤          │                              │
│ 段K: scores_K       │          │  output = Go / Gs            │
│   → m_K, s_K, o_K  │──────→   │                              │
└─────────────────────┘          └──────────────────────────────┘
         并行（多 SM）                    归约（开销远小于单段）
```

归约的开销与 `num_splits` 成正比，远小于顺序扫描整段 KV 的代价。

---

## 4. 实现细节

### 朴素实现 `naive_decode_attention`

```python
def naive_decode_attention(q, K, V):
    # q: [heads, d_head]   K/V: [seq, heads, d_head]
    scores = torch.einsum('hd,shd->sh', q, K) / math.sqrt(q.size(-1))
    attn   = torch.softmax(scores, dim=0)   # 沿 seq 维度 softmax
    return torch.einsum('sh,shd->hd', attn, V)
```

顺序遍历全部 KV，单 SM 线性扫描。

### split-K 实现 `flash_decode_splitk`

```python
def flash_decode_splitk(q, K, V, num_splits=4):
    seq   = K.size(0)
    chunk = (seq + num_splits - 1) // num_splits   # 向上取整

    local_outs, local_max, local_sum = [], [], []

    # 第 1 步：各段独立局部 softmax（教学版串行，真实 GPU 并行）
    for i in range(num_splits):
        lo, hi = i*chunk, min((i+1)*chunk, seq)
        if lo >= hi: continue
        Kc, Vc = K[lo:hi], V[lo:hi]
        scores = torch.einsum('hd,shd->sh', q, Kc) / math.sqrt(q.size(-1))
        m = scores.max(dim=0).values           # 局部最大值（稳定性）
        p = torch.exp(scores - m.unsqueeze(0)) # 移位指数
        s = p.sum(dim=0)                       # 局部分母
        o = torch.einsum('sh,shd->hd', p, Vc)  # 局部加权 value
        local_outs.append(o); local_max.append(m); local_sum.append(s)

    # 第 2 步：online softmax 全局归约
    Gm = torch.stack(local_max).max(dim=0).values
    Go = torch.zeros_like(q)
    Gs = torch.zeros(q.size(0), device=q.device, dtype=q.dtype)
    for o, m, s in zip(local_outs, local_max, local_sum):
        w   = torch.exp(m - Gm)
        Go += o * w.unsqueeze(-1)
        Gs += s * w
    return Go / Gs.unsqueeze(-1)
```

关键点：

- 每段的 `m`（局部最大值）保证数值稳定，无需全局 max 就能安全计算 `exp`
- 归约时 `w = exp(m_k - Gm)` 把各段的 softmax"对齐"到同一基准
- 数学上严格等价于在全段序列上一次做 softmax（可用 `torch.allclose` 验证）

---

## 5. 教学版 vs 真实框架

### 本教学版的局限

| 方面 | 教学版（本文件） | 说明 |
|------|----------------|------|
| 执行方式 | 串行 for 循环 | 模拟分段逻辑，**无真实加速** |
| 设备 | CPU（纯 torch） | 无 CUDA 依赖，随时可运行 |
| 精度 | float32 | 真实框架通常用 float16/bfloat16 |
| batch | 无（单请求） | 真实引擎需处理批量请求 |

**单机串行无真实加速**：本教学版的 `num_splits` 越大，Python for 循环次数越多，
实际上会更慢。加速来自真实 GPU 上不同 SM 的物理并行，本文件只展示数学等价性。

### FlashAttention-2 的 Flash-Decoding

Flash-Decoding 最早由 Tri Dao 等人在 2023 年提出（[flash-decoding blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)），
作为 FlashAttention-2 的扩展，专门优化 decode 阶段的 1-to-N 注意力：

```
FlashAttention-2 decode 路径（flash_attn_with_kvcache）：

  Q [1, heads, d]  ←→  KV Cache [seq, heads, d]

  内部执行：
    1. 把 KV 按 seq 方向切成 num_splits 段
    2. 每段由一个 CUDA Thread Block 处理（物理并行）
    3. 各段结果写到临时 buffer（显存）
    4. 第二次 kernel 执行全局归约

  真实加速效果（来自原论文）：
    seq=8K：比标准 attention 快约 8×
    seq 越长，split-K 并行度越高，收益越大
```

### FlashInfer 库

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 是专为 LLM serving 设计的注意力算子库，
比 FlashAttention 在 decode 场景下有进一步优化：

```
FlashInfer 的主要特点：

  1. Cascade Attention（级联注意力）
     把 KV 按"稳定前缀"和"动态后缀"分两层处理
     前缀可跨请求共享，减少重复计算

  2. 分页 KV Cache 原生支持
     直接传入 block_table，kernel 内部处理非连续物理内存
     避免 gather 步骤（类似 flash_attn_with_kvcache 的 block_table 参数）

  3. 各阶段算子融合
     Prefill / Decode / Append KV 三个阶段各有优化 kernel
     不同请求混批（Continuous Batching）时显著减少 kernel 启动次数

  4. 分段 softmax（split-K）在更细粒度上实现
     根据 seq 长度动态选择 splits，避免短序列时归约开销超过收益

  对比本教学版：
    FlashInfer decode attention ≈ flash_decode_splitk（数学等价）
    但实现在 CUDA kernel 内部，无 Python 循环开销，支持 float16/bf16
```

```
接口示意（非本教学版可运行代码）：

  import flashinfer
  # 单步 decode
  output = flashinfer.single_decode_with_kv_cache(
      q,          # [num_heads, head_dim]
      k_cache,    # [seq_len, num_kv_heads, head_dim]
      v_cache,    # [seq_len, num_kv_heads, head_dim]
  )
```

---

## 6. 运行

```bash
cd advanced/adv04_flash_decoding
python run.py
```

预期输出（纯 CPU，无需 CUDA）：

```
============================================================
adv04_flash_decoding 正确性验证
============================================================

[1] 基础用例（seq 可整除 splits）
  seq=  128  heads=8  d_head=64  splits=4  max_diff=...e-0x  [PASS]
  seq=  512  heads=8  d_head=64  splits=4  max_diff=...e-0x  [PASS]
  seq= 1024  heads=8  d_head=64  splits=8  max_diff=...e-0x  [PASS]

[2] seq 不能整除 splits（最后一段更短）
  ...

[3] 极端情况：splits=1
  ...

[4] 极端情况：splits > seq
  ...

[5] 较长序列，模拟 decode 长 KV Cache 场景
  seq= 4096  heads=16  d_head=128  splits=16  max_diff=...e-0x  [PASS]
  seq= 8192  heads=16  d_head=128  splits=32  max_diff=...e-0x  [PASS]

============================================================

✅ adv04_flash_decoding 通过
```

所有用例均断言 `torch.allclose(atol=1e-5)`，确认 split-K 与 naive 数值等价。

---

## 7. 下一步

**adv05: Radix Tree KV Cache + Copy-on-Write（Prefix Sharing）**

Flash-Decoding 解决了单个请求 decode 时 SM 利用率低的问题。
但当多个请求共享相同前缀（系统 prompt、few-shot 示例）时，
KV Cache 中存储了大量重复内容，浪费显存。

adv05 将介绍：
- **Radix Tree** 如何按 token 前缀共享 KV Cache block
- **Copy-on-Write（CoW）** 如何在请求分叉时延迟复制，避免不必要的内存拷贝
- 与 vLLM 的 `prefix_caching` 功能的对应关系
