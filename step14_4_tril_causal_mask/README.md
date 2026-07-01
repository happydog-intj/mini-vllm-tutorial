# step14_4 — torch.tril causal mask：消除逐行 Python 循环

## 问题

`Paged Prefix Cache` 每次 forward 都用 Python 循环逐行构造 causal mask：

```python
mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device)
for i in range(seq_len):
    mask[i, :start_pos + i + 1] = False
scores = scores.masked_fill(mask, float("-inf"))
```

**问题：**
1. `seq_len` 次 Python 循环，每次一次 tensor slice 赋值
2. 每次 forward 都重新 allocate 一个 `[seq_len, total_len]` 的 bool tensor
3. decode 阶段 `seq_len=1`，整个 mask 全是 `False`，构造和 masked_fill 都在浪费时间

## 解决方案：broadcast 比较，一次生成

causal mask 的本质是：位置 `(i, j)` 被 mask 当且仅当 `j > start_pos + i`。
用 broadcast 一次生成，无需 Python 循环：

```python
q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)    # [seq_len, 1]
k_idx = torch.arange(total_len, device=x.device).unsqueeze(0)  # [1, total_len]
causal_mask = k_idx > (start_pos + q_idx)                       # [seq_len, total_len]
scores = scores.masked_fill(causal_mask, float("-inf"))
```

**decode 阶段的特殊优化：**

decode 时 `seq_len=1`，当前 token 可以 attend 到所有历史，mask 全为 `False`，直接跳过：

```python
if seq_len == 1:
    # decode：无需 mask
    weights = torch.softmax(scores, dim=-1)
else:
    # prefill：构造 causal mask
    q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
    k_idx = torch.arange(total_len, device=x.device).unsqueeze(0)
    causal_mask = k_idx > (start_pos + q_idx)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
```

## Python 循环次数对比

| | Paged Prefix Cache | step14_4 |
|---|---|---|
| mask 构造 | seq_len 次 Python 循环 | 0 次（纯 tensor broadcast）|
| tensor 分配 | 每次 forward 1 次 | 1 次（可进一步缓存复用）|
| decode 特殊处理 | 无（seq_len=1 也走循环）| 跳过整个 mask 操作 |

## 与 vLLM 的对比

vLLM 使用 FlashAttention，causal mask 在 kernel 内部隐式处理，完全不需要显式构造 mask tensor，也不产生任何额外显存分配。`is_causal=True` 参数告知 kernel 使用下三角掩码即可。

本章是在保持标准 softmax attention 的前提下，消除 Python 循环的最低成本方案，也是通向 `F.scaled_dot_product_attention(is_causal=True)` 的过渡。

## 实现

见 `model.py` — `PagedMultiHeadAttention.forward` 中 causal mask 构造部分。

## 运行

```bash
python run.py
```
