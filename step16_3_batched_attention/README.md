# step16_3 — 批量 Attention：消除逐 head 的 Python 循环

## 问题

`Paged Prefix Cache` 对每个注意力头单独做 matmul，用 Python `for` 循环串行执行：

```python
outputs = []
for h in range(self.num_heads):
    q_h = Q[:, h, :]       # [seq_len, d_head]
    k_h = K_full[:, h, :]  # [total_len, d_head]
    v_h = V_full[:, h, :]

    scores = torch.matmul(q_h, k_h.T) / math.sqrt(self.d_head)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out_h = torch.matmul(weights, v_h)
    outputs.append(out_h)

concat = torch.cat(outputs, dim=-1)
```

**性能代价：**
- `num_heads` 次 Python 循环（典型值 8~32）
- `3 × num_heads` 次独立 kernel launch（matmul + softmax + matmul）
- 每个 head 的矩阵很小（`[seq_len, d_head]`），单独 launch kernel GPU 利用率极低
- `torch.cat` 额外一次内存拷贝

## 解决方案：batch matmul，一次处理所有 head

```python
# Q/K/V reshape 到 [num_heads, seq/total_len, d_head]
Q_t = Q.transpose(0, 1)      # [num_heads, seq_len, d_head]
K_t = K_full.transpose(0, 1) # [num_heads, total_len, d_head]
V_t = V_full.transpose(0, 1) # [num_heads, total_len, d_head]

# 一次 bmm 完成所有 head 的 QK^T
scores = torch.bmm(Q_t, K_t.transpose(1, 2)) / math.sqrt(self.d_head)
# scores: [num_heads, seq_len, total_len]

scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
weights = torch.softmax(scores, dim=-1)

# 一次 bmm 完成所有 head 的加权求和
out = torch.bmm(weights, V_t)             # [num_heads, seq_len, d_head]
out = out.transpose(0, 1).reshape(seq_len, -1)  # [seq_len, d_model]
```

**关键变化：**
- `num_heads` 次循环 → **0 次** Python 循环
- `3 × num_heads` 次 kernel launch → **3 次**
- GPU 看到的矩阵更大（`[num_heads, seq_len, d_head]`），硬件并行度更高
- 消除 `torch.cat`

## 更进一步：F.scaled_dot_product_attention

PyTorch 2.0+ 提供融合实现，一次 kernel 完成 QK^T + softmax + 乘 V：

```python
import torch.nn.functional as F

# [num_heads, seq_len, d_head]
Q_t = Q.transpose(0, 1).unsqueeze(0)
K_t = K_full.transpose(0, 1).unsqueeze(0)
V_t = V_full.transpose(0, 1).unsqueeze(0)

out = F.scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=(start_pos == 0))
out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
```

在有 FlashAttention 支持时自动使用 fused kernel，是通向 `step16_flash_attention` 的直接过渡。

## 与 vLLM 的对比

| | Paged Prefix Cache | step16_3（bmm）| vLLM |
|---|---|---|---|
| Attention 计算 | num_heads 次循环 | 1 次 bmm | FlashAttention kernel（fused，O(n) HBM）|
| kernel launch 数 | 3×num_heads | 3 | 1 |
| 中间矩阵显存 | 逐头分配 | batch 一次分配 | 不写回 HBM（SRAM-only）|

## 实现

见 `model.py` — `PagedMultiHeadAttention.forward` attention 计算部分。

## 运行

```bash
python run.py
```
