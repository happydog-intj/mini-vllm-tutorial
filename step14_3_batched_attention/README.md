# step14_3 — 批量 Attention：消除逐 head 的 Python 循环

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

## torch.bmm 介绍

### 接口

```python
torch.bmm(input, mat2) → Tensor
```

- `input`：形状 `[batch, n, m]`
- `mat2`：形状 `[batch, m, p]`
- 输出：形状 `[batch, n, p]`

对 batch 维度的每一个矩阵独立做矩阵乘法，**batch 维度上的所有计算并行执行**：

```
output[i] = input[i] @ mat2[i]   ← 对所有 i 同时计算
```

### 与 torch.matmul 的区别

| | `torch.matmul` | `torch.bmm` |
|---|---|---|
| 输入形状 | 任意维度，支持广播 | 严格要求 3D，无广播 |
| batch 支持 | ✅（自动广播） | ✅（严格逐一对应）|
| 语义 | 通用矩阵乘，规则复杂 | 明确的批量矩阵乘 |
| 性能 | 相同 | 相同（底层同一 CUDA kernel）|

在 Attention 计算里，batch 维度就是 `num_heads`，bmm 语义和代码意图完全一致，优先用 bmm。

### 为什么 bmm 比逐 head 循环快？

**1. kernel launch 次数**

```
逐 head 循环：num_heads 次 matmul → num_heads 次 kernel launch
bmm：         1 次 kernel launch
```

每次 kernel launch 有固定的 CPU→GPU 调度开销（约 5~20μs）。`num_heads=8` 时循环版光 launch 开销就比 bmm 多 7 次。

**2. GPU 硬件利用率**

单个 head 的矩阵很小（decode 时 `[1, d_head]`），单独 launch 时 GPU 大量 CUDA core 空闲：

```
逐 head：每次只给 GPU 一个 [1, 64] 的矩阵，CUDA core 利用率极低
bmm：    一次给 GPU [num_heads, 1, 64] 的 3D 张量，所有 head 并行，利用率更高
```

**3. 内存访问模式**

逐 head 循环每次处理一小块数据，cache 频繁失效；bmm 把所有 head 的数据连续布局，一次加载，cache 更友好。

### 形状变换

Attention 里用 bmm 需要把 `[seq_len, num_heads, d_head]` 转成 `[num_heads, seq_len, d_head]`：

```python
Q_t = Q.transpose(0, 1)   # [seq_len, num_heads, d_head] → [num_heads, seq_len, d_head]
```

`.transpose(0, 1)` 返回的是原 tensor 的 **view**（不复制数据，只改 stride），所以这步几乎没有开销。

最后再把结果 reshape 回来：

```python
out = out.transpose(0, 1).reshape(seq_len, -1)
# [num_heads, seq_len, d_head] → [seq_len, num_heads, d_head] → [seq_len, d_model]
```

`reshape` 在内存连续时也是 view，不复制数据。

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

在有 FlashAttention 支持时自动使用 fused kernel，是通向 FlashAttention 的直接过渡。

## 与 vLLM 的对比

| | Paged Prefix Cache | step14_3（bmm）| vLLM |
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
