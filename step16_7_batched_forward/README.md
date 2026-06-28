# step16_7 — Batched Forward：所有请求一次 forward，真正的批处理

## 问题

`Paged Prefix Cache` 对每个序列单独调用一次 `model()`：

```python
for seq in prefill_seqs:
    self._do_prefill_step(seq)   # model(chunk, ...) — 1 次 forward

for seq in decode_seqs:
    self._do_decode_step(seq)    # model([token], ...) — 1 次 forward
```

假设同时有 8 个 decode 请求：
- 当前：8 次 `model([1 token])` forward
- 期望：1 次 `model([8 tokens])` forward

**8 个 `[1, d_model]` 的矩阵乘法 vs 1 个 `[8, d_model]` 的矩阵乘法：**

GPU 的矩阵乘法吞吐量在 batch_size=1 时约是 batch_size=8 的 1/10——不是因为计算量不同，而是因为 kernel launch 开销固定、硬件利用率极低。这是 step14 与真实 vLLM 最大的性能差距来源。

## 解决方案：变长 batch（varlen）

将所有请求的 token 拼接成一个 flat batch，用 `cu_seqlens` 标记边界：

```
请求A（prefill chunk 3 tokens）: [t0, t1, t2]
请求B（decode 1 token）:         [t3]
请求C（prefill chunk 2 tokens）: [t4, t5]

拼接：tokens = [t0, t1, t2, t3, t4, t5]   # [total_tokens]
cu_seqlens = [0, 3, 4, 6]                  # [num_seqs + 1]
```

**Linear 层真正批量：**

```python
# Embedding + W_q/W_k/W_v/W_o/MLP/lm_head 全部变成一次大矩阵乘
x = self.embed(tokens)          # [total_tokens, d_model] — 1 次，而非 num_seqs 次
Q = self.W_q(x)                 # [total_tokens, d_model] — 1 次
...
```

**Attention 按序列分段：**

```python
outputs = []
for i in range(num_seqs):
    start, end = cu_seqlens[i], cu_seqlens[i + 1]
    seq_q = Q[start:end]   # 切片，无拷贝
    K_full = gather_kv_from_blocks(kv_pool_k, block_tables[i], total_len_i, block_size)
    V_full = gather_kv_from_blocks(kv_pool_v, block_tables[i], total_len_i, block_size)
    out_i = attention(seq_q, K_full, V_full, start_positions[i])
    outputs.append(out_i)

x = torch.cat(outputs, dim=0)  # [total_tokens, d_model]
```

Attention 部分由于每条序列有不同的 block_table 和 start_pos，仍需逐序列处理（用 `flash_attn_varlen_func` 才能彻底消除，见 `step16_flash_attention`）。

## 收益分解

模型计算量中 Linear 层（Embedding + QKV + O + MLP + lm_head）约占 **85%**。把这部分合并成一次大矩阵乘，已经能带来显著提升。

| 组件 | Paged Prefix Cache | step16_7 |
|------|-----------|---------|
| Embedding | num_seqs 次 | 1 次 |
| W_q/W_k/W_v Linear | num_seqs 次小矩阵乘 | 1 次大矩阵乘 |
| MLP（2 个 Linear）| num_seqs 次 | 1 次 |
| LM head | num_seqs 次 | 1 次 |
| Attention | num_seqs 次 | num_seqs 次（待 flash_attn_varlen 优化）|
| kernel launch 总数 | O(num_seqs × layers × 6) | O(layers × 6 + num_seqs) |

## 与 vLLM 的对比

vLLM（nano-vllm）使用 `flash_attn_varlen_func`：
- 把所有序列拼成一个 batch，连 attention 的逐序列循环也消除
- kernel 内部按 `cu_seqlens` 分段，直接按 paged block_table 访问 kv_pool
- 整个 forward 是真正意义上的"一次 kernel 调用"

本章实现了 Linear 层的批量，是通向 `flash_attn_varlen` 的必经步骤。

## 实现

见 `model.py` — `TinyTransformerPaged.forward_batched`（新增方法）；`engine.py` — `generate_batch` 调度循环改为收集所有 seq 后统一调用 `forward_batched`。

## 运行

```bash
python run.py
```
