# step14_7 — Batched Forward：所有请求一次 forward，真正的批处理

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

### 为什么 Linear 层可以直接批量？

`nn.Linear` 的本质是矩阵乘法：

```
output = input @ W^T + b
形状：[N, in_features] @ [in_features, out_features] → [N, out_features]
```

这里 `N` 是 token 数量。Linear 层**不关心 N 行 token 属于哪几个序列**——每行 token 的计算完全独立，不需要跨 token 的信息（attention 才需要）。

```python
# 逐序列（Paged Prefix Cache 的做法）：
seq_A_out = W(seq_A_x)   # [3, d_model] @ W^T → [3, d_model]，3 tokens
seq_B_out = W(seq_B_x)   # [1, d_model] @ W^T → [1, d_model]，1 token
seq_C_out = W(seq_C_x)   # [2, d_model] @ W^T → [2, d_model]，2 tokens
# 3 次独立的矩阵乘法，3 次 kernel launch

# 批量（step14_7 的做法）：
all_x   = torch.cat([seq_A_x, seq_B_x, seq_C_x], dim=0)  # [6, d_model]
all_out = W(all_x)   # [6, d_model] @ W^T → [6, d_model]
# 1 次矩阵乘法，1 次 kernel launch，结果完全等价
```

`all_out[:3]` 等于 `seq_A_out`，`all_out[3:4]` 等于 `seq_B_out`，以此类推——**结果完全相同**，因为每行的计算不依赖其他行。

### MLP 同理

MLP 由两个 Linear 组成：

```python
# step14_7 的 MLP forward（layer 是 PagedTransformerDecoderLayer）
x = x + layer.mlp(layer.norm2(x))   # x 形状 [total_tokens, d_model]
```

`norm2` 是 RMSNorm（逐 token 归一化，行间独立），`mlp` 内部是两个 `nn.Linear`，同样对 token 维度完全独立。整个 MLP 对 `[total_tokens, d_model]` 的输入直接批量执行，无需改动。

### Embedding 的批量

```python
x = self.embed(token_ids)   # [total_tokens] → [total_tokens, d_model]
```

`nn.Embedding` 本质是查表（indexed select），每个 token ID 独立查表，天然支持任意长度的 flat batch。

### cu_seqlens 只在 Attention 里才需要

拼接后的 `[total_tokens, d_model]` 对 Linear/Embedding 完全透明——它们不知道也不需要知道 token 的序列归属。

只有 **Attention** 需要 `cu_seqlens`：因果 mask 要求同一序列内的 token 才有注意力关系，跨序列的 token 不能互相 attend：

```
tokens = [t0_A, t1_A, t2_A, t3_B, t4_C, t5_C]

Linear/Embedding：全部一起算，不管序列边界
Attention：       按 cu_seqlens=[0,3,4,6] 分段
                  A 的 token 只看 A，B 只看 B，C 只看 C
```

这也是为什么 `forward_batched` 里 Linear 层在循环外，Attention 在循环内：

```python
# model.py — forward_batched 核心结构
x = self.embed(token_ids)              # ← 循环外，批量

for layer_idx, layer in enumerate(self.layers):
    normed = layer.norm1(x)            # ← 循环外，批量
    Q = attn.W_q(normed)              # ← 循环外，批量 QKV 投影
    K = attn.W_k(normed)
    V = attn.W_v(normed)

    attn_outputs = []
    for i in range(num_seqs):          # ← 循环内，按序列做 attention
        ...
        attn_outputs.append(attn.W_o(out_i))

    x = x + torch.cat(attn_outputs)   # ← W_o 在循环内，待优化
    x = x + layer.mlp(layer.norm2(x)) # ← 循环外，批量 MLP
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

Attention 部分由于每条序列有不同的 block_table 和 start_pos，仍需逐序列处理（用 `flash_attn_varlen_func` 才能彻底消除，见 FlashAttention 章节）。

## 收益分解

模型计算量中 Linear 层（Embedding + QKV + O + MLP + lm_head）约占 **85%**。把这部分合并成一次大矩阵乘，已经能带来显著提升。

| 组件 | Paged Prefix Cache | step14_7 |
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
