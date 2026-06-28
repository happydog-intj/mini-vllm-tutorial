# step16_1 — 向量化 KV 写入：消除 Python 逐 token 循环

## 问题

`step14` 的 KV 写入是一个 Python `for` 循环，每次迭代写入一个 token 的 K/V：

```python
# model.py — 每次 forward 都执行 seq_len 次 Python 循环
for i in range(seq_len):
    pos = start_pos + i
    block_idx    = pos // block_size
    slot_in_block = pos % block_size
    physical_block = block_table[block_idx]
    kv_pool_k[physical_block, slot_in_block] = K[i]
    kv_pool_v[physical_block, slot_in_block] = V[i]
```

**性能代价：**
- `seq_len` 次 Python 循环 → `seq_len` 次 Python 解释器开销
- 每次赋值触发一次 CUDA kernel launch（scalar write）
- prefill 阶段 seq_len 可达几百甚至几千，循环代价随之线性增长

## 解决方案：预计算物理槽位索引，一次 scatter 写入

将所有 token 的物理槽位索引提前算好，用向量化操作一次完成写入：

```python
def write_kv_to_pool(
    kv_pool_k: Tensor,       # [total_blocks, block_size, num_heads, d_head]
    kv_pool_v: Tensor,
    K: Tensor,               # [seq_len, num_heads, d_head]
    V: Tensor,
    block_table: List[int],
    start_pos: int,
    block_size: int,
):
    seq_len = K.size(0)
    positions = torch.arange(start_pos, start_pos + seq_len, device=K.device)

    block_indices = positions // block_size          # [seq_len]
    slot_indices  = positions % block_size           # [seq_len]

    # block_table 转为 tensor 做向量化查找
    bt = torch.tensor(block_table, device=K.device)
    physical_blocks = bt[block_indices]              # [seq_len]

    # 一次性写入：不再有 Python 循环
    kv_pool_k[physical_blocks, slot_indices] = K     # [seq_len, num_heads, d_head]
    kv_pool_v[physical_blocks, slot_indices] = V
```

**关键变化：**
- `block_indices`、`slot_indices`、`physical_blocks` 全部是 tensor，在 GPU 上并行计算
- 最终的 `kv_pool_k[physical_blocks, slot_indices] = K` 是 PyTorch 的 advanced indexing，底层一次 scatter kernel
- Python 循环从 `seq_len` 次降为 **0 次**

## 与 vLLM 的对比

| | step14 | step16_1 | vLLM (nano-vllm) |
|---|---|---|---|
| 写入方式 | Python for 循环 | advanced indexing | Triton `reshape_and_cache` kernel |
| kernel launches | seq_len 次 | 1 次 | 1 次（fused） |
| Python 循环 | seq_len 次 | 0 次 | 0 次 |

vLLM 使用自定义 Triton kernel 是因为它同时处理多个序列的 batch，需要更复杂的索引结构（`slot_mapping` 是跨所有序列的扁平化槽位列表）。本章的 advanced indexing 方案已经消除了 Python 循环，是教学实现能达到的最简洁形式。

## 实现

见 `model.py` — `PagedMultiHeadAttention.forward` 中的 `write_kv_to_pool` 调用。

## 运行

```bash
python run.py
```
