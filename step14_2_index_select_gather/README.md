# step14_2 — index_select gather：消除 KV 读取的 Python 循环

## 问题

`Paged Prefix Cache` 的 `gather_kv_from_blocks` 用 Python 循环逐 block 拼接 K/V：

```python
def gather_kv_from_blocks(pool, block_table, seq_len, block_size):
    chunks = []
    remaining = seq_len
    for block_id in block_table:          # Python 循环，len(block_table) 次
        if remaining <= 0:
            break
        slots = min(block_size, remaining)
        chunks.append(pool[block_id, :slots])
        remaining -= slots
    return torch.cat(chunks, dim=0)       # 额外一次 cat（内存拷贝）
```

**性能代价：**
- `len(block_table)` 次 Python 循环（= `ceil(seq_len / block_size)` 次）
- 每次 `pool[block_id, :slots]` 是一次独立的 GPU 内存访问
- `torch.cat` 触发额外的内存分配和拷贝
- decode 阶段序列很长时 block_table 可达几十个，循环代价明显

## 解决方案：构造扁平化索引，一次 advanced indexing

```python
def gather_kv_from_blocks_v2(
    pool: Tensor,           # [total_blocks, block_size, num_heads, d_head]
    block_table: List[int],
    seq_len: int,
    block_size: int,
) -> Tensor:
    positions = torch.arange(seq_len, device=pool.device)
    block_indices = positions // block_size      # [seq_len]
    slot_indices  = positions % block_size       # [seq_len]

    bt = torch.tensor(block_table, device=pool.device)
    physical_blocks = bt[block_indices]          # [seq_len]

    # 一次 advanced indexing，无 Python 循环，无 torch.cat
    return pool[physical_blocks, slot_indices]   # [seq_len, num_heads, d_head]
```

**关键变化：**
- Python 循环从 `num_blocks` 次降为 **0 次**
- 消除 `torch.cat`，结果直接在一块连续内存中
- 底层对应一次 CUDA gather kernel

## 与 step14_1 的关系

step14_1（写入）和 step14_2（读取）用的是同一套思路：**把逐元素的 Python 循环转成 tensor 索引操作**。

写入：`kv_pool[physical_blocks, slot_indices] = K`（scatter）
读取：`pool[physical_blocks, slot_indices]`（gather）

## 与 vLLM 的对比

| | Paged Prefix Cache | step14_2 | vLLM |
|---|---|---|---|
| gather 方式 | Python 循环 + torch.cat | advanced indexing | paged attention kernel（fused gather + attention）|
| Python 循环 | num_blocks 次 | 0 次 | 0 次 |
| 额外内存拷贝 | torch.cat | 无 | 无（kernel 内直接访问）|

vLLM 走得更远：FlashAttention paged kernel 在计算 attention 时直接按 block_table 访问 kv_pool，连 gather 这一步都消除了——K/V 从未被拷贝到连续内存，attention scores 直接在分散的 block 上累积。本章是迈向这个方向的第一步。

## 实现

见 `model.py` — `gather_kv_from_blocks` 函数替换为向量化版本。

## 运行

```bash
python run.py
```
