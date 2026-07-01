# step14_1 — 向量化 KV 写入：消除 Python 逐 token 循环

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

## PyTorch Advanced Indexing 基础

理解这章的关键是理解 PyTorch 的两种索引方式：

### Basic Indexing vs Advanced Indexing

**Basic Indexing**：用整数或 slice 索引，返回原 tensor 的一个视图（view），不复制数据：

```python
x = torch.tensor([[1,2],[3,4],[5,6]])
x[0]        # [1, 2] — 取第0行，view
x[1:3]      # [[3,4],[5,6]] — slice，view
```

**Advanced Indexing**：用 **tensor** 作为索引，返回新 tensor（复制数据），可以一次取出任意位置的元素：

```python
x = torch.tensor([[1,2],[3,4],[5,6]])
idx = torch.tensor([2, 0, 2])   # 索引是一个 tensor
x[idx]      # [[5,6],[1,2],[5,6]] — 按 idx 指定的行依次取出，可以重复
```

关键特性：**`idx` 里写什么顺序，输出就是什么顺序，可以重复，可以乱序**。

### 多维 Advanced Indexing

当有多个 tensor 索引时，它们在对应维度上**逐元素配对**：

```python
pool = torch.zeros(8, 16, 4, 32)  # [total_blocks, block_size, num_heads, d_head]
physical_blocks = torch.tensor([3, 3, 5])  # [seq_len=3]
slot_indices    = torch.tensor([0, 1, 2])  # [seq_len=3]

# 逐元素配对：
# 结果[0] = pool[3, 0, :, :]
# 结果[1] = pool[3, 1, :, :]
# 结果[2] = pool[5, 2, :, :]
pool[physical_blocks, slot_indices]  # [3, num_heads, d_head]
```

这和 Python 循环等价：
```python
# 等价的循环写法
[pool[physical_blocks[i], slot_indices[i]] for i in range(3)]
```

### 写入（Scatter）也是同样语法

Advanced indexing 不只能读，也能写：

```python
K = torch.randn(3, 4, 32)  # [seq_len, num_heads, d_head]

# 把 K[0] 写入 pool[3, 0]，K[1] 写入 pool[3, 1]，K[2] 写入 pool[5, 2]
pool[physical_blocks, slot_indices] = K
```

这就是本章的核心：用一行赋值替代 `seq_len` 次 Python 循环。

### 为什么需要先把 block_table 转成 tensor？

```python
bt = torch.tensor(block_table, device=K.device)
physical_blocks = bt[block_indices]   # tensor[tensor] → advanced indexing
```

`block_table` 是 Python `List[int]`，不能直接被 tensor 索引。转成 tensor 后，`bt[block_indices]` 才是 advanced indexing，在 GPU 上并行执行。如果用 Python list 索引 tensor，PyTorch 会退化为逐元素的 Python 循环。

### 底层执行

```
pool[physical_blocks, slot_indices] = K
         ↓
PyTorch 构造一次 scatter kernel launch
         ↓
CUDA 在 GPU 上并行执行所有写入
（seq_len 个写操作同时进行，没有串行依赖）
```

对比原来的 Python 循环：每次 `kv_pool_k[physical_block, slot_in_block] = K[i]` 都是一次独立的 CUDA kernel launch，`seq_len` 次循环就是 `seq_len` 次 launch，每次 launch 有固定的 CPU→GPU 调度开销（约 5~20μs），这些开销全部叠加。

## 与 vLLM 的对比

| | step14 | step14_1 | vLLM (nano-vllm) |
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
