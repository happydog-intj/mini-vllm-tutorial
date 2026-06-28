# Paged Prefix Cache — Paged Prefix Cache

## 阶段性小结
前面我们讲了Continuous Batching，chunked prefill，paged attention，prefix cache等一系列的优化，每个章节为了简化学习和理解成本，都做了一定的简化，其中一个简化就是kv cache并没有真正的做到全局唯一，且在显存中复用；另外一个简化就是控制单一变量，每章只实现这章要讲的概念，其他章的优化并没有一起串起来，这就导致我们前面的实现还是比较偏demo性能，和实际vllm的实现有较大的差距。本章的目的，就是把前面所有章节的优化全部结合起来，真正做到一个接近生产级的实现。

## 教学目标

把 [PagedAttention：分页内存管理 的分页内存管理](../step12_paged_attention/README.md) 和 [Prefix Caching：相同前缀只算一次 的前缀缓存](../step13_prefix_cache/README.md) 结合起来，实现正确的、Block 粒度的前缀缓存：

- 以 **Block 为单位**存储前缀 KV 快照（而非整段 past_key_values）
- 命中时 `ref_count++`，直接复用，物理显存零拷贝
- 未命中时**逐 Block 增量 prefill**，在每个 Block 边界处保存**正确的** KV 快照
- 释放时 `ref_count--`，归零才真正回收

## Prefix Caching：相同前缀只算一次 的两个核心问题

Prefix Caching：相同前缀只算一次 的实现能跑通，但有两个根本性缺陷：

**问题1：缓存了错误的 KV 快照**

```python
# Prefix Caching：相同前缀只算一次 的做法：prompt 全部跑完后按照block计算hash并缓存
for end in range(block_size, prompt_len + 1, block_size):
    h = compute_hash(tokens, end)
    prefix_cache[h] = past_kv   # ← past_kv 含完整 prompt 的信息，不是前 end 个 token！
```

`past_kv` 是整个 prompt 跑完后的状态，包含了 token `end` 之后的所有信息。下次命中 `h` 时，拿到的 KV 包含了"未来"信息，在真实模型上会产生错误输出。

**问题2：无法跨请求共享物理 Block**

Prefix Caching：相同前缀只算一次 缓存的是 Python 对象（`past_key_values` tuple），每个请求命中后得到的是对同一对象的引用，没有 `ref_count` 控制生命周期——无法安全地在多请求之间共享并释放。

## Paged Prefix Cache 的修正：逐 Block 增量 prefill

正确做法是**在 prefill 过程中**，每处理完一个完整 Block 就立即保存快照：

```python
# block_table 已提前分配好（命中的 cached_block_ids + 新分配的 block）
pos = cached_len   # 从命中断点继续（未命中则从 0 开始）

while pos < prompt_len:
    end = min(pos + block_size, prompt_len)
    chunk = prompt_ids[pos:end]
    logits = model(chunk, block_table=block_table, start_pos=pos)
    # ↑ model 内部把 chunk 的 K/V 写入 kv_pool[block_table[...]]
    pos = end

    # 只在完整 Block 边界处缓存 block_id 列表快照
    if pos % block_size == 0:
        h = chain_hash(tokens, prev_hash, pos - block_size, pos)
        if h not in prefix_cache:
            prefix_block_ids = block_table[:pos // block_size]  # 前 pos 个 token 的物理 Block
            prefix_cache[h] = {
                "block_ids": list(prefix_block_ids),  # ← 缓存 block_id 列表，不是 past_kv
                "length":    pos,
            }
        prev_hash = h
```

此时 `past_kv` 只包含 `tokens[0:pos]` 的 KV，是真正意义上的"前 pos 个 token 的缓存"。

### 与 OS 虚拟内存的完整类比

Paged Prefix Cache 的设计和 OS 内存管理高度对应：

| 操作系统 | Paged Prefix Cache KV Cache |
|---------|----------------|
| 进程创建 → 分配虚拟地址空间 + 建页表 | 请求开始 → allocate() + 建 block_table |
| 页表：虚拟页 → 物理页帧 | block_table：逻辑 token 位置 → 物理 Block ID |
| 进程运行 → 按需写入物理页 | prefill/decode → 按顺序写入 kv_pool |
| 进程退出 → 归还页帧 | 请求结束 → free(block_table) |
| `fork()` 后父子进程共享物理页（CoW）| prefix cache 命中 → 多请求共享同一物理 Block |
| 写时复制（CoW）：写操作才分配新页 | prefix cache 只读共享：新 token 才分配新 Block |

最后一行的 CoW 类比最精妙：OS 中 `fork()` 后父子进程的页表指向同一物理页，只要没有写操作就不复制（`ref_count` 控制）；Paged Prefix Cache 中多个请求命中同一前缀时，`block_table` 指向同一物理 Block，只要不写新 token 就不分配新 Block——`ref_count++` 即可，物理显存零拷贝。

### 顺带实现了 Chunked Prefill

逐 Block 增量 prefill 的循环结构和 [Chunked Prefill：切片长 Prompt 的 Chunked Prefill](../step10_chunked_prefill/README.md) 完全相同——都是把长 prompt 切成小块依次处理：

```python
while pos < prompt_len:
    chunk = prompt_ids[pos : pos + block_size]   # chunk_size = block_size
    logits, past_kv = model(chunk, past_kv=past_kv)
    pos += len(chunk)
```

两者解决的问题不同，但机制一致：

| | Chunked Prefill：切片长 Prompt Chunked Prefill | Paged Prefix Cache 逐 Block Prefill |
|---|---|---|
| 切块目的 | 不阻塞 decode，每块后让 decode 执行一步 | 在边界处保存正确的 KV 快照 |
| chunk_size 由什么决定 | 调度参数（如 512） | block_size（如 16） |
| 副作用 | 顺带可以在边界缓存 KV | 顺带避免长 prefill 独占 GPU |

真实 vLLM 把两者合并为同一个机制：chunk_size = block_size，每次处理一个 Block，既满足 prefix cache 的边界对齐需求，又天然支持与 decode 交替执行。

## ref_count 管理

每个缓存 Block 有 `ref_count` 控制生命周期：

```
缓存写入时：   block_manager.allocate(1)  → ref_count = 1
命中时：       block.ref_count += 1       → 当前请求持有引用
请求结束时：   block.ref_count -= 1       → 归零则可驱逐
```

多个请求同时命中同一前缀时：

```
请求A 命中 block_id=5：ref_count = 2（缓存1份 + 请求A1份）
请求B 命中 block_id=5：ref_count = 3（缓存1份 + 请求A + 请求B）
请求A 结束：           ref_count = 2  ← 不回收，请求B 还在用
请求B 结束：           ref_count = 1  ← 不回收，缓存还持有
（驱逐时）：           ref_count = 0  ← 才真正回收，放回空闲池
```

## Prefix Caching：相同前缀只算一次 vs Paged Prefix Cache 对比

| 特性 | Prefix Caching：相同前缀只算一次 | Paged Prefix Cache（本章） |
|------|--------|---------------|
| KV 快照正确性 | ❌ 含整个 prompt 信息 | ✅ 只含前 N 个 token |
| 缓存粒度 | 整段 past_key_values | Block 粒度 |
| 跨请求共享 | ❌ 无 ref_count | ✅ ref_count 控制 |
| 驱逐支持 | ❌ 永不驱逐 | ✅ ref_count 归零可驱逐 |
| 增量 prefill | ❌ 全量后缓存 | ✅ 逐 Block 边界缓存 |
| past_kv 存储 | Python 对象（游离） | ❌ 彻底消失 |
| KV 数据托管 | ❌ Python 堆内存 | ✅ kv_pool 张量（BlockManager 管理） |
| prefix cache 命中 | 复制 Python 对象引用 | ✅ block_table 复用，零拷贝 |

## TinyTransformerPaged

`model.py` 实现了 `TinyTransformerPaged`，把 `past_key_values` 彻底替换为 `kv_pool + block_table`：

```python
# TinyTransformerWithKVCache（单请求 KV Cache~Prefix Caching：相同前缀只算一次）：
logits, past_kv = model(chunk, past_key_values=past_kv)  # past_kv 是 Python 对象，每步返回

# TinyTransformerPaged（本章）：
logits = model(chunk, block_table=block_table, start_pos=pos)  # KV 直接写入 kv_pool，无返回值
```

`kv_pool` 作为模型的 `register_buffer` 注册，每层各有一对：

```python
self.kv_pool_k = torch.zeros(num_layers, total_blocks, block_size, num_heads, d_head)
self.kv_pool_v = torch.zeros(num_layers, total_blocks, block_size, num_heads, d_head)
```

### 为什么 kv_pool 的形状是 `[total_blocks, block_size, num_heads, d_head]`

每个维度对应一个设计决策，从外到内：

| 维度 | 含义 | 设计原因 |
|------|------|---------|
| `total_blocks` | 物理 Block 总数 | 类比 OS 的物理页帧池，用 `block_table` 做虚拟→物理映射，**打破 KV Cache 必须连续存储的假设** |
| `block_size` | 每个 Block 存几个 token | 分配粒度的权衡：太小→碎片化开销大；太大→最后一个 Block 内部浪费多（internal fragmentation） |
| `num_heads` | 注意力头数 | 每个 token 对每个头有独立的 K/V 向量，存储后可直接按 `[:, h, :]` 切头，无需 reshape |
| `d_head` | 每个头的向量维度 | 实际的 K/V 数值，`d_head = d_model // num_heads` |

写入时通过两步寻址定位槽位：

```python
block_idx     = pos // block_size   # 去哪个 block
slot_in_block = pos % block_size    # block 内第几个槽
kv_pool_k[physical_block, slot_in_block] = K[i]
```

**与旧的 `past_key_values` 对比：**

| | past_key_values（多请求 KV Cache + Static Batching~07）| kv_pool（Paged Prefix Cache）|
|---|---|---|
| 形状 | `[batch, num_heads, seq_len, d_head]` | `[total_blocks, block_size, num_heads, d_head]` |
| 内存组织 | 按**序列**组织，每个序列独占连续显存 | 按**物理页**组织，所有序列共享同一物理池 |
| 碎片 | 严重（序列长度各异） | 最多 1 个 Block 的内部碎片 |
| prefix cache | 需要复制 Python 对象 | block_table 复用，**零拷贝** |

本质上，这个形状是把「按序列组织」变成「按物理页组织」——和 OS 虚拟内存管理是同一个思路。

**attention forward 的变化**：写入 + gather 替代 torch.cat

```python
# 写入：当前 token 的 K/V 写入物理槽位
kv_pool_k[block_table[block_idx], slot_in_block] = K[i]

# 读取：从非连续 Block 中 gather 出历史 K/V
K_full = gather_kv_from_blocks(kv_pool_k, block_table, total_len, block_size)

# Attention 照常计算（因果 mask 逻辑不变）
scores = Q @ K_full.T / sqrt(d_head)
```

**prefix cache 命中时**：不再复制 Python 对象，只把已缓存的 `block_id` 列表加入 `block_table`：

```python
# 命中：zero-copy！物理显存里的 KV 数据不动
block_table = cached_block_ids + new_block_table

# model 直接用 block_table 读取已缓存的 KV
logits = model(chunk, block_table=block_table, start_pos=cached_len)
```

## Scheduler：Continuous Batching + Prefix Cache

Paged Prefix Cache 基于 `PagedScheduler` 实现了批处理引擎 `PagedPrefixCacheEngine`，把分页内存管理、前缀缓存和 Continuous Batching 三者结合在一起。

### 核心数据结构：Sequence

```python
class Sequence:
    prompt_ids: Tensor
    block_table: List[int]   # 替代 past_key_values，指向 kv_pool 中的物理 Block
    prefill_offset: int      # 已完成 prefill 的 token 数（支持增量 prefill）
    status: SequenceStatus   # WAITING → PREFILLING → RUNNING → FINISHED
```

`block_table` 是 Sequence 持有的唯一 KV 状态——不是 Python 张量，而是一组物理 Block ID。

### PagedScheduler 调度循环

```python
def schedule() -> (prefill_seqs, decode_seqs):
    # 1. 把已完成的请求移出 running（block 释放由 engine 统一处理）
    # 2. 从 waiting 补充新请求到 running，直到满 max_running
    # 3. 分类：prefill_done=False → prefill_seqs；prefill_done=True → decode_seqs
```

每轮循环中 prefill 和 decode 序列**同时存在于 running 队列**，GPU 不会空转等待。

### Engine 执行逻辑

```python
while scheduler.has_work:
    prefill_seqs, decode_seqs = scheduler.schedule()

    for seq in prefill_seqs:
        _do_prefill_step(seq)   # 处理一个 block_size 的 chunk
        # 首次进入时：lookup prefix cache → 分配 block_table

    for seq in decode_seqs:
        _do_decode_step(seq)    # 用 block_table 读 kv_pool，生成下一 token

    # 释放本轮刚完成的请求（ref_count-- + free new_blocks）
    for seq in finished_seqs:
        _free_seq(seq)
```

Block 生命周期完全由 engine 管理，scheduler 只负责调度，不做任何内存操作。

### 与 Chunked Prefill：切片长 Prompt Chunked Prefill 的关系

`_do_prefill_step` 每次只处理一个 block_size 的 chunk，scheduler 每轮只推进一步——和 Chunked Prefill：切片长 Prompt「1 chunk + 1 decode 交替」的思路完全一致，只是这里 chunk_size = block_size，两个机制统一了。

## 代码结构

| 文件 | 说明 |
|------|------|
| `model.py` | `TinyTransformerPaged`：kv_pool + block_table 替代 past_key_values |
| `block_manager.py` | `BlockManager`：物理 Block 分配/释放/ref_count |
| `engine.py` | `PagedPrefixCacheEngine`：Paged Prefix Cache + Continuous Batching |
| `scheduler.py` | `Sequence`（block_table 版）+ `PagedScheduler` |
| `run.py` | 测试：冷/热两轮串行验证 + scheduler 引擎与串行版输出对比 |

## 运行

```bash
python run.py
```

预期输出：

```
第一轮（冷启动）: XX ms  命中率 90%   (9/10)   ← 第一个请求 miss，其余命中
第二轮（缓存热）: XX ms  命中率 100%  (10/10)  ← 全部命中

两轮输出完全相同 ✅（缓存只是优化，不改变结果）
所有 Block ref_count 正常 ✅
```

第一轮命中率 90% 而非 0% 的原因：第一个请求 miss 后把共享前缀存入缓存，第 2~10 个请求全部命中。

## 下一步

真实模型：加载 Qwen3-0.6B 将加载**真实的 Qwen3-0.6B 模型**，替换本教程一直使用的 TinyTransformer，验证以上所有优化在真实模型上同样成立。
