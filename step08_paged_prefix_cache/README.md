# step08 — Paged Prefix Cache

## 教学目标

把 [step06 的分页内存管理](../step06_paged_attention/README.md) 和 [step07 的前缀缓存](../step07_prefix_cache/README.md) 结合起来，实现正确的、Block 粒度的前缀缓存：

- 以 **Block 为单位**存储前缀 KV 快照（而非整段 past_key_values）
- 命中时 `ref_count++`，直接复用，物理显存零拷贝
- 未命中时**逐 Block 增量 prefill**，在每个 Block 边界处保存**正确的** KV 快照
- 释放时 `ref_count--`，归零才真正回收

## step07 的两个核心问题

step07 的实现能跑通，但有两个根本性缺陷：

**问题1：缓存了错误的 KV 快照**

```python
# step07 的做法：prompt 全部跑完后按照block计算hash并缓存
for end in range(block_size, prompt_len + 1, block_size):
    h = compute_hash(tokens, end)
    prefix_cache[h] = past_kv   # ← past_kv 含完整 prompt 的信息，不是前 end 个 token！
```

`past_kv` 是整个 prompt 跑完后的状态，包含了 token `end` 之后的所有信息。下次命中 `h` 时，拿到的 KV 包含了"未来"信息，在真实模型上会产生错误输出。

**问题2：无法跨请求共享物理 Block**

step07 缓存的是 Python 对象（`past_key_values` tuple），每个请求命中后得到的是对同一对象的引用，没有 `ref_count` 控制生命周期——无法安全地在多请求之间共享并释放。

## step08 的修正：逐 Block 增量 prefill

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

step08 的设计和 OS 内存管理高度对应：

| 操作系统 | step08 KV Cache |
|---------|----------------|
| 进程创建 → 分配虚拟地址空间 + 建页表 | 请求开始 → allocate() + 建 block_table |
| 页表：虚拟页 → 物理页帧 | block_table：逻辑 token 位置 → 物理 Block ID |
| 进程运行 → 按需写入物理页 | prefill/decode → 按顺序写入 kv_pool |
| 进程退出 → 归还页帧 | 请求结束 → free(block_table) |
| `fork()` 后父子进程共享物理页（CoW）| prefix cache 命中 → 多请求共享同一物理 Block |
| 写时复制（CoW）：写操作才分配新页 | prefix cache 只读共享：新 token 才分配新 Block |

最后一行的 CoW 类比最精妙：OS 中 `fork()` 后父子进程的页表指向同一物理页，只要没有写操作就不复制（`ref_count` 控制）；step08 中多个请求命中同一前缀时，`block_table` 指向同一物理 Block，只要不写新 token 就不分配新 Block——`ref_count++` 即可，物理显存零拷贝。

### 顺带实现了 Chunked Prefill

逐 Block 增量 prefill 的循环结构和 [step05a 的 Chunked Prefill](../step05a_chunked_prefill/README.md) 完全相同——都是把长 prompt 切成小块依次处理：

```python
while pos < prompt_len:
    chunk = prompt_ids[pos : pos + block_size]   # chunk_size = block_size
    logits, past_kv = model(chunk, past_kv=past_kv)
    pos += len(chunk)
```

两者解决的问题不同，但机制一致：

| | step05a Chunked Prefill | step08 逐 Block Prefill |
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

## step07 vs step08 对比

| 特性 | step07 | step08（本章） |
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

`model_paged.py` 实现了 `TinyTransformerPaged`，把 `past_key_values` 彻底替换为 `kv_pool + block_table`：

```python
# TinyTransformerWithKVCache（step03a~step07）：
logits, past_kv = model(chunk, past_key_values=past_kv)  # past_kv 是 Python 对象，每步返回

# TinyTransformerPaged（本章）：
logits = model(chunk, block_table=block_table, start_pos=pos)  # KV 直接写入 kv_pool，无返回值
```

`kv_pool` 作为模型的 `register_buffer` 注册，每层各有一对：

```python
self.kv_pool_k = torch.zeros(num_layers, total_blocks, block_size, num_heads, d_head)
self.kv_pool_v = torch.zeros(num_layers, total_blocks, block_size, num_heads, d_head)
```

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

step09 将加载**真实的 Qwen3-0.6B 模型**，替换本教程一直使用的 TinyTransformer，验证以上所有优化在真实模型上同样成立。
