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
# step07 的做法：prompt 全部跑完后再缓存各边界的 hash
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
past_kv = cached_kv   # 从命中的断点开始（或 None）
pos = cached_len

while pos < prompt_len:
    end = min(pos + block_size, prompt_len)
    chunk = prompt_ids[pos:end]
    logits, past_kv = model(chunk, past_key_values=past_kv)
    pos = end

    # 只在完整 Block 边界处缓存
    if pos % block_size == 0:
        h = chain_hash(tokens, prev_hash, pos - block_size, pos)
        if h not in prefix_cache:
            blk = block_manager.allocate(1)
            prefix_cache[h] = {
                "block_id": blk[0],
                "past_kv":  past_kv,   # ← 正确：此时只含前 pos 个 token 的信息
                "length":   pos,
            }
        prev_hash = h
```

此时 `past_kv` 只包含 `tokens[0:pos]` 的 KV，是真正意义上的"前 pos 个 token 的缓存"。

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

## step07 vs step08 V1 vs step08 V2 对比

| 特性 | step07 | step08 V1 | step08 V2（TinyTransformerPaged）|
|------|--------|-----------|----------------------------------|
| KV 快照正确性 | ❌ 含整个 prompt 信息 | ✅ 只含前 N 个 token | ✅ 只含前 N 个 token |
| 缓存粒度 | 整段 past_key_values | Block 粒度 | Block 粒度 |
| 跨请求共享 | ❌ 无 ref_count | ✅ ref_count 控制 | ✅ ref_count 控制 |
| 驱逐支持 | ❌ 永不驱逐 | ✅ ref_count 归零可驱逐 | ✅ ref_count 归零可驱逐 |
| 增量 prefill | ❌ 全量后缓存 | ✅ 逐 Block 边界缓存 | ✅ 逐 Block 边界缓存 |
| past_kv 存储 | Python 对象 | Python 对象（游离） | ❌ 彻底消失 |
| KV 数据托管 | ❌ Python 堆内存 | ❌ Python 堆内存 | ✅ kv_pool 张量（BlockManager 管理） |
| prefix cache 命中 | 复制 Python 对象引用 | 复制 Python 对象引用 | ✅ block_table 复用，零拷贝 |

## V2：TinyTransformerPaged

`model_paged.py` 实现了 `TinyTransformerPaged`，把 `past_key_values` 彻底替换为 `kv_pool + block_table`：

```python
# V1（TinyTransformerWithKVCache）：
logits, past_kv = model(chunk, past_key_values=past_kv)  # past_kv 是 Python 对象

# V2（TinyTransformerPaged）：
logits = model(chunk, block_table=block_table, start_pos=pos)  # KV 写入 kv_pool，无返回值
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

### 教学版的剩余局限：past_kv 仍是游离的 Python 对象

step08 的 `_prefix_cache` 里，`block_id` 用于 ref_count 生命周期管理，但 `past_kv` 本身仍是一个普通的 Python/PyTorch tensor，存在 Python 堆内存里，不受 BlockManager 控制：

```python
prefix_cache[h] = {
    "block_id": blk[0],    # ← 只用于 ref_count，不是 KV 数据的实际存储位置
    "past_kv":  past_kv,   # ← 仍是 Python 对象，在 Python 内存里游离
    "length":   pos,
}
```

**真正的生产实现应该是：**

所有 KV Cache 数据统一存储在一个由 BlockManager 管理的全局张量里，`past_kv` 不再是独立对象，而是这块显存的切片：

```python
# 全局 KV Cache 张量，BlockManager 管理的就是这块物理显存
kv_pool = torch.zeros(total_blocks, block_size, num_heads, head_dim)

# 写入时：把计算好的 K/V 写入对应的物理 Block
kv_pool[block_id, slot_in_block] = k_or_v

# 前缀缓存命中时：复用的是 kv_pool 里已写入的那些 Block
# 不需要复制数据，只需把 block_id 加入新请求的 block_table
new_request.block_table.extend(cached_block_ids)  # 零拷贝！

# Attention 计算时：FlashAttention kernel 接收 block_table，从 kv_pool 按地址读取
flash_attn_with_kvcache(Q, kv_pool, block_table=new_request.block_table, ...)
```

**为什么 step08 还做不到这一点？**

step08 使用的是 `TinyTransformer`，它的接口是 `model(chunk, past_key_values=past_kv)`，没有 `block_table` 的概念。把 `block_table` 真正接入 attention 计算需要 FlashAttention 的 PagedAttention 接口（`flash_attn_with_kvcache` + `block_table` 参数），将在 step10 引入。

到 step10，分页内存管理和前缀缓存的闭环才真正合拢：BlockManager 管理的物理 Block 不再只是元数据，而是 attention kernel 直接读取的显存地址。

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
