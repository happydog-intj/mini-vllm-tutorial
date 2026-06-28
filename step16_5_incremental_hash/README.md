# step16_5 — 增量 Hash：避免 lookup 时重算整个 prompt 的 hash 链

## 问题

`Paged Prefix Cache` 的 `_lookup_prefix_cache` 每次查找都从头重算整个 prompt 的链式 hash：

```python
def _lookup_prefix_cache(self, tokens: List[int]):
    prev_hash = 0
    block_hashes = []
    # 每次请求到来都遍历整个 prompt，重新计算所有 block 的 hash
    for start in range(0, prompt_len - prompt_len % self.block_size, self.block_size):
        h = self._chain_hash(tokens, prev_hash, start, start + self.block_size)
        block_hashes.append((end, h))
        prev_hash = h
    for end, h in reversed(block_hashes):
        if h in self._prefix_cache:
            ...
```

**性能代价：**
- 每次请求都遍历整个 prompt，xxhash 计算次数 = `prompt_len // block_size`
- 长 prompt（4096 token，block_size=16）需要算 256 次 hash
- hash 结果没有缓存，相同 prompt 再来还是重算
- `_save_prefix_cache` 依赖外部传入的 `prev_hash`，状态管理分散在 engine 和 seq 之间

## 解决方案：hash 状态维护在 Sequence 上，增量更新

将 `prev_hash` 和已算好的 block hash 列表作为 Sequence 的持久状态：

```python
class Sequence:
    def __init__(self, prompt_ids, max_new_tokens):
        ...
        self._block_hashes: List[int] = []  # 按顺序积累，每个完整 block 一个
        self._prev_hash: int = 0
```

**prefill 过程中在 block 边界增量计算：**

```python
def _do_prefill_step(self, seq):
    ...
    if end % self.block_size == 0 and end <= prompt_len:
        h = self._chain_hash(seq.prompt_ids.tolist(), seq._prev_hash,
                             end - self.block_size, end)
        seq._block_hashes.append(h)
        seq._prev_hash = h
        self._save_prefix_cache(seq, h, end)
```

**lookup 直接用已算好的 hash，无需重算：**

```python
def _lookup_prefix_cache(self, seq: Sequence):
    # 冷请求：一次性预计算全部 block hash
    if not seq._block_hashes:
        self._precompute_block_hashes(seq)

    # 从最长前缀往回找，O(num_blocks) 次 dict 查找，无 hash 计算
    for i in range(len(seq._block_hashes) - 1, -1, -1):
        h = seq._block_hashes[i]
        if h in self._prefix_cache:
            entry = self._prefix_cache[h]
            ...
            return cached_block_ids, entry["length"], h
    return [], 0, 0
```

## 收益

| 场景 | Paged Prefix Cache | step16_5 |
|------|-----------|---------|
| 第一次请求（冷）| 算 N 次（lookup）+ N 次（save 重算）| 算 N 次（预计算一遍）|
| prefill 过程中保存 | 需外部传入 prev_hash | seq._prev_hash 自动维护 |
| 相同 prompt 再次请求 | 重算 N 次 hash | 0 次（hash 已在 seq 上）|
| 代码复杂度 | prev_hash 在 engine 和 seq 间传递 | 状态集中在 seq 上 |

## 与 vLLM 的对比

vLLM 的 `SequenceGroup` 维护完整的 block hash 链，`BlockSpaceManager` 在分配 block 时直接查已有的 hash，不重算。本章实现了同样的增量维护思路。

## 实现

见 `scheduler.py` — `Sequence` 新增 `_block_hashes`/`_prev_hash` 字段；`engine.py` — `_lookup_prefix_cache` 和 `_do_prefill_step` 改为增量方式。

## 运行

```bash
python run.py
```
