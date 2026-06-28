# step16_6 — BlockManager.release()：封装引用计数释放，消除直接操作内部状态

## 问题

`Paged Prefix Cache` 的 `_free_seq` 直接操作 `BlockManager` 的私有字段：

```python
# engine.py — 绕过接口直接改内部状态
def _free_seq(self, seq: Sequence):
    cached_ids = seq.block_table[:len(seq.block_table) - len(getattr(seq, '_new_blocks', []))]
    for bid in cached_ids:
        self.block_manager._blocks[bid].ref_count -= 1  # ← 直接访问私有属性
    self.block_manager.free(getattr(seq, '_new_blocks', []))
```

**设计问题：**
1. engine 直接操作 `_blocks[bid].ref_count`，破坏封装
2. `retain()`（增加引用）走接口，`release`（减少引用）绕过接口——不对称
3. engine 需要自己区分"哪些是 prefix cache 的 block，哪些是新分配的 block"，这个逻辑应该在 BlockManager 里
4. 如果 BlockManager 内部引入 LRU 或其他策略，engine 里的代码也要跟着改

## 解决方案：对称的 release() 接口

```python
class BlockManager:
    def retain(self, block_ids: List[int]):
        """增加引用计数（prefix cache 命中/保存时）。"""
        for bid in block_ids:
            self._blocks[bid].ref_count += 1

    def release(self, block_ids: List[int]):
        """减少引用计数，归零时放回空闲池（prefix cache 引用释放时）。"""
        for bid in block_ids:
            blk = self._blocks[bid]
            blk.ref_count -= 1
            if blk.ref_count <= 0:
                blk.ref_count = 0
                self._free.append(blk)
```

**`release()` 与已有 `free()` 的区别：**

| 方法 | 语义 | 用途 |
|------|------|------|
| `free(block_ids)` | 直接回收，ref_count 不判断 | 序列结束时释放新分配的 block |
| `release(block_ids)` | ref_count -= 1，归零才回收 | prefix cache 引用释放，可能还被其他序列持有 |

**engine 的 `_free_seq` 变得清晰：**

```python
def _free_seq(self, seq: Sequence):
    cached_count = len(seq.block_table) - len(getattr(seq, '_new_blocks', []))
    self.block_manager.release(seq.block_table[:cached_count])  # prefix cache 引用
    self.block_manager.free(getattr(seq, '_new_blocks', []))    # 本次新分配的 block
```

## 完整接口一览

```
allocate(n)          → 分配 n 个 block，ref_count = 1
retain(block_ids)    → ref_count += 1（prefix cache 写入/命中时）
release(block_ids)   → ref_count -= 1，归零回收（prefix cache 引用释放时）
free(block_ids)      → 直接回收（序列新分配的 block）
append_slot(...)     → 按需扩展 block_table
```

engine 不再需要接触 `_blocks` 私有属性。

## 与 vLLM 的对比

vLLM 的 `BlockSpaceManager` 提供完整的引用计数接口，engine 层完全不直接操作 block 内部状态。本章实现了同样的封装原则：**BlockManager 是 KV 内存的唯一管理者，engine 只调接口。**

## 实现

见 `block_manager.py` — 新增 `release()` 方法；`engine.py` — `_free_seq` 改用 `release()`。

## 运行

```bash
python run.py
```
