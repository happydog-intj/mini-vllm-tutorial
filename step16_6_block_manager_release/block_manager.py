"""
step16_6: BlockManager — 新增 release()，与 retain() 对称

在 step16_5 基础上的改动：
  - 新增 release(block_ids)：ref_count -= 1，归零时放回空闲池
  - engine._free_seq 改用 release()，不再直接操作 _blocks 内部状态
"""

from collections import deque
from typing import List


class Block:
    """KV Cache 物理 Block。"""
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0

    def __repr__(self):
        return f"Block(id={self.block_id}, ref={self.ref_count})"


class BlockManager:
    """
    分页 KV Cache 内存管理器。

    核心接口：
      allocate(num_blocks) → block_table（物理 Block ID 列表）
      append_slot(block_table, token_count) → 更新后的 block_table
      free(block_table) → 归还 Block
      translate_slot(block_table, token_pos) → 物理槽位编号
    """

    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self._blocks: List[Block] = [Block(i) for i in range(total_blocks)]
        self._free: deque = deque(self._blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self._free) >= num_blocks

    def allocate(self, num_blocks: int = 1) -> List[int]:
        """分配 num_blocks 个 Block，返回物理 Block ID 列表。"""
        if len(self._free) < num_blocks:
            raise RuntimeError(
                f"KV Block 不足: 需要 {num_blocks}, 剩余 {len(self._free)}"
            )
        block_table = []
        for _ in range(num_blocks):
            blk = self._free.popleft()
            blk.ref_count = 1
            block_table.append(blk.block_id)
        return block_table

    def append_slot(self, block_table: List[int], token_count: int) -> List[int]:
        """检查 token_count 个 token 是否需要新 Block，按需分配。"""
        needed_blocks = (token_count + self.block_size - 1) // self.block_size
        current_blocks = len(block_table)
        if needed_blocks > current_blocks:
            new_blocks = self.allocate(needed_blocks - current_blocks)
            block_table = block_table + new_blocks
        return block_table

    def retain(self, block_ids: List[int]):
        """对一组 Block 增加引用计数（prefix cache 命中/保存时使用）。"""
        for bid in block_ids:
            self._blocks[bid].ref_count += 1

    def release(self, block_ids: List[int]):
        """减少引用计数，归零时放回空闲池（prefix cache 引用释放时使用）。"""
        for bid in block_ids:
            blk = self._blocks[bid]
            blk.ref_count -= 1
            if blk.ref_count <= 0:
                blk.ref_count = 0
                self._free.append(blk)

    def free(self, block_table: List[int]):
        """释放整个 block_table 对应的所有 Block。"""
        for bid in block_table:
            blk = self._blocks[bid]
            blk.ref_count -= 1
            if blk.ref_count <= 0:
                blk.ref_count = 0
                self._free.append(blk)

    def translate_slot(self, block_table: List[int], token_pos: int) -> int:
        """
        将逻辑 token 位置转换为物理 KV 存储槽位编号。

        逻辑位置 token_pos:
          block_idx     = token_pos // block_size
          slot_in_block = token_pos  % block_size
        物理槽位:
          physical_slot = block_table[block_idx] * block_size + slot_in_block
        """
        block_idx = token_pos // self.block_size
        slot_in_block = token_pos % self.block_size
        physical_block_id = block_table[block_idx]
        return physical_block_id * self.block_size + slot_in_block
