"""
step14_5: Sequence + Scheduler（增量 hash 维护）

在 step14_4 基础上的变化：
  - Sequence 新增 _block_hashes / _prev_hash，hash 状态随 prefill 增量积累
  - 避免 lookup 时重算整个 prompt 的 hash 链
"""

from enum import Enum, auto
from typing import List, Tuple, Optional
from collections import deque
import torch
from torch import Tensor
from block_manager import BlockManager


class SequenceStatus(Enum):
    WAITING = auto()
    PREFILLING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """单个推理请求的完整状态（block_table 版本）。"""

    EOS_TOKEN_ID = 1

    def __init__(self, prompt_ids: Tensor, max_new_tokens: int):
        self.prompt_ids: Tensor = prompt_ids
        self.token_ids: List[int] = prompt_ids.tolist()
        self.status = SequenceStatus.WAITING
        self.max_new_tokens = max_new_tokens
        self._generated_count = 0
        self.prefill_offset: int = 0       # 已完成 prefill 的 token 数
        self.block_table: List[int] = []   # 物理 Block ID 列表（替代 past_key_values）
        self._block_hashes: List[int] = [] # 每个完整 block 的链式 hash（增量积累）
        self._prev_hash: int = 0           # 最新的链式 hash 状态

    @property
    def prefill_done(self) -> bool:
        return self.prefill_offset >= len(self.prompt_ids)

    @property
    def is_done(self) -> bool:
        return (
            self._generated_count >= self.max_new_tokens
            or (self.token_ids and self.token_ids[-1] == self.EOS_TOKEN_ID)
        )

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self._generated_count += 1

    def get_last_token(self) -> Tensor:
        return torch.tensor([self.token_ids[-1]])

    @property
    def current_len(self) -> int:
        """当前序列总长度（prompt + 已生成 token）"""
        return len(self.token_ids)


class PagedScheduler:
    """
    支持 block_table + prefix cache 的 Continuous Batching 调度器。

    每轮 schedule()：
      1. 移除已完成的请求，free 其 block_table
      2. 从 waiting 补充新请求到 running（直到满 max_running）
      3. 新进来的需要 prefill，已有的需要 decode
    """

    def __init__(self, block_manager: BlockManager, max_running: int = 4, block_size: int = 16):
        self.block_manager = block_manager
        self.max_running = max_running
        self.block_size = block_size
        self.waiting: deque = deque()
        self.running: List[Sequence] = []

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], List[Sequence]]:
        """
        返回:
          prefill_seqs: 本轮需要 prefill 的（刚从 WAITING 进来，prefill_done=False）
          decode_seqs:  本轮需要 decode 的（prefill 已完成）
        """
        # 移除已完成的（不在这里 free block，由 engine 统一管理）
        for seq in list(self.running):
            if seq.is_done:
                seq.status = SequenceStatus.FINISHED
                self.running.remove(seq)

        # 补充新请求
        while self.waiting and len(self.running) < self.max_running:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.PREFILLING
            self.running.append(seq)

        prefill_seqs = [s for s in self.running if not s.prefill_done]
        decode_seqs  = [s for s in self.running if s.prefill_done and not s.is_done]
        return prefill_seqs, decode_seqs

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
