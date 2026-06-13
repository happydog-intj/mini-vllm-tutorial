"""
step05a: 支持 Chunked Prefill 的调度器

相对 step04 的变化：
  - Sequence 新增 prefill_offset：已处理了多少 prompt token
  - schedule() 返回 prefill_chunks（分块信息）+ decode_seqs
  - 每步最多处理 chunk_size 个 prefill token
"""

from enum import Enum, auto
from typing import List, Tuple
from collections import deque
import torch
from torch import Tensor


class SequenceStatus(Enum):
    WAITING = auto()
    PREFILLING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    def __init__(self, prompt_ids: Tensor, max_new_tokens: int):
        self.prompt_ids = prompt_ids
        self.token_ids: List[int] = prompt_ids.tolist()
        self.past_key_values = None
        self.status = SequenceStatus.WAITING
        self.max_new_tokens = max_new_tokens
        self._generated_count = 0
        self.prefill_offset = 0

    @property
    def prefill_done(self) -> bool:
        return self.prefill_offset >= len(self.prompt_ids)

    @property
    def is_done(self) -> bool:
        return (
            self._generated_count >= self.max_new_tokens
            or (self.token_ids and self.token_ids[-1] == 1)
        )

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self._generated_count += 1

    def get_last_token(self) -> Tensor:
        return torch.tensor([self.token_ids[-1]])


class ChunkedScheduler:
    """
    支持 Chunked Prefill 的调度器。

    长 prompt 每步只处理 chunk_size 个 token，
    与 decode 请求混合调度，不阻塞 decode。
    """

    def __init__(self, max_running: int = 8, chunk_size: int = 50):
        self.waiting: deque = deque()
        self.running: List[Sequence] = []
        self.max_running = max_running
        self.chunk_size = chunk_size

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Tuple], List[Sequence]]:
        """
        返回：
          prefill_chunks: List[(seq, start, end)]
          decode_seqs:    List[Sequence]
        """
        # 移除已完成的
        for s in list(self.running):
            if s.is_done:
                s.status = SequenceStatus.FINISHED
                self.running.remove(s)

        # 补充新请求
        while self.waiting and len(self.running) < self.max_running:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.PREFILLING
            self.running.append(seq)

        # 分配 prefill chunks
        prefill_chunks = []
        remaining = self.chunk_size
        for seq in self.running:
            if seq.prefill_done:
                continue
            start = seq.prefill_offset
            end = min(start + remaining, len(seq.prompt_ids))
            prefill_chunks.append((seq, start, end))
            remaining -= (end - start)
            if remaining <= 0:
                break

        # 已完成 prefill 的做 decode
        decode_seqs = [s for s in self.running if s.prefill_done and not s.is_done]
        return prefill_chunks, decode_seqs

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
