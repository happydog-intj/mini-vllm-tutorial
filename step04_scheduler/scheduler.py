"""
step04: Sequence 状态机 + Continuous Batching Scheduler

教学要点:
  - Sequence：代表一个推理请求，持有自己的 token_ids 和 KV Cache
  - 三状态机：WAITING → RUNNING → FINISHED
  - Scheduler：维护 waiting 和 running 两个队列，每步动态调度
  - Continuous Batching：完成即释放，新请求立即补入
"""

from enum import Enum, auto
from typing import List, Tuple
from collections import deque
import torch
from torch import Tensor


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """单个推理请求的完整状态。"""

    EOS_TOKEN_ID = 1

    def __init__(self, prompt_ids: Tensor, max_new_tokens: int):
        self.token_ids: List[int] = prompt_ids.tolist()
        self.past_key_values = None
        self.status = SequenceStatus.WAITING
        self.max_new_tokens = max_new_tokens
        self._generated_count = 0

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self._generated_count += 1

    @property
    def is_done(self) -> bool:
        return (
            self._generated_count >= self.max_new_tokens
            or (self.token_ids and self.token_ids[-1] == self.EOS_TOKEN_ID)
        )

    def get_last_token(self) -> Tensor:
        return torch.tensor([self.token_ids[-1]])


class Scheduler:
    """
    Continuous Batching 调度器。

    每轮 schedule()：
      1. 移除已完成的请求
      2. 从 waiting 补充新请求到 running（直到满 max_running）
      3. 新进来的做 prefill，已有的做 decode
    """

    def __init__(self, max_running: int = 4):
        self.waiting: deque = deque()
        self.running: List[Sequence] = []
        self.max_running = max_running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], List[Sequence]]:
        """
        返回:
          prefill_seqs: 本轮需要 prefill 的（刚从 WAITING 进来）
          decode_seqs:  本轮需要 decode 的（已有 KV Cache）
        """
        # 移除已完成的
        finished = [s for s in self.running if s.is_done]
        for s in finished:
            s.status = SequenceStatus.FINISHED
            self.running.remove(s)

        # 补充新请求
        prefill_seqs = []
        while self.waiting and len(self.running) < self.max_running:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill_seqs.append(seq)

        decode_seqs = [s for s in self.running if s not in prefill_seqs]
        return prefill_seqs, decode_seqs

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
