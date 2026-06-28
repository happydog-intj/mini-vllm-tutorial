"""
step11: 带 Preemption 的调度器

相对 step09 的变化：
  - Sequence 增加 kv_len 属性（当前占用的 KV 槽位数）
  - schedule() 接受 max_kv_slots 参数
  - 若 running 请求所需 slots > max_kv_slots，驱逐最后加入的
  - 被驱逐的请求：释放 KV Cache，重置到 prompt 状态，插回 waiting 队首
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
    PREEMPTED = auto()


class Sequence:
    def __init__(self, prompt_ids: Tensor, max_new_tokens: int, seq_id: int = 0):
        self.seq_id = seq_id
        self.prompt_ids = prompt_ids
        self.token_ids: List[int] = prompt_ids.tolist()
        self.past_key_values = None
        self.status = SequenceStatus.WAITING
        self.max_new_tokens = max_new_tokens
        self._generated_count = 0

    @property
    def kv_len(self) -> int:
        """当前占用的 KV 槽位数"""
        return len(self.token_ids)

    @property
    def is_done(self) -> bool:
        return self._generated_count >= self.max_new_tokens

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self._generated_count += 1

    def get_last_token(self) -> Tensor:
        return torch.tensor([self.token_ids[-1]])

    def free_kv_cache(self):
        """释放 KV Cache（被抢占时调用）"""
        self.past_key_values = None


class PreemptionScheduler:
    def __init__(self, max_running: int = 8):
        self.waiting: deque = deque()
        self.running: List[Sequence] = []
        self.max_running = max_running
        self.preempt_count = 0

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self, max_kv_slots: int) -> Tuple[List[Sequence], List[Sequence]]:
        """
        max_kv_slots: 系统最大 KV 槽位数

        若 running 中所有请求下一步所需 slots 超出上限，
        则驱逐最后加入的请求（LIFO 策略）。
        """
        # 移除已完成的
        for s in list(self.running):
            if s.is_done:
                s.status = SequenceStatus.FINISHED
                self.running.remove(s)

        # 检查是否需要抢占（每个请求下一步需要 kv_len+1 个槽位）
        while self.running:
            needed = sum(s.kv_len + 1 for s in self.running)
            if needed <= max_kv_slots:
                break
            # 驱逐最后加入的（优先级最低）
            victim = self.running[-1]
            victim.status = SequenceStatus.PREEMPTED
            victim.free_kv_cache()
            # 重置到 prompt 状态
            victim.token_ids = victim.prompt_ids.tolist()
            victim._generated_count = 0
            self.running.remove(victim)
            self.waiting.appendleft(victim)  # 插回队首优先恢复
            self.preempt_count += 1

        # 补充新请求
        prefill_seqs = []
        while (self.waiting and len(self.running) < self.max_running
               and sum(s.kv_len for s in self.running) + len(self.waiting[0].prompt_ids) <= max_kv_slots):
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill_seqs.append(seq)

        decode_seqs = [s for s in self.running if s not in prefill_seqs]
        return prefill_seqs, decode_seqs

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
