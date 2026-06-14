"""
step05b: NoPreemptionEngine（会 OOM）+ PreemptionEngine（优雅降级）
"""

import torch
from torch import Tensor
from typing import List, Tuple
from model import TinyTransformerWithKVCache
from scheduler import Sequence, PreemptionScheduler


class NoPreemptionEngine:
    """无抢占引擎：KV 槽位不足时直接报错（对照组）"""

    def __init__(self, max_kv_slots: int = 1000):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()
        self.max_kv_slots = max_kv_slots
        self._used_slots = 0

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        seqs = []
        for prompt_ids, max_new in requests:
            if self._used_slots + len(prompt_ids) > self.max_kv_slots:
                raise RuntimeError(
                    f"KV Cache 已满：已用 {self._used_slots}, "
                    f"需要 {len(prompt_ids)}, 上限 {self.max_kv_slots}"
                )
            logits, pkv = self.model(prompt_ids)
            self._used_slots += len(prompt_ids)
            seq = Sequence(prompt_ids, max_new)
            seq.past_key_values = pkv
            seq.append_token(torch.argmax(logits[-1]).item())
            seqs.append(seq)

        for _ in range(max(n for _, n in requests) - 1):
            for seq in seqs:
                if seq.is_done:
                    continue
                if self._used_slots + 1 > self.max_kv_slots:
                    raise RuntimeError(f"KV Cache 已满：已用 {self._used_slots}, 上限 {self.max_kv_slots}")
                logits, seq.past_key_values = self.model(
                    seq.get_last_token(), past_key_values=seq.past_key_values
                )
                seq.append_token(torch.argmax(logits[-1]).item())
                self._used_slots += 1

        return [torch.tensor(s.token_ids) for s in seqs]


class PreemptionEngine:
    """带 Preemption 的引擎：KV 满时驱逐低优先级请求，不崩溃"""

    def __init__(self, max_kv_slots: int = 1000):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()
        self.max_kv_slots = max_kv_slots
        self.scheduler = PreemptionScheduler(max_running=8)

    @property
    def preempt_count(self) -> int:
        return self.scheduler.preempt_count

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        seqs = []
        for i, (prompt_ids, max_new) in enumerate(requests):
            seq = Sequence(prompt_ids, max_new, seq_id=i)
            self.scheduler.add(seq)
            seqs.append(seq)

        while self.scheduler.has_work:
            prefill_seqs, decode_seqs = self.scheduler.schedule(
                max_kv_slots=self.max_kv_slots
            )

            for seq in prefill_seqs:
                logits, seq.past_key_values = self.model(seq.prompt_ids)
                seq.append_token(torch.argmax(logits[-1]).item())

            for seq in decode_seqs:
                logits, seq.past_key_values = self.model(
                    seq.get_last_token(), past_key_values=seq.past_key_values
                )
                seq.append_token(torch.argmax(logits[-1]).item())

        return [torch.tensor(s.token_ids) for s in seqs]
