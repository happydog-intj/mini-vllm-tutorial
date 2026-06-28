"""
step10: NormalEngine（对照）+ ChunkedPrefillEngine
"""

import torch
from torch import Tensor
from typing import List, Tuple
from model import TinyTransformerWithKVCache
from scheduler import Sequence, ChunkedScheduler, SequenceStatus


class NormalEngine:
    """无 Chunked Prefill 的对照引擎（step09 风格）"""

    def __init__(self):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()

    @torch.no_grad()
    def generate_batch(self, requests):
        seqs = [Sequence(p, n) for p, n in requests]
        for seq in seqs:
            logits, seq.past_key_values = self.model(seq.prompt_ids)
            seq.append_token(torch.argmax(logits[-1]).item())
        for _ in range(max(n for _, n in requests) - 1):
            for seq in seqs:
                if seq.is_done:
                    continue
                logits, seq.past_key_values = self.model(
                    seq.get_last_token(), past_key_values=seq.past_key_values
                )
                seq.append_token(torch.argmax(logits[-1]).item())
        return [torch.tensor(s.token_ids) for s in seqs]


class ChunkedPrefillEngine:
    """Chunked Prefill 引擎：长 prompt 分块处理，不阻塞 decode。"""

    def __init__(self, chunk_size: int = 50):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()
        self.chunk_size = chunk_size

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        scheduler = ChunkedScheduler(max_running=8, chunk_size=self.chunk_size)
        seqs = []
        for prompt_ids, max_new in requests:
            seq = Sequence(prompt_ids, max_new)
            scheduler.add(seq)
            seqs.append(seq)

        while scheduler.has_work:
            prefill_chunk, decode_seq = scheduler.schedule()

            # 每步至多一块 prefill → 立刻一个 decode，严格分时交替
            if prefill_chunk:
                seq, start, end = prefill_chunk[0]
                chunk = seq.prompt_ids[start:end]
                logits, seq.past_key_values = self.model(
                    chunk, past_key_values=seq.past_key_values
                )
                seq.prefill_offset = end
                if seq.prefill_done:
                    seq.append_token(torch.argmax(logits[-1]).item())
                    seq.status = SequenceStatus.RUNNING

            if decode_seq:
                seq = decode_seq[0]
                logits, seq.past_key_values = self.model(
                    seq.get_last_token(), past_key_values=seq.past_key_values
                )
                seq.append_token(torch.argmax(logits[-1]).item())

        return [torch.tensor(s.token_ids) for s in seqs]
