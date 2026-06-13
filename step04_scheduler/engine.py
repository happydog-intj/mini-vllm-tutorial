"""
step04: StaticBatchingEngine（对照）+ ContinuousBatchingEngine（本步核心）
"""

import torch
from torch import Tensor
from typing import List, Tuple
from model import TinyTransformerWithKVCache
from scheduler import Sequence, Scheduler, SequenceStatus


class StaticBatchingEngine:
    """step03b 风格的 Static Batching（对照组）"""

    def __init__(self):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        past_kvs = []
        last_ids = []
        for prompt_ids, _ in requests:
            logits, pkv = self.model(prompt_ids)
            past_kvs.append(pkv)
            last_ids.append(torch.argmax(logits[-1]).item())

        generated = [[lid] for lid in last_ids]
        max_new = max(n for _, n in requests)

        for step in range(max_new - 1):
            for i, (_, max_new_i) in enumerate(requests):
                if step >= max_new_i - 1:
                    continue
                nid = torch.tensor([generated[i][-1]])
                logits, past_kvs[i] = self.model(nid, past_key_values=past_kvs[i])
                generated[i].append(torch.argmax(logits[-1]).item())

        return [
            torch.cat([prompt_ids, torch.tensor(generated[i])])
            for i, (prompt_ids, _) in enumerate(requests)
        ]


class ContinuousBatchingEngine:
    """
    Continuous Batching 推理引擎。

    主循环：
      while scheduler.has_work:
        prefill_seqs, decode_seqs = scheduler.schedule()
        对 prefill_seqs 逐个 prefill
        对 decode_seqs 逐个 decode（各自传 1 个 token）
      完成的请求即时被调度器移除，新请求立即进来
    """

    def __init__(self, max_running: int = 4):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()
        self.max_running = max_running

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        scheduler = Scheduler(max_running=self.max_running)
        seqs = []
        for prompt_ids, max_new in requests:
            seq = Sequence(prompt_ids, max_new)
            scheduler.add(seq)
            seqs.append(seq)

        while scheduler.has_work:
            prefill_seqs, decode_seqs = scheduler.schedule()

            for seq in prefill_seqs:
                prompt_tensor = torch.tensor(seq.token_ids)
                logits, seq.past_key_values = self.model(prompt_tensor)
                next_id = torch.argmax(logits[-1]).item()
                seq.append_token(next_id)

            for seq in decode_seqs:
                logits, seq.past_key_values = self.model(
                    seq.get_last_token(), past_key_values=seq.past_key_values
                )
                next_id = torch.argmax(logits[-1]).item()
                seq.append_token(next_id)

        return [torch.tensor(seq.token_ids) for seq in seqs]
