"""
step06: PagedAttentionEngine

核心变化：引入 BlockManager，KV Cache 按需分页分配。
教学版：用 past_key_values 存储 KV（简化），Block 体现在内存分配逻辑上。
"""

import torch
from torch import Tensor
from typing import List, Tuple
from model import TinyTransformerWithKVCache
from scheduler import Sequence, Scheduler
from block_manager import BlockManager


class PagedAttentionEngine:
    def __init__(self, total_blocks: int = 50, block_size: int = 4):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()
        self.block_manager = BlockManager(total_blocks, block_size)

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        scheduler = Scheduler(max_running=8)
        seqs = []
        for prompt_ids, max_new in requests:
            seq = Sequence(prompt_ids, max_new)
            seq.block_table = []
            scheduler.add(seq)
            seqs.append(seq)

        while scheduler.has_work:
            prefill_seqs, decode_seqs = scheduler.schedule()

            for seq in prefill_seqs:
                needed = (len(seq.token_ids) + self.block_manager.block_size - 1) \
                         // self.block_manager.block_size
                if not self.block_manager.can_allocate(needed):
                    continue
                seq.block_table = self.block_manager.allocate(needed)
                logits, seq.past_key_values = self.model(
                    torch.tensor(seq.token_ids)
                )
                seq.append_token(torch.argmax(logits[-1]).item())

            for seq in decode_seqs:
                seq.block_table = self.block_manager.append_slot(
                    seq.block_table, len(seq.token_ids) + 1
                )
                logits, seq.past_key_values = self.model(
                    seq.get_last_token(), past_key_values=seq.past_key_values
                )
                seq.append_token(torch.argmax(logits[-1]).item())
                if seq.is_done:
                    self.block_manager.free(seq.block_table)

        return [torch.tensor(s.token_ids) for s in seqs]
