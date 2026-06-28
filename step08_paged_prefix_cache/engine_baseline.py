"""
step08: 对照组 — 无前缀缓存

每次请求都完整重新 prefill，作为性能基准。
"""

import torch
from torch import Tensor
from typing import List, Tuple
from model import TinyTransformerWithKVCache


class NoPrefixCacheEngine:
    """无前缀缓存（对照组）"""

    def __init__(self):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        results = []
        for prompt_ids, max_new in requests:
            logits, past_kv = self.model(prompt_ids)
            nid = torch.argmax(logits[-1])
            generated = [nid]
            for _ in range(max_new - 1):
                logits, past_kv = self.model(nid.unsqueeze(0), past_key_values=past_kv)
                nid = torch.argmax(logits[-1])
                generated.append(nid)
            results.append(torch.cat([prompt_ids, torch.stack(generated)]))
        return results
