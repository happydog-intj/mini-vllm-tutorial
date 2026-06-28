"""
step13: NoPrefixCacheEngine（对照）+ PrefixCacheEngine（本步核心）

Prefix Caching：相同前缀的 token 序列有相同的 K/V，
可以跨请求共享，只需计算一次。
链式 xxhash 保证不同前缀的 hash 唯一。
"""

import torch
from torch import Tensor
from typing import List, Tuple, Dict
import xxhash
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


class PrefixCacheEngine:
    """
    带 Prefix Caching 的引擎。

    _prefix_kv_cache: Dict[int, past_key_values]
      按 xxhash 索引已计算的 prompt 前缀 KV Cache。
      相同前缀命中时跳过 prefill，直接从断点继续。

    链式 hash：
      h0 = xxhash64(tokens[0:B])
      h1 = xxhash64(str(h0) + tokens[B:2B])
      ...
      保证：不同前缀→不同 hash（无误命中）
    """

    def __init__(self, block_size: int = 16):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()
        self.block_size = block_size
        self._prefix_kv_cache: Dict[int, object] = {}
        self.cache_hits = 0

    def _compute_hash(self, tokens: List[int], up_to: int) -> int:
        """计算 tokens[0:up_to] 的链式 xxhash（只对完整 Block 边界计算）"""
        h = 0
        for start in range(0, up_to, self.block_size):
            end = min(start + self.block_size, up_to)
            if end - start < self.block_size:
                break
            hh = xxhash.xxh64()
            hh.update(str(h).encode())
            hh.update(bytes(tokens[start:end]))
            h = hh.intdigest()
        return h

    @torch.no_grad()
    def _generate_one(self, prompt_ids: Tensor, max_new: int) -> Tensor:
        tokens = prompt_ids.tolist()
        prompt_len = len(tokens)

        # 查找最长缓存前缀（从长到短，步长=block_size）
        cached_kv = None
        cached_len = 0
        for end in range((prompt_len // self.block_size) * self.block_size,
                         0, -self.block_size):
            h = self._compute_hash(tokens, end)
            if h in self._prefix_kv_cache:
                cached_kv = self._prefix_kv_cache[h]
                cached_len = end
                self.cache_hits += 1
                break

        # Prefill（从 cached_len 继续）
        if cached_len > 0 and cached_kv is not None:
            remaining = torch.tensor(tokens[cached_len:])
            if len(remaining) > 0:
                logits, past_kv = self.model(remaining, past_key_values=cached_kv)
            else:
                logits, past_kv = self.model(
                    torch.tensor([tokens[-1]]), past_key_values=cached_kv
                )
        else:
            logits, past_kv = self.model(prompt_ids)

        # 缓存当前 prompt 的 KV（按 Block 边界）
        for end in range(self.block_size, prompt_len + 1, self.block_size):
            h = self._compute_hash(tokens, end)
            if h not in self._prefix_kv_cache:
                self._prefix_kv_cache[h] = past_kv

        # Decode
        nid = torch.argmax(logits[-1])
        generated = [nid]
        for _ in range(max_new - 1):
            logits, past_kv = self.model(nid.unsqueeze(0), past_key_values=past_kv)
            nid = torch.argmax(logits[-1])
            generated.append(nid)

        return torch.cat([prompt_ids, torch.stack(generated)])

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        return [self._generate_one(p, n) for p, n in requests]
