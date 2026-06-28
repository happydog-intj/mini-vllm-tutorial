"""
step08: Paged Prefix Cache

结合 step06（Block 粒度分页）和 step07（前缀复用）：
  - 以 Block 为单位缓存前缀 KV 快照
  - 命中时 ref_count++，直接复用，无需重新 prefill
  - 未命中时逐 Block 增量 prefill，在边界处保存正确的 KV 快照
  - 释放时 ref_count--，归零才真正回收 Block

相比 step07 的改进：
  1. 缓存的 KV 快照是正确的（前 N 个 token 的状态，不含后续 token）
  2. Block 粒度管理，支持跨请求共享物理 Block
  3. ref_count 控制 Block 生命周期，避免提前释放
"""

import torch
from torch import Tensor
from typing import List, Tuple, Dict, Optional
import xxhash
from model import TinyTransformerWithKVCache
from block_manager import BlockManager


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


class PagedPrefixCacheEngine:
    """
    Paged Prefix Cache 引擎。

    _prefix_cache: Dict[int, dict]
      key: 链式 xxhash（代表前 N 个 token 的前缀）
      value: {
        "block_id": int,       # 该前缀末尾 Block 的物理 ID（用于 ref_count 管理）
        "past_kv":  object,    # 前 N 个 token prefill 完成时的 KV 快照（正确的边界状态）
        "length":   int,       # 该快照对应的 token 数量
      }
    """

    def __init__(self, block_size: int = 16, total_blocks: int = 64):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()
        self.block_size = block_size
        self.block_manager = BlockManager(total_blocks=total_blocks, block_size=block_size)
        # hash → {"block_id": int, "past_kv": ..., "length": int}
        self._prefix_cache: Dict[int, dict] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _chain_hash(self, tokens: List[int], prev_hash: int, start: int, end: int) -> int:
        """计算 tokens[start:end] 在链式 hash 下的 hash 值。"""
        hh = xxhash.xxh64()
        hh.update(str(prev_hash).encode())
        hh.update(bytes(tokens[start:end]))
        return hh.intdigest()

    @torch.no_grad()
    def _generate_one(self, prompt_ids: Tensor, max_new: int) -> Tensor:
        tokens = prompt_ids.tolist()
        prompt_len = len(tokens)

        # ── 1. 查找最长命中前缀（从长到短，步长=block_size）──
        cached_kv: Optional[object] = None
        cached_len: int = 0
        cached_hash: int = 0
        prev_hash = 0
        block_hashes: List[Tuple[int, int]] = []  # [(end, hash), ...]

        # 预计算每个 block 边界的链式 hash
        for start in range(0, prompt_len - prompt_len % self.block_size, self.block_size):
            end = start + self.block_size
            if end > prompt_len:
                break
            h = self._chain_hash(tokens, prev_hash, start, end)
            block_hashes.append((end, h))
            prev_hash = h

        # 从最长前缀开始查找
        for end, h in reversed(block_hashes):
            if h in self._prefix_cache:
                entry = self._prefix_cache[h]
                cached_kv = entry["past_kv"]
                cached_len = entry["length"]
                cached_hash = h
                # 命中：ref_count++（该 block 被当前请求引用）
                self.block_manager._blocks[entry["block_id"]].ref_count += 1
                self.cache_hits += 1
                break
        else:
            self.cache_misses += 1

        # ── 2. 增量 Prefill（从 cached_len 开始，逐 block 推进）──
        past_kv = cached_kv
        prev_hash = cached_hash

        # 找到 cached_len 对应的 prev_hash
        if cached_len > 0:
            for end, h in block_hashes:
                if end == cached_len:
                    prev_hash = h
                    break

        # 逐 block 增量 prefill，在每个 block 边界处保存正确快照
        pos = cached_len
        while pos < prompt_len:
            end = min(pos + self.block_size, prompt_len)
            chunk = torch.tensor(tokens[pos:end])
            logits, past_kv = self.model(chunk, past_key_values=past_kv)
            pos = end

            # 只在完整 block 边界处缓存快照（不缓存 prompt 末尾的残余块）
            if pos % self.block_size == 0 and pos <= prompt_len:
                h = self._chain_hash(tokens, prev_hash, pos - self.block_size, pos)
                if h not in self._prefix_cache:
                    # 分配一个 Block 用于跟踪引用计数
                    try:
                        blk_table = self.block_manager.allocate(1)
                        self._prefix_cache[h] = {
                            "block_id": blk_table[0],
                            "past_kv":  past_kv,   # ← 正确：此时 past_kv 只含前 pos 个 token
                            "length":   pos,
                        }
                    except RuntimeError:
                        pass  # Block 池满时跳过缓存，不影响正确性
                prev_hash = h

        # ── 3. Decode ──
        nid = torch.argmax(logits[-1])
        generated = [nid]
        for _ in range(max_new - 1):
            logits, past_kv = self.model(nid.unsqueeze(0), past_key_values=past_kv)
            nid = torch.argmax(logits[-1])
            generated.append(nid)

        # 释放命中的缓存 block 的引用（ref_count--）
        if cached_kv is not None and cached_hash in self._prefix_cache:
            entry = self._prefix_cache[cached_hash]
            blk = self.block_manager._blocks[entry["block_id"]]
            blk.ref_count -= 1
            # 注意：ref_count > 0 时不回收（可能还有其他请求在用）

        return torch.cat([prompt_ids, torch.stack(generated)])

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        return [self._generate_one(p, n) for p, n in requests]
