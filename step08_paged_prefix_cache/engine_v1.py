"""
step08 V1: Paged Prefix Cache（past_kv 版本）

相比 step07 的改进：
  1. 逐 Block 增量 prefill，在边界处保存正确的 KV 快照
  2. block_id 做 ref_count，控制 Block 生命周期
  3. 跨请求共享元数据（但 past_kv 仍是 Python 对象）

局限：past_kv 仍是游离的 Python tensor，不在 BlockManager 管理的显存里。
     → V2（engine_v2.py）解决这个问题。
"""

import torch
from torch import Tensor
from typing import List, Tuple, Dict, Optional
import xxhash
from model import TinyTransformerWithKVCache
from block_manager import BlockManager


class PagedPrefixCacheEngine:
    """
    Paged Prefix Cache 引擎 V1。

    _prefix_cache: Dict[int, dict]
      key:   链式 xxhash（代表前 N 个 token 的前缀）
      value: {
        "block_id": int,    # 该前缀末尾 Block 的物理 ID（用于 ref_count 管理）
        "past_kv":  object, # 前 N 个 token prefill 完成时的 KV 快照（正确的边界状态）
        "length":   int,    # 该快照对应的 token 数量
      }
    """

    def __init__(self, block_size: int = 16, total_blocks: int = 64):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()
        self.block_size = block_size
        self.block_manager = BlockManager(total_blocks=total_blocks, block_size=block_size)
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

        for start in range(0, prompt_len - prompt_len % self.block_size, self.block_size):
            end = start + self.block_size
            if end > prompt_len:
                break
            h = self._chain_hash(tokens, prev_hash, start, end)
            block_hashes.append((end, h))
            prev_hash = h

        for end, h in reversed(block_hashes):
            if h in self._prefix_cache:
                entry = self._prefix_cache[h]
                cached_kv = entry["past_kv"]
                cached_len = entry["length"]
                cached_hash = h
                self.block_manager._blocks[entry["block_id"]].ref_count += 1
                self.cache_hits += 1
                break
        else:
            self.cache_misses += 1

        # ── 2. 增量 Prefill（从 cached_len 开始，逐 block 推进）──
        past_kv = cached_kv
        prev_hash = cached_hash

        if cached_len > 0:
            for end, h in block_hashes:
                if end == cached_len:
                    prev_hash = h
                    break

        pos = cached_len
        while pos < prompt_len:
            end = min(pos + self.block_size, prompt_len)
            chunk = torch.tensor(tokens[pos:end])
            logits, past_kv = self.model(chunk, past_key_values=past_kv)
            pos = end

            # 只在完整 block 边界处缓存快照
            if pos % self.block_size == 0 and pos <= prompt_len:
                h = self._chain_hash(tokens, prev_hash, pos - self.block_size, pos)
                if h not in self._prefix_cache:
                    try:
                        blk_table = self.block_manager.allocate(1)
                        self._prefix_cache[h] = {
                            "block_id": blk_table[0],
                            "past_kv":  past_kv,  # ← 正确：此时 past_kv 只含前 pos 个 token
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

        # ── 4. 释放命中 block 的引用（ref_count--）──
        if cached_kv is not None and cached_hash in self._prefix_cache:
            entry = self._prefix_cache[cached_hash]
            self.block_manager._blocks[entry["block_id"]].ref_count -= 1

        return torch.cat([prompt_ids, torch.stack(generated)])

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        return [self._generate_one(p, n) for p, n in requests]
