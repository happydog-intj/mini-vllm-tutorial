"""
step08 V2: Paged Prefix Cache（kv_pool 版本）

相比 V1 的改进：
  - past_kv 彻底消失，所有 KV 数据存储在 model.kv_pool_k/v 里
  - prefix cache 命中 = 把已缓存的 block_id 直接加入 block_table，零拷贝
  - BlockManager 真正管理存放 KV 数据的物理显存地址空间

这是最接近真实 vLLM 设计的教学版本。
"""

import torch
from torch import Tensor
from typing import List, Tuple, Dict
import xxhash
from model_paged import TinyTransformerPaged
from block_manager import BlockManager


class PagedPrefixCacheEngineV2:
    """
    Paged Prefix Cache 引擎 V2。

    _prefix_cache: Dict[int, dict]
      key:   链式 xxhash
      value: {
        "block_ids": List[int],  # 该前缀占用的所有物理 Block ID
        "length":    int,        # 对应的 token 数量
      }
    """

    def __init__(self, block_size: int = 16, total_blocks: int = 64):
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.block_manager = BlockManager(total_blocks=total_blocks, block_size=block_size)
        self.model = TinyTransformerPaged(
            total_blocks=total_blocks,
            block_size=block_size,
        )
        self.model.eval()
        self._prefix_cache: Dict[int, dict] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _chain_hash(self, tokens: List[int], prev_hash: int, start: int, end: int) -> int:
        hh = xxhash.xxh64()
        hh.update(str(prev_hash).encode())
        hh.update(bytes(tokens[start:end]))
        return hh.intdigest()

    @torch.no_grad()
    def _generate_one(self, prompt_ids: Tensor, max_new: int) -> Tensor:
        tokens = prompt_ids.tolist()
        prompt_len = len(tokens)

        # ── 1. 查找最长命中前缀 ──
        cached_block_ids: List[int] = []
        cached_len: int = 0
        prev_hash = 0
        block_hashes: List[Tuple[int, int]] = []

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
                cached_block_ids = list(entry["block_ids"])
                cached_len = entry["length"]
                # 命中：ref_count++ for all cached blocks（零拷贝复用）
                for bid in cached_block_ids:
                    self.block_manager._blocks[bid].ref_count += 1
                self.cache_hits += 1
                break
        else:
            self.cache_misses += 1

        # ── 2. 为当前请求分配 block_table（复用命中的 + 新分配的）──
        total_needed_blocks = (prompt_len + max_new + self.block_size - 1) // self.block_size
        new_blocks_needed = total_needed_blocks - len(cached_block_ids)
        new_block_table = self.block_manager.allocate(max(new_blocks_needed, 1))
        block_table = cached_block_ids + new_block_table

        # ── 3. 增量 prefill（从 cached_len 继续，逐 block 推进，边界处缓存）──
        prev_hash_for_cache = 0
        for end, h in block_hashes:
            if end == cached_len:
                prev_hash_for_cache = h
                break

        pos = cached_len
        logits = None
        while pos < prompt_len:
            end = min(pos + self.block_size, prompt_len)
            chunk = torch.tensor(tokens[pos:end])
            logits = self.model(chunk, block_table=block_table, start_pos=pos)
            pos = end

            # 在完整 block 边界处缓存 block_id 列表
            if pos % self.block_size == 0:
                h = self._chain_hash(tokens, prev_hash_for_cache, pos - self.block_size, pos)
                if h not in self._prefix_cache:
                    prefix_block_ids = list(block_table[:pos // self.block_size])
                    for bid in prefix_block_ids:
                        self.block_manager._blocks[bid].ref_count += 1
                    self._prefix_cache[h] = {
                        "block_ids": prefix_block_ids,
                        "length":    pos,
                    }
                prev_hash_for_cache = h

        # ── 4. Decode ──
        nid = torch.argmax(logits[-1])
        generated = [nid]
        for step in range(max_new - 1):
            cur_pos = prompt_len + step
            logits = self.model(nid.unsqueeze(0), block_table=block_table, start_pos=cur_pos)
            nid = torch.argmax(logits[-1])
            generated.append(nid)

        # ── 5. 释放引用 ──
        for bid in cached_block_ids:
            self.block_manager._blocks[bid].ref_count -= 1
        self.block_manager.free(new_block_table)

        return torch.cat([prompt_ids, torch.stack(generated)])

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        return [self._generate_one(p, n) for p, n in requests]
