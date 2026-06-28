"""
step08: Paged Prefix Cache + Continuous Batching

设计要点：
  - past_kv 彻底消失，所有 KV 数据存储在 model.kv_pool_k/v 里
  - prefix cache 命中 = 把已缓存的 block_id 直接加入 block_table，零拷贝
  - BlockManager 真正管理存放 KV 数据的物理显存地址空间
  - PagedScheduler 实现 Continuous Batching：prefill 和 decode 交替执行

这是最接近真实 vLLM 设计的教学版本。
"""

import torch
import xxhash
from torch import Tensor
from typing import List, Tuple, Dict
from model import TinyTransformerPaged
from block_manager import BlockManager
from scheduler import Sequence, PagedScheduler, SequenceStatus


class PagedPrefixCacheEngine:
    """
    Paged Prefix Cache 引擎，结合 PagedScheduler 实现 Continuous Batching。

    每步循环：
      1. scheduler.schedule() 返回本轮的 prefill_seqs 和 decode_seqs
      2. prefill：查找 prefix cache，分配 block_table，逐 Block 增量 prefill
      3. decode：用 block_table 读取 kv_pool，生成下一个 token

    _prefix_cache: Dict[int, dict]
      key:   链式 xxhash
      value: {
        "block_ids": List[int],  # 该前缀占用的所有物理 Block ID
        "length":    int,        # 对应的 token 数量
      }
    """

    def __init__(self, block_size: int = 16, total_blocks: int = 128, max_running: int = 4):
        self.block_size = block_size
        self.block_manager = BlockManager(total_blocks=total_blocks, block_size=block_size)
        self.model = TinyTransformerPaged(total_blocks=total_blocks, block_size=block_size)
        self.model.eval()
        self._prefix_cache: Dict[int, dict] = {}
        self.scheduler = PagedScheduler(
            block_manager=self.block_manager,
            max_running=max_running,
            block_size=block_size,
        )
        self.cache_hits = 0
        self.cache_misses = 0

    def _chain_hash(self, tokens: List[int], prev_hash: int, start: int, end: int) -> int:
        hh = xxhash.xxh64()
        hh.update(str(prev_hash).encode())
        hh.update(bytes(tokens[start:end]))
        return hh.intdigest()

    def _lookup_prefix_cache(self, tokens: List[int]) -> Tuple[List[int], int, int]:
        """查找最长命中前缀，返回 (cached_block_ids, cached_len, prev_hash)。"""
        prompt_len = len(tokens)
        prev_hash = 0
        block_hashes = []
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
                for bid in cached_block_ids:
                    self.block_manager._blocks[bid].ref_count += 1
                self.cache_hits += 1
                return cached_block_ids, entry["length"], h
        self.cache_misses += 1
        return [], 0, 0

    def _save_prefix_cache(self, tokens: List[int], block_table: List[int],
                           pos: int, prev_hash: int) -> int:
        """在 block 边界处保存 prefix cache，返回新的 prev_hash。"""
        h = self._chain_hash(tokens, prev_hash, pos - self.block_size, pos)
        if h not in self._prefix_cache:
            prefix_block_ids = list(block_table[:pos // self.block_size])
            for bid in prefix_block_ids:
                self.block_manager._blocks[bid].ref_count += 1
            self._prefix_cache[h] = {"block_ids": prefix_block_ids, "length": pos}
        return h

    @torch.no_grad()
    def _do_prefill_step(self, seq: Sequence):
        """对 seq 做一块 prefill（一个 block_size 的 chunk）。"""
        prompt_len = len(seq.prompt_ids)

        # 首次进入 prefill：查找 prefix cache，分配 block_table
        if seq.prefill_offset == 0 and not seq.block_table:
            cached_block_ids, cached_len, prev_hash = self._lookup_prefix_cache(
                seq.prompt_ids.tolist()
            )
            seq.prefill_offset = cached_len
            seq._prev_hash = prev_hash  # 挂在 seq 上，后续 block 边界缓存时使用

            # 分配足够的新 block（前缀命中的 block 复用，剩余部分新分配）
            total_needed = (prompt_len + seq.max_new_tokens + self.block_size - 1) // self.block_size
            new_needed = max(total_needed - len(cached_block_ids), 1)
            new_blocks = self.block_manager.allocate(new_needed)
            seq.block_table = cached_block_ids + new_blocks
            seq._new_blocks = new_blocks  # 记录新分配的，用于释放时区分

        pos = seq.prefill_offset
        end = min(pos + self.block_size, prompt_len)
        chunk = seq.prompt_ids[pos:end]
        logits = self.model(chunk, block_table=seq.block_table, start_pos=pos)
        seq.prefill_offset = end

        # block 边界处缓存
        if end % self.block_size == 0 and end <= prompt_len:
            seq._prev_hash = self._save_prefix_cache(
                seq.prompt_ids.tolist(), seq.block_table, end, seq._prev_hash
            )

        # prefill 完成：采样第一个 token
        if seq.prefill_done:
            seq.append_token(torch.argmax(logits[-1]).item())
            seq.status = SequenceStatus.RUNNING

    @torch.no_grad()
    def _do_decode_step(self, seq: Sequence):
        """对 seq 做一步 decode。"""
        cur_pos = len(seq.token_ids) - 1
        logits = self.model(seq.get_last_token(), block_table=seq.block_table, start_pos=cur_pos)
        seq.append_token(torch.argmax(logits[-1]).item())

    def _free_seq(self, seq: Sequence):
        """释放 seq 占用的资源（prefix cache 引用 + 新分配的 block）。"""
        # 释放命中的 prefix cache 引用
        cached_ids = seq.block_table[:len(seq.block_table) - len(getattr(seq, '_new_blocks', []))]
        for bid in cached_ids:
            self.block_manager._blocks[bid].ref_count -= 1
        # 释放新分配的 block
        self.block_manager.free(getattr(seq, '_new_blocks', []))

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        seqs = []
        for prompt_ids, max_new in requests:
            seq = Sequence(prompt_ids, max_new)
            self.scheduler.add(seq)
            seqs.append(seq)

        while self.scheduler.has_work:
            prefill_seqs, decode_seqs = self.scheduler.schedule()

            for seq in prefill_seqs:
                self._do_prefill_step(seq)

            for seq in decode_seqs:
                self._do_decode_step(seq)

            # 释放本轮刚完成的请求的资源
            for seq in seqs:
                if seq.status == SequenceStatus.FINISHED and hasattr(seq, '_new_blocks'):
                    self._free_seq(seq)
                    del seq._new_blocks  # 防止重复释放

        return [torch.tensor(s.token_ids) for s in seqs]
