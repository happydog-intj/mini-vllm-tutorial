"""
step16_6: Paged Prefix Cache + BlockManager.release()

在 step16_5 基础上的改动：
  - _free_seq 改用 block_manager.release() 释放 prefix cache 引用
  - 消除直接操作 block_manager._blocks 内部状态
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

    def _precompute_block_hashes(self, seq: Sequence):
        """冷请求首次 lookup 前，一次性预计算所有完整 block 的 hash 链。"""
        tokens = seq.prompt_ids.tolist()
        prompt_len = len(tokens)
        prev_hash = 0
        for start in range(0, prompt_len - prompt_len % self.block_size, self.block_size):
            h = self._chain_hash(tokens, prev_hash, start, start + self.block_size)
            seq._block_hashes.append(h)
            prev_hash = h
        seq._prev_hash = prev_hash

    def _lookup_prefix_cache(self, seq: Sequence) -> Tuple[List[int], int, int]:
        """查找最长命中前缀，返回 (cached_block_ids, cached_len, last_matched_hash)。

        利用 seq._block_hashes 直接查 dict，不重算 hash。
        冷请求首次调用时先预计算一次。
        """
        if not seq._block_hashes:
            self._precompute_block_hashes(seq)

        for i in range(len(seq._block_hashes) - 1, -1, -1):
            h = seq._block_hashes[i]
            if h in self._prefix_cache:
                entry = self._prefix_cache[h]
                cached_block_ids = list(entry["block_ids"])
                self.block_manager.retain(cached_block_ids)
                self.cache_hits += 1
                return cached_block_ids, entry["length"], h
        self.cache_misses += 1
        return [], 0, 0

    def _save_prefix_cache(self, seq: Sequence, pos: int):
        """在 block 边界处增量保存 prefix cache，同步更新 seq._prev_hash。"""
        # hash 在预计算时已算好，直接取
        block_idx = pos // self.block_size - 1
        h = seq._block_hashes[block_idx]
        if h not in self._prefix_cache:
            prefix_block_ids = list(seq.block_table[:pos // self.block_size])
            self.block_manager.retain(prefix_block_ids)
            self._prefix_cache[h] = {"block_ids": prefix_block_ids, "length": pos}

    @torch.no_grad()
    def _do_prefill_step(self, seq: Sequence):
        """对 seq 做一块 prefill（一个 block_size 的 chunk）。"""
        prompt_len = len(seq.prompt_ids)

        # 首次进入 prefill：查找 prefix cache，分配 block_table
        if seq.prefill_offset == 0 and not seq.block_table:
            cached_block_ids, cached_len, _ = self._lookup_prefix_cache(seq)
            seq.prefill_offset = cached_len

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

        # block 边界处缓存（hash 已在 _block_hashes 里，无需重算）
        if end % self.block_size == 0 and end <= prompt_len:
            self._save_prefix_cache(seq, end)

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
        cached_count = len(seq.block_table) - len(getattr(seq, '_new_blocks', []))
        # prefix cache 引用的 block：ref_count -= 1，归零才回收
        self.block_manager.release(seq.block_table[:cached_count])
        # 本次新分配的 block：直接回收
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
