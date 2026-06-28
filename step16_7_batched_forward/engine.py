"""
step16_7: Paged Prefix Cache + Batched Forward

在 step16_6 基础上的改动：
  - generate_batch 主循环把所有 prefill/decode 序列收集后，
    一次调用 model.forward_batched，Linear 层真正批处理
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

    def _prepare_prefill(self, seq: Sequence):
        """首次进入 prefill 时初始化 block_table（prefix cache lookup + 分配）。"""
        if seq.prefill_offset == 0 and not seq.block_table:
            prompt_len = len(seq.prompt_ids)
            cached_block_ids, cached_len, _ = self._lookup_prefix_cache(seq)
            seq.prefill_offset = cached_len
            total_needed = (prompt_len + seq.max_new_tokens + self.block_size - 1) // self.block_size
            new_needed = max(total_needed - len(cached_block_ids), 1)
            new_blocks = self.block_manager.allocate(new_needed)
            seq.block_table = cached_block_ids + new_blocks
            seq._new_blocks = new_blocks

    @torch.no_grad()
    def _run_batched_step(self, prefill_seqs: List[Sequence], decode_seqs: List[Sequence]):
        """
        把所有 prefill chunk 和 decode token 拼成一个 flat batch，
        一次 forward_batched 完成所有 Linear 层计算。
        """
        all_seqs: List[Sequence] = []
        tokens_list: List[Tensor] = []
        block_tables: List[List[int]] = []
        start_positions: List[int] = []

        for seq in prefill_seqs:
            self._prepare_prefill(seq)
            pos = seq.prefill_offset
            end = min(pos + self.block_size, len(seq.prompt_ids))
            tokens_list.append(seq.prompt_ids[pos:end])
            block_tables.append(seq.block_table)
            start_positions.append(pos)
            all_seqs.append(('prefill', seq, pos, end))

        for seq in decode_seqs:
            cur_pos = len(seq.token_ids) - 1
            tokens_list.append(seq.get_last_token())
            block_tables.append(seq.block_table)
            start_positions.append(cur_pos)
            all_seqs.append(('decode', seq, cur_pos, None))

        if not tokens_list:
            return

        # 拼接所有 token，构造 cu_seqlens
        token_tensor = torch.cat(tokens_list, dim=0)       # [total_tokens]
        cu_seqlens = [0]
        for t in tokens_list:
            cu_seqlens.append(cu_seqlens[-1] + t.size(0))

        # 一次 forward_batched
        all_logits = self.model.forward_batched(
            token_tensor, block_tables, start_positions, cu_seqlens
        )  # [total_tokens, vocab_size]

        # 按序列取结果
        for idx, entry in enumerate(all_seqs):
            kind, seq, pos, end = entry
            s, e = cu_seqlens[idx], cu_seqlens[idx + 1]
            logits = all_logits[s:e]

            if kind == 'prefill':
                seq.prefill_offset = end
                if end % self.block_size == 0 and end <= len(seq.prompt_ids):
                    self._save_prefix_cache(seq, end)
                if seq.prefill_done:
                    seq.append_token(torch.argmax(logits[-1]).item())
                    seq.status = SequenceStatus.RUNNING
            else:
                seq.append_token(torch.argmax(logits[-1]).item())

    def _free_seq(self, seq: Sequence):
        """释放 seq 占用的资源（prefix cache 引用 + 新分配的 block）。"""
        cached_count = len(seq.block_table) - len(getattr(seq, '_new_blocks', []))
        self.block_manager.release(seq.block_table[:cached_count])
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
            self._run_batched_step(prefill_seqs, decode_seqs)

            for seq in seqs:
                if seq.status == SequenceStatus.FINISHED and hasattr(seq, '_new_blocks'):
                    self._free_seq(seq)
                    del seq._new_blocks

        return [torch.tensor(s.token_ids) for s in seqs]
