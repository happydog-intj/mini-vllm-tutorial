"""
step03b: 多请求 KV Cache Batch 推理

新增内容（相对 step03a）：
  - SerialEngine：逐个请求串行处理（对照组）
  - BatchKVCacheEngine：多请求同步推进（Static Batching）

Static Batching 的两大问题（后续步骤解决）：
  1. 短请求完成后必须等最长请求 → 利用率低
  2. 预分配 max_len 个 KV 槽位 → 内存碎片
"""

import torch
from torch import Tensor
from typing import List, Tuple
import sys, os, importlib
_step03a_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'step03a_kvcache_single'))
if _step03a_path not in sys.path:
    sys.path.insert(0, _step03a_path)
_mod = importlib.import_module('model')
TinyTransformerWithKVCache = _mod.TinyTransformerWithKVCache


class SerialEngine:
    """串行处理引擎（对照组）：逐个请求依次生成。"""

    def __init__(self):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int) -> Tensor:
        """单请求生成（与 step03a KVCacheEngine 完全相同）"""
        logits, past_kv = self.model(prompt_ids)
        next_id = torch.argmax(logits[-1])
        generated = [next_id]
        for _ in range(max_new_tokens - 1):
            logits, past_kv = self.model(next_id.unsqueeze(0), past_key_values=past_kv)
            next_id = torch.argmax(logits[-1])
            generated.append(next_id)
        return torch.cat([prompt_ids, torch.stack(generated)])

    @torch.no_grad()
    def generate_batch(self, requests: List[Tuple[Tensor, int]]) -> List[Tensor]:
        """串行逐个处理多请求"""
        return [self.generate(p, n) for p, n in requests]


class BatchKVCacheEngine:
    """
    Static Batching 引擎：多请求同步推进直到最长请求完成。

    实现：每个请求独立维护 past_key_values，同步推进。
    核心展示「等最长请求」和「padding 浪费」这两个问题。
    """

    def __init__(self):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()

    @torch.no_grad()
    def generate_batch(
        self,
        requests: List[Tuple[Tensor, int]],
    ) -> List[Tensor]:
        batch_size = len(requests)
        max_new = max(n for _, n in requests)

        # Prefill：每个请求独立 prefill
        past_kvs = []
        last_ids = []
        for prompt_ids, _ in requests:
            logits, pkv = self.model(prompt_ids)
            past_kvs.append(pkv)
            last_ids.append(torch.argmax(logits[-1]).item())

        generated = [[lid] for lid in last_ids]
        done = [False] * batch_size

        # Decode：所有请求同步推进，等最长的完成
        for step in range(max_new - 1):
            for i, (_, max_new_i) in enumerate(requests):
                if done[i] or step >= max_new_i - 1:
                    done[i] = True
                    continue
                nid = torch.tensor([generated[i][-1]])
                logits, past_kvs[i] = self.model(nid, past_key_values=past_kvs[i])
                generated[i].append(torch.argmax(logits[-1]).item())

        results = []
        for i, (prompt_ids, _) in enumerate(requests):
            results.append(torch.cat([prompt_ids, torch.tensor(generated[i])]))
        return results
