"""
step08: 多请求 KV Cache Batch 推理

新增内容（相对 step07）：
  - SerialEngine：逐个请求串行处理（对照组）
  - BatchKVCacheEngine：把多请求 pad 到同一长度，一次 batch forward

Static Batching 的核心机制与问题：
  - 优势：一次矩阵乘法处理多请求，GPU 并行利用率高
  - 问题 1：不同长度必须 pad 到最长 → 约 30-50% 计算浪费在 <pad> token 上
  - 问题 2：短请求完成后必须等最长请求 → 槽位空转
  注：CPU 上矩阵运算无并行加速，本步重点是理解 padding 浪费的结构
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional
import sys, os, importlib

_step03a_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'step07_kvcache_single')
)
if _step03a_path not in sys.path:
    sys.path.insert(0, _step03a_path)
_mod = importlib.import_module('model')
TinyTransformerWithKVCache = _mod.TinyTransformerWithKVCache

PAD_ID = 0  # padding token id


class SerialEngine:
    """串行处理引擎（对照组）：逐个请求依次生成。"""

    def __init__(self):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int) -> Tensor:
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
        return [self.generate(p, n) for p, n in requests]


# ─────────────────────────────────────────────────────────────────────
# 支持 batch 维度的模型封装
# step07 的 TinyTransformerWithKVCache 只支持 [seq_len]，
# 这里包装一层使其支持 [batch, seq_len] 的 Prefill 阶段。
# ─────────────────────────────────────────────────────────────────────

class BatchPrefillWrapper:
    """
    把 [batch, seq_len] 的 padded batch 拆开逐条 prefill，
    并跟踪每条序列各自的 past_key_values。

    真实 vLLM / TensorRT 中 prefill 直接用 batch matmul + attention mask，
    这里教学版用拆分法保持 model 代码不变，重点展示 padding 结构。
    """

    def __init__(self, model):
        self.model = model

    def prefill_batch(
        self,
        padded_ids: Tensor,      # [batch, max_seq_len]  含 PAD_ID
        lengths: List[int],       # 每条序列的真实长度（去掉 padding）
    ):
        """
        返回：
          past_kvs: list[batch] of past_key_values（每条序列自己的 KV Cache）
          last_logits: [batch, vocab_size]（每条序列最后一个真实 token 的 logits）
        """
        batch_size = padded_ids.size(0)
        past_kvs = []
        last_logits_list = []

        for i in range(batch_size):
            # 只取真实 token（去掉 padding）
            real_ids = padded_ids[i, :lengths[i]]
            logits, pkv = self.model(real_ids)
            past_kvs.append(pkv)
            last_logits_list.append(logits[-1])  # 最后一个真实 token 的 logits

        return past_kvs, torch.stack(last_logits_list)  # [batch, vocab_size]


class BatchKVCacheEngine:
    """
    Static Batching 引擎。

    Prefill 阶段：
      1. 所有 prompt pad 到 max_prompt_len，拼成 [batch, max_prompt_len]
      2. 逐条取真实 token 做 prefill（教学版；真实实现用 batch matmul + mask）
      3. 展示 padding 浪费：padding token 占用了多少计算槽位

    Decode 阶段：
      - 所有请求同步推进，等最长请求完成
      - 已完成的请求继续占用槽位（Static Batching 的核心缺陷，step09 解决）
    """

    def __init__(self):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()
        self._wrapper = BatchPrefillWrapper(self.model)

        # 统计信息（供 run.py 读取）
        self.padded_prefill_slots = 0   # 含 padding 的 prefill 总槽位数
        self.actual_prefill_tokens = 0  # 真实 prompt token 数

    @torch.no_grad()
    def generate_batch(
        self,
        requests: List[Tuple[Tensor, int]],
    ) -> List[Tensor]:
        batch_size = len(requests)
        prompt_lengths = [len(p) for p, _ in requests]
        max_prompt_len = max(prompt_lengths)
        max_new = max(n for _, n in requests)

        # ── Prefill：pad 所有 prompt 到 max_prompt_len ──────────────
        # 这一步展示了 Static Batching 必须做的事情：
        #   短 prompt 被补 PAD_ID 到和最长 prompt 一样长
        padded = torch.full((batch_size, max_prompt_len), PAD_ID, dtype=torch.long)
        for i, (prompt_ids, _) in enumerate(requests):
            padded[i, :len(prompt_ids)] = prompt_ids  # 左对齐，右边补 pad

        # 记录 padding 浪费量
        self.padded_prefill_slots = batch_size * max_prompt_len
        self.actual_prefill_tokens = sum(prompt_lengths)

        # Prefill：每条序列用真实长度 prefill（不包含 pad token）
        past_kvs, last_logits = self._wrapper.prefill_batch(padded, prompt_lengths)

        # ── Decode：同步推进所有请求 ────────────────────────────────
        # 短请求完成后继续等待最长请求，其 KV 槽位不释放
        # （这是 Static Batching 的第二个缺陷，step09 的 Continuous Batching 解决）
        generated = [[torch.argmax(last_logits[i]).item()] for i in range(batch_size)]
        done = [False] * batch_size

        for step in range(max_new - 1):
            for i, (_, max_new_i) in enumerate(requests):
                if done[i] or step >= max_new_i - 1:
                    done[i] = True
                    continue  # 已完成，但本轮仍占用内存槽位 ← Static Batching 浪费
                nid = torch.tensor([generated[i][-1]])
                logits, past_kvs[i] = self.model(nid, past_key_values=past_kvs[i])
                generated[i].append(torch.argmax(logits[-1]).item())

        results = []
        for i, (prompt_ids, _) in enumerate(requests):
            results.append(torch.cat([prompt_ids, torch.tensor(generated[i])]))
        return results
