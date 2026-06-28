"""
step05: 朴素自回归推理引擎 (NaiveEngine)

教学要点:
  - 自回归：每次用完整历史 token 序列预测下一个 token
  - 每步调用 model.forward(全部 token)：计算量随序列长度增加
  - O(n²) 复杂度来源：Attention 需要计算所有位置对的相似度
  - 这是 vLLM / KV Cache 要解决的根本问题
"""

import torch
from torch import Tensor
from model import TinyTransformer


class NaiveEngine:
    """
    朴素推理引擎：每步都把完整 token 序列传入模型做全量前向传播。

    问题：
      - 序列长度为 n 时，Attention 计算 n × n 的相似度矩阵
      - Decode 第 k 步：K/V 矩阵需要重新计算前 k 个 token
      - 总计算量：O(1) + O(2) + ... + O(n) = O(n²)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        self.model = TinyTransformer(vocab_size, d_model, num_heads, num_layers)
        self.model.eval()

    @torch.no_grad()
    def decode_one_step(self, input_ids: Tensor) -> Tensor:
        """
        给定当前完整 token 序列，预测下一个 token。

        input_ids: [seq_len]
        返回:      标量 tensor（下一个 token 的 id）

        注意：每次调用都要把 input_ids 全部传入模型，
        即使之前计算过的 K/V 也会被重新计算！
        """
        # 全量前向：重新计算所有 token 的 Q/K/V
        logits = self.model(input_ids)  # [seq_len, vocab_size]

        # 只取最后一个位置的 logits（预测下一个 token）
        last_logits = logits[-1]        # [vocab_size]

        # Greedy 采样：选概率最大的 token
        next_id = torch.argmax(last_logits)
        return next_id

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int) -> Tensor:
        """
        自回归生成完整序列。

        prompt_ids:     [prompt_len]
        max_new_tokens: 最多生成多少个新 token
        返回:           [prompt_len + generated_len]
        """
        input_ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            next_id = self.decode_one_step(input_ids)
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)])
        return input_ids
