"""
step00b: Embedding — token_id 到向量的查表过程

教学要点:
  - Embedding 本质是一个可学习的查找表（矩阵）
  - 形状：[vocab_size, d_model]
  - 操作：用 token_id 作为行索引取出对应行
  - 训练后，语义相似的词在向量空间中距离更近
"""

import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    """
    词嵌入层：将 token_id 映射为 d_model 维向量。

    内部就是一个矩阵 weight: [vocab_size, d_model]
    调用时 weight[token_id] 取出对应行。

    Args:
        vocab_size: 词表大小（token_id 的取值范围 0~vocab_size-1）
        d_model: 向量维度（模型隐层宽度）
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # 标准正态初始化，训练后会学习到语义信息
        # 形状：[vocab_size, d_model]
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model))
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        输入: token_ids  形状 [seq_len]，值域 [0, vocab_size)
        输出: 向量矩阵   形状 [seq_len, d_model]

        操作本质：
            output[i] = self.weight[token_ids[i]]
        即对每个 token_id，取出 weight 矩阵的对应行。
        """
        return self.weight[token_ids]  # 矩阵行索引，等价于 nn.Embedding


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    """
    计算两个向量（或矩阵第0行）的余弦相似度。

    公式：cos(θ) = (a·b) / (‖a‖ × ‖b‖)
    取值：[-1, 1]，越接近1越相似
    """
    a = a.flatten().float()
    b = b.flatten().float()
    return (
        torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
    ).item()
