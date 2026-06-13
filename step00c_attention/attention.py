"""
step00c: Scaled Dot-Product Attention + Multi-Head Attention

教学要点:
  - Q（Query）：当前 token 的"问题"
  - K（Key）：各位置的"关键词"
  - V（Value）：各位置的"内容"
  - scores = Q·Kᵀ/√d：衡量 Q 和各 K 的相似度
  - 因果 mask：生成时不能看未来的 token
  - 多头：多个子空间独立做注意力，捕获不同模式
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


def scaled_dot_product_attention(
    Q: Tensor,  # [seq_len, d_head]
    K: Tensor,  # [seq_len, d_head]
    V: Tensor,  # [seq_len, d_head]
    causal: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Scaled Dot-Product Attention（单头版本，教学用）。

    计算流程：
      1. scores = Q·Kᵀ / √d_head      # [seq_len, seq_len]
      2. 应用因果 mask（可选）
      3. weights = softmax(scores)      # [seq_len, seq_len]
      4. output = weights · V           # [seq_len, d_head]

    Returns:
        output:  [seq_len, d_head]
        weights: [seq_len, seq_len]  （注意力权重，用于可视化）
    """
    d_head = Q.size(-1)

    # Step 1: 计算相似度分数
    # Q·Kᵀ 的每个元素 scores[i,j] = Q[i] 与 K[j] 的点积
    # 除以 √d_head 防止点积过大导致 softmax 梯度消失
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
    # scores: [seq_len, seq_len]

    # Step 2: 应用因果（causal）mask
    # 生成时 token i 不能看到 token j>i（未来），否则模型会"作弊"
    if causal:
        seq_len = Q.size(0)
        # tril：下三角矩阵（包含对角线），上三角置为 -inf
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    # Step 3: softmax 得到归一化的注意力权重
    # 每行的权重和为 1
    weights = torch.softmax(scores, dim=-1)

    # Step 4: 用权重对 V 加权求和
    # output[i] = Σ_j weights[i,j] * V[j]
    output = torch.matmul(weights, V)

    return output, weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力（Multi-Head Attention）。

    核心思想：将 d_model 维向量切成 num_heads 份（每份 d_head = d_model/num_heads），
    每个"头"在自己的子空间独立做注意力，捕获不同类型的依赖关系，
    最后拼回 d_model 维。

    Args:
        d_model:   模型隐层维度
        num_heads: 注意力头数（d_model 必须能被整除）
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 线性投影：将输入 x 投影为 Q、K、V
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # 输出投影：将多头输出合并回 d_model
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        输入: x  形状 [seq_len, d_model]
        输出:    形状 [seq_len, d_model]
        """
        seq_len, d_model = x.shape

        # Step 1: 投影为 Q、K、V
        Q = self.W_q(x)  # [seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: 切成多头
        # [seq_len, d_model] → [seq_len, num_heads, d_head] → [num_heads, seq_len, d_head]
        def split_heads(t: Tensor) -> Tensor:
            return t.view(seq_len, self.num_heads, self.d_head).transpose(0, 1)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        # 现在 shape: [num_heads, seq_len, d_head]

        # Step 3: 每个头独立做注意力
        outputs = []
        for h in range(self.num_heads):
            out_h, _ = scaled_dot_product_attention(Q[h], K[h], V[h], causal=True)
            outputs.append(out_h)  # [seq_len, d_head]

        # Step 4: 拼接多头输出
        # [num_heads, seq_len, d_head] → [seq_len, num_heads*d_head] = [seq_len, d_model]
        concat = torch.cat(outputs, dim=-1)  # [seq_len, d_model]

        # Step 5: 输出投影
        return self.W_o(concat)
