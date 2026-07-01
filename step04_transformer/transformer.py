"""
step04: 完整 Transformer Decoder 层 + TinyTransformer

教学要点:
  - Decoder 层 = MultiHeadAttention + MLP + 残差 + RMSNorm
  - Pre-Norm：先 Norm 再 Attention/MLP（现代 LLM 的标准做法）
  - 残差连接：x = x + F(norm(x))，保证梯度能流过深层网络
  - MLP：两层线性 + 激活函数（SiLU）
  - RMSNorm vs LayerNorm：只对均方根归一化，比 LayerNorm 快
"""

import sys
import os
import torch
import torch.nn as nn
from torch import Tensor

# 引入上一步实现的 MultiHeadAttention
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step03_attention'))
from attention import MultiHeadAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。

    比 LayerNorm 更简单：不减均值，只除均方根。
    公式：output = x / RMS(x) * weight
    现代 LLM（LLaMA、Qwen3 等）都用 RMSNorm。
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # RMS(x) = sqrt(mean(x²))
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class MLP(nn.Module):
    """
    Transformer MLP 层（SiLU gate 版本，即 SwiGLU）。

    结构：
      gate = SiLU(x · W_gate)
      up   = x · W_up
      output = (gate * up) · W_down
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.W_down(self.act(self.W_gate(x)) * self.W_up(x))


class TransformerDecoderLayer(nn.Module):
    """
    单个 Transformer Decoder 层（Pre-Norm 结构）。

    数据流：
      x → norm1 → MultiHeadAttention → + x  (残差)
        → norm2 → MLP               → + x  (残差)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-Norm + 残差：先 Norm，做变换，加回原始 x
        print(f"x:{x}")
        x = x + self.attn(self.norm1(x))   # 注意力子层
        x = x + self.mlp(self.norm2(x))    # MLP 子层
        return x


class TinyTransformer(nn.Module):
    """
    用于教学的小型 Transformer 语言模型。

    结构：Embedding → N × DecoderLayer → RMSNorm → LM Head
    随机初始化权重（不会生成有意义的文字，用于演示推理流程）。

    Args:
        vocab_size: 词表大小
        d_model:    隐层维度
        num_heads:  注意力头数
        num_layers: Transformer 层数
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        d_ff = d_model * 4

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        输入: token_ids  形状 [seq_len]
        输出: logits     形状 [seq_len, vocab_size]
        """
        x = self.embed(token_ids)           # [seq_len, d_model]
        for layer in self.layers:
            x = layer(x)                    # [seq_len, d_model]
        x = self.norm(x)
        logits = self.lm_head(x)            # [seq_len, vocab_size]
        return logits
