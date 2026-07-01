"""
step07: 支持 KV Cache 的 TinyTransformer

与 step05 的区别：
  - MultiHeadAttention 新增 past_kv 参数
  - Prefill 模式：past_kv=None，正常计算
  - Decode 模式：past_kv 提供历史 K/V，只计算新 token
"""

import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Tuple

# 复用 step04 的基础组件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step04_transformer'))
from transformer import RMSNorm, MLP

KVCache = Tuple[Tensor, Tensor]  # (K, V) 各形状 [total_seq_len, num_heads, d_head]


class MultiHeadAttentionWithKVCache(nn.Module):
    """
    支持 KV Cache 的多头注意力。

    Prefill (past_kv=None): 计算所有 token 的 Q/K/V，返回完整 KV
    Decode  (past_kv=(K_past, V_past)): 只计算新 token 的 K/V，拼到历史上
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Tensor,                         # [seq_len, d_model]
        past_kv: Optional[KVCache] = None,
    ) -> Tuple[Tensor, KVCache]:
        seq_len = x.size(0)

        # 计算当前输入的 Q/K/V
        Q = self.W_q(x).view(seq_len, self.num_heads, self.d_head)
        K = self.W_k(x).view(seq_len, self.num_heads, self.d_head)
        V = self.W_v(x).view(seq_len, self.num_heads, self.d_head)

        # KV Cache：拼接历史 K/V
        if past_kv is not None:
            K_past, V_past = past_kv
            K_full = torch.cat([K_past, K], dim=0)
            V_full = torch.cat([V_past, V], dim=0)
        else:
            K_full = K
            V_full = V

        total_len = K_full.size(0)

        # 对每个头做 Attention
        outputs = []
        for h in range(self.num_heads):
            q_h = Q[:, h, :]       # [seq_len, d_head]
            k_h = K_full[:, h, :]  # [total_len, d_head]
            v_h = V_full[:, h, :]
            print(f"Head\n {h}: q_h=\n{q_h}, k_h=\n{k_h}, v_h=\n{v_h}")

            scores = torch.matmul(q_h, k_h.T) / math.sqrt(self.d_head)
            # [seq_len, total_len]
            print(f"Head {h}: scores=\n{scores}")

            # 因果 mask
            past_len = total_len - seq_len
            mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device)
            for i in range(seq_len):
                mask[i, :past_len + i + 1] = False
            scores = scores.masked_fill(mask, float("-inf"))
            print(f"Head {h}: masked scores=\n{scores}")
            weights = torch.softmax(scores, dim=-1)
            out_h = torch.matmul(weights, v_h)
            outputs.append(out_h)

        concat = torch.cat(outputs, dim=-1)
        output = self.W_o(concat)
        return output, (K_full, V_full)


class TransformerDecoderLayerWithKV(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttentionWithKVCache(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(
        self,
        x: Tensor,
        past_kv: Optional[KVCache] = None,
    ) -> Tuple[Tensor, KVCache]:
        print(f"x:{x}")
        attn_out, new_kv = self.attn(self.norm1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class TinyTransformerWithKVCache(nn.Module):
    """
    支持 KV Cache 的 TinyTransformer。

    past_key_values: None → prefill（全量前向）
                     list[KVCache] → decode（每层的历史 K/V）
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 4,#128,
        num_heads: int = 1,#4,
        num_layers: int = 1,#2,
    ):
        super().__init__()
        d_ff = d_model * 4
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayerWithKV(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        token_ids: Tensor,
        past_key_values: Optional[List[KVCache]] = None,
    ) -> Tuple[Tensor, List[KVCache]]:
        x = self.embed(token_ids)

        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = layer(x, past_kv)
            new_past_key_values.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_past_key_values
