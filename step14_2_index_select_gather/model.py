"""
step14_2: 向量化 KV gather — 消除 Python 逐 block 循环

在 step14_1 基础上的改动：
  - gather_kv_from_blocks 改用 advanced indexing，从 num_blocks 次 Python 循环降为 0 次

核心变化（对比 step07 的 TinyTransformerWithKVCache）：
  - 新增全局 kv_pool 张量：kv_pool[layer][total_blocks, block_size, num_heads, d_head]
  - attention forward 接收 block_table + current_pos，写入并 gather 历史 K/V
  - past_key_values 彻底消失，所有 KV 数据统一存储在 kv_pool 里
  - prefix cache 命中 = 只需把已缓存的 block_id 加入 block_table，零拷贝
"""

import math
import sys
import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step04_transformer'))
from transformer import RMSNorm, MLP


def gather_kv_from_blocks(
    pool: Tensor,           # [total_blocks, block_size, num_heads, d_head]
    block_table: List[int], # 物理 Block ID 列表
    seq_len: int,           # 要读取的 token 数量
    block_size: int,
) -> Tensor:
    """
    从非连续的物理 Block 中 gather 出前 seq_len 个 token 的 K 或 V。

    向量化实现：预计算每个 token 对应的物理槽位，一次 advanced indexing 完成，
    无 Python 循环，无 torch.cat。

    返回形状：[seq_len, num_heads, d_head]
    """
    positions = torch.arange(seq_len, device=pool.device)
    block_indices   = positions // block_size                    # [seq_len]
    slot_indices    = positions % block_size                     # [seq_len]
    bt              = torch.tensor(block_table, device=pool.device)
    physical_blocks = bt[block_indices]                          # [seq_len]
    return pool[physical_blocks, slot_indices]                   # [seq_len, num_heads, d_head]


class PagedMultiHeadAttention(nn.Module):
    """
    支持 block_table + kv_pool 的多头注意力。

    写入：把当前 token 的 K/V 写入 kv_pool 的对应物理槽位
    读取：从 kv_pool 按 block_table gather 出完整历史 K/V，做 attention
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
        x: Tensor,              # [seq_len, d_model]
        kv_pool_k: Tensor,      # [total_blocks, block_size, num_heads, d_head]
        kv_pool_v: Tensor,      # [total_blocks, block_size, num_heads, d_head]
        block_table: List[int], # 当前序列的物理 Block ID 列表
        start_pos: int,         # 当前 token 序列在整个序列中的起始位置
        block_size: int,
    ) -> Tensor:
        seq_len = x.size(0)

        # 计算当前输入的 Q/K/V
        Q = self.W_q(x).view(seq_len, self.num_heads, self.d_head)
        K = self.W_k(x).view(seq_len, self.num_heads, self.d_head)
        V = self.W_v(x).view(seq_len, self.num_heads, self.d_head)

        # 把当前 token 的 K/V 写入 kv_pool 对应的物理槽位（向量化，无 Python 循环）
        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        block_indices = positions // block_size                              # [seq_len]
        slot_indices  = positions % block_size                               # [seq_len]
        bt = torch.tensor(block_table, device=x.device)
        physical_blocks = bt[block_indices]                                  # [seq_len]
        kv_pool_k[physical_blocks, slot_indices] = K                        # scatter
        kv_pool_v[physical_blocks, slot_indices] = V

        # 从 kv_pool 按 block_table gather 完整历史 K/V（含刚写入的）
        total_len = start_pos + seq_len
        K_full = gather_kv_from_blocks(kv_pool_k, block_table, total_len, block_size)
        V_full = gather_kv_from_blocks(kv_pool_v, block_table, total_len, block_size)

        # 对每个头做 Attention（因果 mask）
        outputs = []
        for h in range(self.num_heads):
            q_h = Q[:, h, :]       # [seq_len, d_head]
            k_h = K_full[:, h, :]  # [total_len, d_head]
            v_h = V_full[:, h, :]

            scores = torch.matmul(q_h, k_h.T) / math.sqrt(self.d_head)

            mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device)
            for i in range(seq_len):
                mask[i, :start_pos + i + 1] = False
            scores = scores.masked_fill(mask, float("-inf"))

            weights = torch.softmax(scores, dim=-1)
            out_h = torch.matmul(weights, v_h)
            outputs.append(out_h)

        concat = torch.cat(outputs, dim=-1)
        return self.W_o(concat)


class PagedTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = PagedMultiHeadAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(
        self,
        x: Tensor,
        kv_pool_k: Tensor,
        kv_pool_v: Tensor,
        block_table: List[int],
        start_pos: int,
        block_size: int,
    ) -> Tensor:
        attn_out = self.attn(
            self.norm1(x), kv_pool_k, kv_pool_v, block_table, start_pos, block_size
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformerPaged(nn.Module):
    """
    支持 block_table + kv_pool 的 TinyTransformer。

    past_key_values 彻底消失。
    所有 KV 数据存储在 kv_pool_k / kv_pool_v 中（由外部 BlockManager 管理）。
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        total_blocks: int = 64,
        block_size: int = 16,
    ):
        super().__init__()
        d_ff = d_model * 4
        self.num_layers = num_layers
        self.block_size = block_size
        self.d_head = d_model // num_heads

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            PagedTransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 全局 KV Cache 物理存储：每层各一对 [total_blocks, block_size, num_heads, d_head]
        self.register_buffer(
            "kv_pool_k",
            torch.zeros(num_layers, total_blocks, block_size, num_heads, d_model // num_heads)
        )
        self.register_buffer(
            "kv_pool_v",
            torch.zeros(num_layers, total_blocks, block_size, num_heads, d_model // num_heads)
        )

    def forward(
        self,
        token_ids: Tensor,      # [seq_len]
        block_table: List[int], # 当前序列的物理 Block ID 列表
        start_pos: int,         # 序列起始位置（prefill=0，decode=已生成长度）
    ) -> Tensor:
        x = self.embed(token_ids)

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                self.kv_pool_k[i],  # 第 i 层的物理 KV 存储
                self.kv_pool_v[i],
                block_table,
                start_pos,
                self.block_size,
            )

        x = self.norm(x)
        return self.lm_head(x)
