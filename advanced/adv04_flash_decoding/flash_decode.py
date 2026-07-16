"""
adv04_flash_decoding/flash_decode.py

Flash-Decoding (split-K) 教学实现。

核心思路：
  decode 阶段每步只有 1 个新 token（单行 Q），但需要 attend 到超长 KV Cache。
  标准实现必须顺序遍历全部 KV，单个 SM 的显存带宽成为瓶颈。
  Flash-Decoding 把长 KV 按序列方向切成 num_splits 段，各段可在不同 SM 上并行；
  每段独立做局部 softmax，最后用 online-softmax 归约合并为全局结果。

注意：本文件是纯 CPU / PyTorch 教学版，用串行 for 循环模拟各段"并行"，
      不产生真实加速，仅展示 split-K + online-softmax 归约的数学正确性。
"""

import torch
import math


# ---------------------------------------------------------------------------
# 朴素 Decode Attention
# ---------------------------------------------------------------------------

def naive_decode_attention(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """单 token decode 注意力（朴素实现）。

    Args:
        q: [heads, d_head]        — 当前步的单行 Query
        K: [seq, heads, d_head]   — KV Cache 中全部历史 Key
        V: [seq, heads, d_head]   — KV Cache 中全部历史 Value

    Returns:
        output: [heads, d_head]
    """
    # scores: [seq, heads]
    scores = torch.einsum('hd,shd->sh', q, K) / math.sqrt(q.size(-1))
    # softmax 沿序列维度（dim=0）归一化
    attn = torch.softmax(scores, dim=0)          # [seq, heads]
    return torch.einsum('sh,shd->hd', attn, V)   # [heads, d_head]


# ---------------------------------------------------------------------------
# Flash-Decoding (split-K)
# ---------------------------------------------------------------------------

def flash_decode_splitk(
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_splits: int = 4,
) -> torch.Tensor:
    """Flash-Decoding split-K 教学实现。

    把长 KV 序列按 num_splits 段切分，每段独立计算局部 softmax，
    再通过 online-softmax 全局归约合并。串行循环模拟各段并行。

    Args:
        q:          [heads, d_head]
        K:          [seq, heads, d_head]
        V:          [seq, heads, d_head]
        num_splits: 切分段数（对应真实场景中并行的 SM 数量）

    Returns:
        output: [heads, d_head]，数值等价于 naive_decode_attention
    """
    seq = K.size(0)
    chunk = (seq + num_splits - 1) // num_splits  # 向上取整，最后一段可能更短

    local_outs: list[torch.Tensor] = []
    local_max:  list[torch.Tensor] = []
    local_sum:  list[torch.Tensor] = []

    # --- 第 1 步：各段独立计算局部 softmax ---
    # 真实 GPU 实现中，每段分配给不同 SM 并行执行；教学版串行模拟。
    for split_idx in range(num_splits):
        lo = split_idx * chunk
        hi = min((split_idx + 1) * chunk, seq)
        if lo >= hi:
            continue  # 尾部段为空则跳过

        Kc = K[lo:hi]  # [chunk_len, heads, d_head]
        Vc = V[lo:hi]  # [chunk_len, heads, d_head]

        # 局部 attention scores: [chunk_len, heads]
        scores = torch.einsum('hd,shd->sh', q, Kc) / math.sqrt(q.size(-1))

        # 局部 online-softmax：先求最大值以保证数值稳定
        m = scores.max(dim=0).values          # [heads] — 该段的最大值
        p = torch.exp(scores - m.unsqueeze(0))  # [chunk_len, heads] — 移位后的指数
        s = p.sum(dim=0)                      # [heads] — 局部归一化分母
        o = torch.einsum('sh,shd->hd', p, Vc) # [heads, d_head] — 局部加权 value

        local_outs.append(o)
        local_max.append(m)
        local_sum.append(s)

    # --- 第 2 步：online-softmax 全局归约 ---
    #
    # 数学推导：
    #   设两段 A、B，各自有局部最大值 m_A、m_B，局部 sum s_A、s_B，局部输出 o_A、o_B。
    #   全局最大值 Gm = max(m_A, m_B)。
    #   全局归一化分母 = s_A * exp(m_A - Gm) + s_B * exp(m_B - Gm)
    #   全局输出 = (o_A * exp(m_A - Gm) + o_B * exp(m_B - Gm)) / 全局分母
    #
    # 权重 w = exp(m_k - Gm) 把各段的局部 softmax "对齐"到同一基准，
    # 从而正确合并多段结果，等价于在整段序列上一次做 softmax。

    # 全局最大值：逐头取所有段的最大值的最大值
    Gm = torch.stack(local_max).max(dim=0).values  # [heads]

    Go = torch.zeros_like(q)  # [heads, d_head] — 累积输出分子
    Gs = torch.zeros(q.size(0), device=q.device, dtype=q.dtype)  # [heads] — 累积分母

    for o, m, s in zip(local_outs, local_max, local_sum):
        w = torch.exp(m - Gm)               # [heads] — 对齐权重
        Go += o * w.unsqueeze(-1)            # 分子：加权 value 输出
        Gs += s * w                          # 分母：加权 sum

    # 最终归一化：每头除以对应的全局分母
    return Go / Gs.unsqueeze(-1)             # [heads, d_head]
