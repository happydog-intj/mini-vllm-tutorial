"""
adv13: Linear Attention / SSM

教学要点:
  - 标准 softmax attention: O(seq² d) 时间 + O(seq²) 显存
  - 线性 attention: 用 φ(q)φ(k)ᵀ 替代 softmax,降为 O(seq d²)
  - 递推形式: 维护状态矩阵 S,每步 O(d²),总 O(seq d²)
  - 矩阵形式: φ(Q)(φ(K)ᵀV),一次性计算,与递推等价
"""

import torch
import torch.nn as nn
import math


def standard_attention(q, k, v):
    """
    标准 softmax attention。

    Args:
        q, k, v: [seq, d]
    Returns:
        output: [seq, d]

    复杂度: O(seq² d) 时间,O(seq²) 显存（注意力矩阵）
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))  # [seq, seq]
    return torch.softmax(scores, dim=-1) @ v


def _default_feature_map(x):
    """默认特征映射: elu(x) + 1,保证输出非负,近似 exp。"""
    return torch.nn.functional.elu(x) + 1


def linear_attention(q, k, v, feature_map=None):
    """
    线性 attention —— 递推形式 (causal)。

    用 φ(q)φ(k)ᵀ 替代 softmax,然后利用结合律:
        output_t = φ(q_t) @ S_t
        S_t = S_{t-1} + outer(φ(k_t), v_t)

    Args:
        q, k, v: [seq, d]
        feature_map: 可调用,对行向量作特征映射。默认 elu+1。
    Returns:
        output: [seq, d]

    复杂度: O(seq d²) 时间,O(d²) 显存（状态矩阵 S）
    注意: 数值上与 standard_attention 不同,因为 φ 是 softmax 的近似,
          不具备 softmax 的归一化性质。
    """
    if feature_map is None:
        feature_map = _default_feature_map

    qf = feature_map(q)   # [seq, d]
    kf = feature_map(k)   # [seq, d]
    d = q.size(-1)

    S = torch.zeros(d, d, dtype=q.dtype)   # 状态矩阵 [d, d]
    out = []
    for t in range(q.size(0)):
        S = S + torch.outer(kf[t], v[t])   # 累积: S += φ(k_t) ⊗ v_t
        o = qf[t] @ S                       # 输出: φ(q_t) @ S
        out.append(o)
    return torch.stack(out)   # [seq, d]


def linear_attention_matrix(q, k, v, feature_map=None):
    """
    线性 attention —— 矩阵形式 (非 causal,全局)。

    等价于: output = φ(Q) @ (φ(K)ᵀ @ V)

    与 linear_attention 递推形式的关系:
      递推的因果 (causal) 版在 t 步只使用 1..t 的 k/v,
      矩阵形式使用全部序列,因此两者严格等价需要去掉因果性,
      或在相同的非因果设置下验证。

    本函数用于验证: 非因果矩阵形式 == 非因果递推累积(即 S 使用全部 kf/v)。

    Args:
        q, k, v: [seq, d]
        feature_map: 同 linear_attention。
    Returns:
        output: [seq, d]
    """
    if feature_map is None:
        feature_map = _default_feature_map

    qf = feature_map(q)   # [seq, d]
    kf = feature_map(k)   # [seq, d]

    # S = φ(K)ᵀ @ V  →  [d, d]
    S = kf.t() @ v         # [d, d]
    # output = φ(Q) @ S  →  [seq, d]
    return qf @ S


def linear_attention_noncausal(q, k, v, feature_map=None):
    """
    线性 attention —— 非因果递推形式。

    S 先用所有 (k, v) 构建,再对每个 q 计算输出。
    用于与 linear_attention_matrix 做 allclose 验证。

    Args:
        q, k, v: [seq, d]
        feature_map: 同 linear_attention。
    Returns:
        output: [seq, d]
    """
    if feature_map is None:
        feature_map = _default_feature_map

    qf = feature_map(q)
    kf = feature_map(k)
    d = q.size(-1)

    # 先累积所有 k/v 到状态矩阵
    S = torch.zeros(d, d, dtype=q.dtype)
    for t in range(k.size(0)):
        S = S + torch.outer(kf[t], v[t])

    # 对每个 q 输出
    out = []
    for t in range(q.size(0)):
        out.append(qf[t] @ S)
    return torch.stack(out)


class LinearAttentionLayer(nn.Module):
    """
    带可学习投影的线性 attention 层。

    输入: x  [seq, d_model]
    输出:    [seq, d_model]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return linear_attention(self.wq(x), self.wk(x), self.wv(x))
