"""
adv03 投机解码核心逻辑

教学版简化说明:
  • target_verify 对 context + draft_tokens 整体做一次前向,用因果 mask 同时得到
    所有草稿位置的目标预测,无需 KV Cache 跨轮状态管理,逻辑最清晰。
  • 验收标准: argmax 直接比对(真实框架用概率比 r = p_target/p_draft 采样)。
  • 结果等价性: 只要 target 是确定性的(argmax),投机解码结果与纯自回归完全一致。
"""

import torch
from typing import List, Tuple
from torch import Tensor


def draft_speculate(draft_model, context_ids: Tensor, k: int = 4):
    """草稿模型自回归生成 k 个候选 token。

    流程:
      1. 对完整 context 做一次 prefill(复用 KV Cache)
      2. 之后逐 token 解码,共 k 次 decode forward

    Args:
        draft_model  : 小型草稿模型 (与目标同 vocab_size)
        context_ids  : 当前完整上下文 [seq_len]
        k            : 每轮投机步数

    Returns:
        draft_tokens (list[int])    : k 个候选 token
        draft_probs  (list[Tensor]) : 对应的概率分布 [vocab_size]
    """
    tokens: List[int] = []
    probs:  List[Tensor] = []
    kv = None

    # Prefill: 让草稿模型看到完整上下文,拿到首个 logit 和 KV Cache
    logits, kv = draft_model(context_ids, past_key_values=None)

    for _ in range(k):
        p = torch.softmax(logits[-1], dim=-1)
        nxt = p.argmax().item()
        tokens.append(nxt)
        probs.append(p.detach())
        # 仅传入新生成的单 token,复用 KV Cache 做增量解码
        logits, kv = draft_model(torch.tensor([nxt]), past_key_values=kv)

    return tokens, probs


def target_verify(
    target_model,
    context_ids: Tensor,
    draft_tokens: List[int],
    draft_probs: List[Tensor],
) -> List[int]:
    """目标模型一次并行前向验证全部草稿 token (1 次 target forward)。

    核心技巧: 将 context_ids ++ draft_tokens 拼成整体输入,因果 mask 天然保证
    每个位置只看到自身及之前的 token。

        context  |  draft_tokens
        [p0 p1 … pL-1 | d0 d1 d2 … dk-1]

    logits[L-1+j] = 目标模型看到 context+d0+…+d_{j-1} 后对第 j 个草稿位置的预测
    logits[L-1+0] = 目标对紧跟 context 之后位置的预测 = 纯自回归下的 t1

    教学版: argmax 比对(真实框架用概率比接受/拒绝)。

    Args:
        target_model : 目标模型
        context_ids  : 当前完整上下文 [seq_len]
        draft_tokens : 草稿模型生成的 k 个候选 token
        draft_probs  : 草稿概率(教学版未使用;真实框架用于概率比)

    Returns:
        accepted (list[int]): 接受的 token 列表
            - 遇到首个不匹配即停,但将目标模型的正确 token 保留在末尾
            - 长度范围: [1, k]  (至少接受 1 个修正 token)
    """
    all_ids = torch.cat([
        context_ids,
        torch.tensor(draft_tokens, dtype=torch.long),
    ])
    logits, _ = target_model(all_ids, past_key_values=None)

    # logits[L-1+j] 是对 draft_tokens[j] 位置的目标预测
    # L = len(context_ids)
    offset = len(context_ids) - 1

    accepted: List[int] = []
    for j, dt in enumerate(draft_tokens):
        tgt = logits[offset + j].argmax().item()
        if tgt == dt:
            accepted.append(dt)          # ✓ 接受草稿 token
        else:
            accepted.append(tgt)         # ✗ 拒绝, 插入目标模型的 token, 后续草稿丢弃
            break                        # 第一个不匹配即终止本轮 (教学版)

    return accepted
