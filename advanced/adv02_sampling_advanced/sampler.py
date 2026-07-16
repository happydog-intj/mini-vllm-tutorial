"""
adv02: 采样进阶 — MinP / 惩罚项 / Beam Search

教学要点:
  - MinP: 保留概率 >= max_prob * min_p 的候选，比 TopK/TopP 更自适应
  - 频率惩罚 (Frequency Penalty): 出现次数越多，logit 衰减越多
  - 存在惩罚 (Presence Penalty): 出现过就减固定值，最多惩罚一次
  - 重复惩罚 (Repetition Penalty): 出现过的 token logit 除以 penalty(>1 则降低)
  - Beam Search: 每步保留 top-beam 个候选序列，返回累积 logprob 最高的序列
"""

import torch
from torch import Tensor


def min_p_sample(logits: Tensor, min_p: float, temperature: float = 1.0) -> Tensor:
    """
    MinP 采样：保留概率 >= max_prob * min_p 的候选，再从中采样。

    与 Top-p 不同，MinP 用绝对概率阈值（相对于最大概率）过滤，
    分布尖锐时过滤更激进，分布平坦时自动保留更多候选。

    logits:      [vocab_size]
    min_p:       最小概率比例，典型值 0.05~0.2
    temperature: 温度，应用于 softmax 之前
    返回:        标量 tensor（token_id）
    """
    assert 0.0 < min_p < 1.0, "min_p 必须在 (0, 1) 之间"
    assert temperature > 0.0, "temperature 必须 > 0"

    probs = torch.softmax(logits / temperature, dim=-1)
    max_p = probs.max()
    threshold = max_p * min_p

    # 保留概率 >= threshold 的 token，其余归零
    mask = probs >= threshold
    filtered = torch.where(mask, probs, torch.zeros_like(probs))

    # multinomial 按相对权重采样，等价于对过滤后的分布重归一化再采样
    return torch.multinomial(filtered, num_samples=1).squeeze(-1)


def apply_frequency_penalty(logits: Tensor, token_ids: Tensor, penalty: float) -> Tensor:
    """
    频率惩罚：出现次数越多，logit 衰减越多。

    logit[i] -= penalty * count(i)

    适合压制高频重复词；penalty=0 等价于无惩罚。
    """
    freq = torch.bincount(token_ids, minlength=logits.size(-1)).float()
    return logits - penalty * freq


def apply_presence_penalty(logits: Tensor, token_ids: Tensor, penalty: float) -> Tensor:
    """
    存在惩罚：出现过就减固定值，无论出现多少次。

    logit[i] -= penalty  (若 i 在 token_ids 中出现过)

    适合鼓励主题多样性；与频率惩罚的区别：不管出现 1 次还是 100 次，惩罚相同。
    """
    appeared = torch.bincount(token_ids, minlength=logits.size(-1)).clamp(0, 1).float()
    return logits - penalty * appeared


def apply_repetition_penalty(logits: Tensor, token_ids: Tensor, penalty: float) -> Tensor:
    """
    重复惩罚：出现过的 token，logit 除以 penalty(>1 则降低正值 logit)。

    与频率/存在惩罚的区别：这是乘法惩罚，对正值 logit 更有力。
    typical penalty: 1.1 ~ 1.3

    注意：此教学实现仅对正 logit 效果符合直觉（正值/penalty < 正值）；
    生产实现（如 vLLM/transformers）会区分正负号：正值除以 penalty，负值乘以 penalty。
    """
    assert penalty >= 1.0, "penalty 应 >= 1.0（>1 才有惩罚效果）"
    appeared = torch.bincount(token_ids, minlength=logits.size(-1)).clamp(0, 1).bool()
    logits = logits.clone()
    logits[appeared] = logits[appeared] / penalty
    return logits


def beam_search(model, prompt_ids: Tensor, beam_width: int = 3, max_new: int = 10) -> Tensor:
    """
    束搜索（Beam Search）：每步保留 top-beam 个候选序列，返回最优序列。

    与贪心搜索相比：greedy 只保留当前最优，beam 保留 beam_width 个，
    最终从所有候选中选累积 logprob 最高的。

    model:      实现 (token_ids, past_key_values=None) -> (logits, _) 的模型
                logits 形状: [seq_len, vocab_size]
    prompt_ids: [prompt_len]，起始 token 序列
    beam_width: 每步保留的候选数
    max_new:    最多生成的新 token 数
    返回:       [prompt_len + max_new] 的 token 序列（最优候选）
    """
    # 每个 beam 是 (token_ids, 累积 logprob) 的二元组
    beams = [(prompt_ids, 0.0)]

    for _ in range(max_new):
        all_cands = []
        for ids, score in beams:
            # 简化实现：每步全量重算（不使用 KV Cache）
            logits, _ = model(ids[-1:], past_key_values=None)
            logp = torch.log_softmax(logits[-1], dim=-1)   # [vocab_size]
            topk = torch.topk(logp, beam_width)
            for v, idx in zip(topk.values, topk.indices):
                new_ids = torch.cat([ids, idx.view(1)])
                all_cands.append((new_ids, score + v.item()))

        # 按累积 logprob 升序，取最后 beam_width 个（最高分）
        beams = sorted(all_cands, key=lambda c: c[1])[-beam_width:]

    # 返回累积 logprob 最高的序列
    return beams[-1][0]
