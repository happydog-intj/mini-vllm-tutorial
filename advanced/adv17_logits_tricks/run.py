"""
adv17_logits_tricks/run.py

Logits 层面的 5 个实用 Trick 演示
==================================
所有 trick 都工作在 softmax 之前的 logits 向量上，
通过加偏置、设 -inf、读取 logprob 等方式控制模型输出。

运行: python run.py
"""

import torch
import torch.nn.functional as F


# ===========================================================================
# Trick 1: Logit Bias（偏置注入）
# ===========================================================================

def logit_bias(logits: torch.Tensor, bias_map: dict[int, float]) -> torch.Tensor:
    """
    给指定 token 的 logit 加一个常数偏置。

    原理:
        logits[token_id] += bias_value
        正偏置 → 提升该 token 被选中的概率
        负偏置 → 降低该 token 被选中的概率
        极大负偏置（如 -100）≈ 禁止该 token

    用途:
        - OpenAI API 的 logit_bias 参数
        - 鼓励模型输出特定格式词（如 JSON 的 { ）
        - 抑制某些不想看到的 token

    Parameters
    ----------
    logits : shape [vocab_size]
    bias_map : {token_id: bias_value}
    """
    result = logits.clone()
    for tid, bias in bias_map.items():
        result[tid] += bias
    return result


# ===========================================================================
# Trick 2: Forced Token / Constrained Classification
# ===========================================================================

def force_tokens(logits: torch.Tensor, allowed_ids: list[int]) -> torch.Tensor:
    """
    只保留指定 token，其余全部设为 -inf。

    原理:
        对于不在 allowed_ids 中的 token: logits[id] = -inf
        softmax 后这些 token 概率为 0，永远不会被选中
        模型被迫从 allowed_ids 中选择

    用途:
        - 强制模型只输出 "yes" 或 "no"（二元分类）
        - 强制模型只输出 "A", "B", "C", "D"（多选题）
        - 情感分类：只允许 "positive" / "negative" / "neutral"

    Parameters
    ----------
    logits : shape [vocab_size]
    allowed_ids : 允许的 token id 列表
    """
    mask = torch.full_like(logits, float("-inf"))
    mask[allowed_ids] = 0.0
    return logits + mask


# ===========================================================================
# Trick 3: Logprobs 提取（置信度打分）
# ===========================================================================

def extract_logprobs(
    logits: torch.Tensor, target_ids: list[int]
) -> dict[int, float]:
    """
    提取指定 token 的 log probability，用于置信度打分。

    原理:
        log_probs = log_softmax(logits)
        对目标 token 读取其 log_prob 值
        值越高 → 模型对该选项越有信心

    用途:
        - 不生成文本，直接比较 yes/no 谁的 logprob 高
        - 多分类：取 logprob 最高的选项作为答案
        - 校准：用 logprob 做模型置信度阈值

    Parameters
    ----------
    logits : shape [vocab_size]
    target_ids : 要提取 logprob 的 token id 列表

    Returns
    -------
    {token_id: log_probability}
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return {tid: log_probs[tid].item() for tid in target_ids}


# ===========================================================================
# Trick 4: Prefix Forcing（前缀强制）
# ===========================================================================

def prefix_force(
    step: int, prefix_ids: list[int], logits: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    """
    前 N 步强制输出指定 token，不走 logits 选择。

    原理:
        if step < len(prefix_ids):
            直接返回只允许 prefix_ids[step] 的 logits（相当于强制选中）
        else:
            正常返回原始 logits

    用途:
        - JSON 输出时强制以 '{"' 开头
        - 避免模型先输出 "Sure, here's the JSON:"
        - 强制回答以特定格式前缀开始（如 "Answer: "）

    Parameters
    ----------
    step : 当前解码步数（从 0 开始）
    prefix_ids : 要强制输出的前缀 token id 序列
    logits : shape [vocab_size]

    Returns
    -------
    (masked_logits, is_forced): masked logits 和是否是强制步
    """
    if step < len(prefix_ids):
        forced = torch.full_like(logits, float("-inf"))
        forced[prefix_ids[step]] = 0.0
        return forced, True
    return logits, False


# ===========================================================================
# Trick 5: Ban Tokens（禁止词）
# ===========================================================================

def ban_tokens(logits: torch.Tensor, banned_ids: list[int]) -> torch.Tensor:
    """
    永久屏蔽指定 token（设为 -inf）。

    原理:
        logits[banned_id] = -inf
        与 force_tokens 相反：force 是白名单，ban 是黑名单

    用途:
        - 禁止脏话/敏感词对应的 token
        - 防止模型输出 system prompt 中的关键词
        - 禁止 EOS token（强制模型继续生成到指定长度）
        - 禁止换行符（强制单行输出）

    Parameters
    ----------
    logits : shape [vocab_size]
    banned_ids : 要禁止的 token id 列表
    """
    result = logits.clone()
    result[banned_ids] = float("-inf")
    return result


# ===========================================================================
# Demo & 验证
# ===========================================================================

def demo():
    print("=" * 60)
    print("Logits Tricks 工具箱 — 5 个实用技巧演示")
    print("=" * 60)

    # 模拟词表
    vocab = {0: "yes", 1: "no", 2: "maybe", 3: "hello", 4: "{", 5: '"'}
    VOCAB_SIZE = len(vocab)

    # 模拟模型 logits
    torch.manual_seed(42)
    raw_logits = torch.randn(VOCAB_SIZE)
    print(f"\n原始 logits: {dict(zip(vocab.values(), raw_logits.tolist()))}")
    raw_probs = F.softmax(raw_logits, dim=-1)
    print(f"原始 probs:  {dict(zip(vocab.values(), [f'{p:.3f}' for p in raw_probs.tolist()]))}")

    # ----- Trick 1: Logit Bias -----
    print("\n" + "-" * 60)
    print("Trick 1: Logit Bias — 给 'yes' 加 +5 偏置")
    biased = logit_bias(raw_logits, {0: 5.0})  # yes=0
    probs = F.softmax(biased, dim=-1)
    print(f"  偏置后 probs: {dict(zip(vocab.values(), [f'{p:.3f}' for p in probs.tolist()]))}")
    print(f"  → 'yes' 概率从 {raw_probs[0]:.3f} 提升到 {probs[0]:.3f}")
    assert probs[0] > raw_probs[0], "偏置应该提升 yes 的概率"

    # ----- Trick 2: Force Tokens -----
    print("\n" + "-" * 60)
    print("Trick 2: Force Tokens — 只允许 'yes'(0) 和 'no'(1)")
    forced = force_tokens(raw_logits, [0, 1])
    probs = F.softmax(forced, dim=-1)
    print(f"  强制后 probs: {dict(zip(vocab.values(), [f'{p:.3f}' for p in probs.tolist()]))}")
    assert probs[2:].sum() < 1e-6, "非 yes/no token 概率应为 0"
    assert abs(probs.sum() - 1.0) < 1e-5, "概率和应为 1"
    winner = "yes" if probs[0] > probs[1] else "no"
    print(f"  → 模型选择: '{winner}' (只能在 yes/no 中选)")

    # ----- Trick 3: Logprobs 提取 -----
    print("\n" + "-" * 60)
    print("Trick 3: Logprobs 提取 — 比较 'yes' vs 'no' 的置信度")
    lp = extract_logprobs(raw_logits, [0, 1])
    print(f"  logprob('yes') = {lp[0]:.4f}")
    print(f"  logprob('no')  = {lp[1]:.4f}")
    confident = "yes" if lp[0] > lp[1] else "no"
    confidence = abs(lp[0] - lp[1])
    print(f"  → 模型更倾向 '{confident}'，差距 = {confidence:.4f}")

    # ----- Trick 4: Prefix Forcing -----
    print("\n" + "-" * 60)
    print("Trick 4: Prefix Forcing — 强制前两步输出 '{' + '\"'")
    prefix = [4, 5]  # { 和 "
    for step in range(4):
        masked, is_forced = prefix_force(step, prefix, raw_logits)
        chosen = int(torch.argmax(masked).item())
        tag = "[FORCED]" if is_forced else "[FREE]  "
        print(f"  step {step}: {tag} → '{vocab[chosen]}'")
    print("  → 前 2 步强制输出 '{\"'，之后模型自由选择")

    # ----- Trick 5: Ban Tokens -----
    print("\n" + "-" * 60)
    print("Trick 5: Ban Tokens — 禁止 'maybe'(2) 和 'hello'(3)")
    banned = ban_tokens(raw_logits, [2, 3])
    probs = F.softmax(banned, dim=-1)
    print(f"  禁止后 probs: {dict(zip(vocab.values(), [f'{p:.3f}' for p in probs.tolist()]))}")
    assert probs[2] < 1e-6 and probs[3] < 1e-6, "被禁止的 token 概率应为 0"
    print(f"  → 'maybe' 和 'hello' 概率归零")

    # ----- 组合使用 -----
    print("\n" + "-" * 60)
    print("组合: Ban + Bias — 禁止 maybe/hello，同时偏置 yes")
    combined = ban_tokens(raw_logits, [2, 3])
    combined = logit_bias(combined, {0: 3.0})
    probs = F.softmax(combined, dim=-1)
    print(f"  组合后 probs: {dict(zip(vocab.values(), [f'{p:.3f}' for p in probs.tolist()]))}")

    print("\n" + "=" * 60)
    print("✅ 所有 Logits Tricks 演示通过")


if __name__ == "__main__":
    demo()
