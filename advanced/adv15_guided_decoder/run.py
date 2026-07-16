"""
adv15_guided_decoder/run.py

演示：用 RegexGuide 约束生成数字（regex: r'-?\\d+(\\.\\d+)?'）

词表设计
--------
字符级小词表，覆盖所有可能出现在数字中的字符：
  '-'  '0'~'9'  '.'  以及一个终止符 '<EOS>'

模拟流程
--------
1. 构造 RegexGuide(pattern, vocab)
2. 每步生成随机 logits（模拟模型输出）
3. 经 next_allowed() 掩码后 argmax
4. consume() 更新状态
5. is_complete() 判断是否已生成合法完整数字
6. 最终 assert fullmatch 通过
"""

import re

import torch

from guided import RegexGuide, PARTIAL_MATCH_STRATEGY

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
PATTERN = r"-?\d+(\.\d+)?"   # 匹配：整数或小数，允许负号
MAX_STEPS = 20               # 最多生成多少个字符防止死循环

# ---------------------------------------------------------------------------
# 字符级词表：每个 token = 单字符
# ---------------------------------------------------------------------------
VOCAB_CHARS = list("-0123456789.")
EOS_CHAR = "<EOS>"           # 特殊终止 token（不纳入正则匹配但会在词表中）

vocab: dict[int, str] = {}
for i, ch in enumerate(VOCAB_CHARS):
    vocab[i] = ch
EOS_ID = len(VOCAB_CHARS)
vocab[EOS_ID] = EOS_CHAR     # EOS 不会 partial match -> 被 mask 掉

VOCAB_SIZE = len(vocab)

# ---------------------------------------------------------------------------
# 固定随机种子，使结果可复现
# ---------------------------------------------------------------------------
torch.manual_seed(42)


def simulate_guided_decode(target_hint: str = "3.14") -> str:
    """
    模拟一次带 regex 约束的贪心解码。

    target_hint 用于构造带偏置的 logits（让演示生成有意义的数字）；
    约束本身完全依赖 RegexGuide，与 hint 无关。
    """
    guide = RegexGuide(PATTERN, vocab)

    # 为演示目的，给 target_hint 中的字符施加正向偏置
    char_boost: dict[str, float] = {}
    for ch in target_hint:
        char_boost[ch] = char_boost.get(ch, 0.0) + 2.0

    generated_ids = []
    for step in range(MAX_STEPS):
        # 1. 模拟模型输出 logits（随机基础 + 角色偏置）
        base_logits = torch.randn(VOCAB_SIZE)
        for tid, tok in vocab.items():
            if tok in char_boost:
                base_logits[tid] += char_boost[tok]

        # 2. 用 RegexGuide 掩码不合法 token
        masked = guide.next_allowed(base_logits)

        # 3. 检查是否有任何合法 token（安全断言）
        assert not torch.all(masked == float("-inf")), (
            f"Step {step}: 无合法 token 可选（已生成: {guide.generated!r}），"
            "regex 或词表配置有误"
        )

        # 4. Greedy: argmax
        chosen_id = int(torch.argmax(masked).item())
        chosen_tok = vocab[chosen_id]
        generated_ids.append(chosen_id)

        # 5. EOS or is_complete -> 停止（先 consume 再判断）
        guide.consume(chosen_tok)

        print(f"  step {step+1:2d}: token={chosen_tok!r:4s}  generated={guide.generated!r}")

        if guide.is_complete():
            break

    return guide.generated


# ---------------------------------------------------------------------------
# 主测试：断言生成结果完全匹配 regex
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("adv15_guided_decoder — Regex Guided Decoding Demo")
    print("=" * 55)
    print(f"Pattern        : {PATTERN}")
    print(f"Vocab size     : {VOCAB_SIZE} (chars: {VOCAB_CHARS!r} + EOS)")
    print(f"Partial match  : {PARTIAL_MATCH_STRATEGY}")
    print()

    # --- Case 1: 正小数 ---
    print("[Case 1] target hint='3.14'")
    result1 = simulate_guided_decode("3.14")
    print(f"  => generated: {result1!r}")
    assert re.fullmatch(PATTERN, result1), (
        f"Case 1 FAILED: {result1!r} 不匹配 {PATTERN}"
    )
    print(f"  => fullmatch OK\n")

    # --- Case 2: 负整数 ---
    print("[Case 2] target hint='-42'")
    result2 = simulate_guided_decode("-42")
    print(f"  => generated: {result2!r}")
    assert re.fullmatch(PATTERN, result2), (
        f"Case 2 FAILED: {result2!r} 不匹配 {PATTERN}"
    )
    print(f"  => fullmatch OK\n")

    # --- Case 3: 纯整数 ---
    print("[Case 3] target hint='7'")
    result3 = simulate_guided_decode("7")
    print(f"  => generated: {result3!r}")
    assert re.fullmatch(PATTERN, result3), (
        f"Case 3 FAILED: {result3!r} 不匹配 {PATTERN}"
    )
    print(f"  => fullmatch OK\n")

    print("=" * 55)
    print(f"生成结果汇总: {result1!r}  {result2!r}  {result3!r}")
    print()
    print("✅ adv15_guided_decoder 通过")
