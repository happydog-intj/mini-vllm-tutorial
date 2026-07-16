"""
adv03 run.py — 投机解码演示

演示内容:
  1. 正确性: 投机解码(d_model=2 草稿)结果 == 纯自回归贪婪结果
  2. Forward 次数对比:
     a. 随机权重草稿 — 展示结构,接受率极低属正常现象
     b. Self-drafting  — 用 target 自身充当草稿,接受率=100%,展示理想加速
"""

import io
import sys
import contextlib
import torch
from model import TinyTransformerWithKVCache
from spec_decode import draft_speculate, target_verify


# ── 静音工具 ─────────────────────────────────────────────────────────────────
# step07 model.py 中包含调试 print,包装层统一屏蔽

@contextlib.contextmanager
def _suppress():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


class SilentModel:
    """屏蔽 step07 内部调试输出的包装器"""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __call__(self, *args, **kwargs):
        with _suppress():
            return self.model(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.model, name)


# ── 纯自回归生成 ─────────────────────────────────────────────────────────────

def ar_generate(model, prompt_ids: torch.Tensor, n_tokens: int):
    """用 KV Cache 自回归生成 n_tokens 个新 token。

    Returns:
        (tokens: list[int], n_fwd: int)  # n_fwd = target model forward 调用次数
    """
    tokens = []
    kv = None

    # Prefill: 处理完整 prompt,得到第一个新 token 的 logits
    logits, kv = model(prompt_ids, past_key_values=None)
    n_fwd = 1
    t = logits[-1].argmax().item()
    tokens.append(t)

    # Decode: 逐 token 生成
    for _ in range(n_tokens - 1):
        logits, kv = model(torch.tensor([t]), past_key_values=kv)
        n_fwd += 1
        t = logits[-1].argmax().item()
        tokens.append(t)

    return tokens, n_fwd


# ── 投机解码生成 ─────────────────────────────────────────────────────────────

def spec_generate(draft_model, target_model, prompt_ids: torch.Tensor, n_tokens: int, k: int = 4):
    """投机解码生成 n_tokens 个新 token。

    每轮:
      - draft_model 生成 k 个候选 token (共 k draft forwards: 1 prefill + k-1 decodes)
      - target_model 一次并行验证所有草稿 (1 target forward)
      - 接受的 token 追加进上下文

    Returns:
        (tokens: list[int], n_target_fwd: int)  # 只统计 target forward 次数
    """
    tokens = []
    context = prompt_ids.clone()
    n_target_fwd = 0

    while len(tokens) < n_tokens:
        remaining = n_tokens - len(tokens)
        ki = min(k, remaining)  # 最后一轮可能不足 k 个

        draft_tokens, draft_probs = draft_speculate(draft_model, context, k=ki)
        accepted = target_verify(target_model, context, draft_tokens, draft_probs)
        n_target_fwd += 1  # target_verify 做 1 次 target forward

        # 截断至剩余配额
        accepted = accepted[:remaining]
        tokens.extend(accepted)
        context = torch.cat([context, torch.tensor(accepted, dtype=torch.long)])

    return tokens[:n_tokens], n_target_fwd


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    # ── 构建模型 ──────────────────────────────────────────────────
    print("构建模型 ...")
    raw_target = TinyTransformerWithKVCache(vocab_size=256, d_model=4, num_heads=1, num_layers=1)
    raw_draft  = TinyTransformerWithKVCache(vocab_size=256, d_model=2, num_heads=1, num_layers=1)

    target = SilentModel(raw_target)   # 目标模型 (较大, 较精确)
    draft  = SilentModel(raw_draft)    # 草稿模型 (更小, 更快)

    prompt = torch.tensor([65, 66, 67], dtype=torch.long)  # "ABC" → token ids
    N = 16   # 新生成 token 数
    K = 4    # 每轮投机步数

    # ── 纯自回归生成 ──────────────────────────────────────────────
    print(f"自回归生成 {N} 个 token ...")
    ar_tokens, ar_fwd = ar_generate(target, prompt, N)

    # ── 投机解码 (d_model=2 草稿模型) ─────────────────────────────
    print(f"投机解码生成 {N} 个 token (k={K}) ...")
    spec_tokens, spec_target_fwd = spec_generate(draft, target, prompt, N, k=K)

    # ── 正确性断言 ─────────────────────────────────────────────────
    assert ar_tokens == spec_tokens, (
        f"[FAIL] 投机解码结果与纯自回归不一致!\n"
        f"  ar  : {ar_tokens}\n"
        f"  spec: {spec_tokens}"
    )

    # ── 结果打印 ────────────────────────────────────────────────────
    print()
    print("=" * 58)
    print("  adv03 投机解码 — Forward 次数对比 (d_model=2 草稿)")
    print("=" * 58)
    print(f"  生成 tokens 数           : {N}")
    print(f"  草稿步数 k               : {K}")
    print(f"  Target forward (AR)      : {ar_fwd}")
    print(f"  Target forward (Spec)    : {spec_target_fwd}")
    print()
    print(f"  生成序列 : {ar_tokens}")
    print(f"  ✓ 投机解码结果 == 纯自回归贪婪结果 (正确性断言通过)")
    print()

    if spec_target_fwd < ar_fwd:
        speedup = ar_fwd / spec_target_fwd
        print(f"  ✓ target forward 减少: {ar_fwd} → {spec_target_fwd} ({speedup:.1f}×)")
    else:
        print(f"  ○ 随机权重草稿接受率极低 ({spec_target_fwd} ≈ {ar_fwd})")
        print(f"    [符合预期] 生产场景草稿与目标来自同一模型家族,")
        print(f"    接受率通常 70-90%,target forward 可降至 {N//K}~{N//2} 次")

    # ── Self-drafting 演示:理想加速 ────────────────────────────────
    # 用 target 自身充当草稿 → argmax 完全对齐 → 接受率 100%
    # 等价于同模型家族中 draft ≡ target 的理想场景
    print()
    print("─" * 58)
    print("  Self-drafting 演示 (draft=target, 模拟 100% 接受率)")
    print("─" * 58)

    sd_tokens, sd_target_fwd = spec_generate(target, target, prompt, N, k=K)

    assert sd_tokens == ar_tokens, (
        "[FAIL] Self-drafting 结果与自回归不一致!\n"
        f"  ar: {ar_tokens}\n"
        f"  sd: {sd_tokens}"
    )

    speedup_ideal = ar_fwd / sd_target_fwd
    print(f"  Target forward (AR)          : {ar_fwd}")
    print(f"  Target forward (Self-draft)  : {sd_target_fwd}")
    print(f"  实测加速比                   : {speedup_ideal:.1f}×  (理论上限 {K}×)")
    assert sd_target_fwd < ar_fwd, "[FAIL] Self-drafting 应有更少 target forward"
    print(f"  ✓ target forward 减少: {ar_fwd} → {sd_target_fwd}")

    print()
    print("✅ adv03_speculative_decoding 通过")


if __name__ == "__main__":
    main()
