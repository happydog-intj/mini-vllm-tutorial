"""
adv02 run.py — 验证 MinP / 惩罚项 / Beam Search

断言:
  1. MinP 过滤后 multinomial 的采样结果 token 合法（在过滤 mask 内）
  2. 三种 penalty 对已出现 token 的 logit 均有降低效果
  3. Beam Search 输出长度 == prompt_len + max_new
"""

import torch
from sampler import (
    min_p_sample,
    apply_frequency_penalty,
    apply_presence_penalty,
    apply_repetition_penalty,
    beam_search,
)

torch.manual_seed(42)

# ─── 1. MinP 采样 ────────────────────────────────────────────────────────────
print("=" * 50)
print("1. MinP 采样")

vocab_size = 20
logits = torch.randn(vocab_size)

# 计算期望的 mask（temperature=1.0）
probs = torch.softmax(logits, dim=-1)
max_p = probs.max()
min_p = 0.1
mask = probs >= max_p * min_p

print(f"   vocab_size={vocab_size}, min_p={min_p}")
print(f"   过滤后候选 token 数: {mask.sum().item()}")
print(f"   最大概率: {max_p.item():.4f}, 阈值: {(max_p * min_p).item():.4f}")

# 多次采样，验证每次采样结果都在 mask 内
for trial in range(50):
    token = min_p_sample(logits, min_p=min_p, temperature=1.0)
    assert mask[token].item(), (
        f"Trial {trial}: token {token.item()} 不在 MinP 过滤后的候选集内！"
    )

# 验证极端情况：temperature 影响候选集
# 低温 → 分布更尖锐 → 候选集更小（不一定比高温小，但应合法）
token_lo = min_p_sample(logits, min_p=min_p, temperature=0.5)
token_hi = min_p_sample(logits, min_p=min_p, temperature=2.0)
print(f"   T=0.5 采样结果: {token_lo.item()}, T=2.0 采样结果: {token_hi.item()}")
print("   [PASS] MinP 采样 50 次均在候选集内")

# ─── 2. 三种惩罚项 ────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("2. 惩罚项验证")

vocab_size = 50
logits_base = torch.randn(vocab_size)

# 已出现的 token（重复出现以测试频率惩罚）
token_ids = torch.tensor([3, 7, 3, 15, 3, 7])  # token 3 出现 3 次，7 出现 2 次，15 出现 1 次
appeared_once = torch.tensor([3, 7, 15])         # 出现过的 token 集合

penalty = 1.5

# 2a. 频率惩罚
logits_freq = apply_frequency_penalty(logits_base, token_ids, penalty=penalty)
for tok, count in [(3, 3), (7, 2), (15, 1)]:
    expected_drop = penalty * count
    actual_drop = (logits_base[tok] - logits_freq[tok]).item()
    assert abs(actual_drop - expected_drop) < 1e-5, (
        f"频率惩罚 token {tok}: 期望下降 {expected_drop:.2f}, 实际下降 {actual_drop:.2f}"
    )
    print(f"   [频率惩罚] token {tok} (出现{count}次): logit {logits_base[tok].item():.4f} → {logits_freq[tok].item():.4f}  下降 {actual_drop:.4f}")

# 未出现的 token 不受影响
no_appear = 42
assert abs(logits_base[no_appear].item() - logits_freq[no_appear].item()) < 1e-6, \
    "频率惩罚: 未出现 token 不应被修改"
print(f"   [频率惩罚] token {no_appear}(未出现): 保持不变 ✓")

# 2b. 存在惩罚
logits_pres = apply_presence_penalty(logits_base, token_ids, penalty=penalty)
for tok in [3, 7, 15]:  # 出现次数不同，但惩罚相同
    expected_drop = penalty * 1  # 存在惩罚：最多惩罚一次
    actual_drop = (logits_base[tok] - logits_pres[tok]).item()
    assert abs(actual_drop - expected_drop) < 1e-5, (
        f"存在惩罚 token {tok}: 期望下降 {expected_drop:.2f}, 实际下降 {actual_drop:.2f}"
    )
    print(f"   [存在惩罚] token {tok}: logit {logits_base[tok].item():.4f} → {logits_pres[tok].item():.4f}  下降 {actual_drop:.4f}")

# 频率惩罚 vs 存在惩罚：多次出现的 token，频率惩罚更重
assert logits_freq[3].item() < logits_pres[3].item(), \
    "token 3 出现 3 次，频率惩罚应比存在惩罚更大（logit 更低）"
print("   [频率 vs 存在] token 3 出现 3 次: 频率惩罚更重 ✓")

# 2c. 重复惩罚
logits_rep = apply_repetition_penalty(logits_base, token_ids, penalty=1.5)
for tok in [3, 7, 15]:
    before = logits_base[tok].item()
    after = logits_rep[tok].item()
    # 对正 logit，除以 penalty 后应变小
    if before > 0:
        assert after < before, f"重复惩罚 token {tok} (正 logit {before:.4f}): 惩罚后应更小，实际 {after:.4f}"
        print(f"   [重复惩罚] token {tok}: {before:.4f} → {after:.4f}  ✓ (正 logit 降低)")
    else:
        # 负 logit 除以 >1 的 penalty，绝对值变小（更接近 0，即变大）
        assert after > before, f"重复惩罚 token {tok} (负 logit {before:.4f}): 惩罚后应接近 0，实际 {after:.4f}"
        print(f"   [重复惩罚] token {tok}: {before:.4f} → {after:.4f}  ✓ (负 logit 绝对值缩小)")

# 未出现的 token 不受影响
assert abs(logits_base[no_appear].item() - logits_rep[no_appear].item()) < 1e-6, \
    "重复惩罚: 未出现 token 不应被修改"
print(f"   [重复惩罚] token {no_appear}(未出现): 保持不变 ✓")

print("   [PASS] 三种惩罚项验证通过")

# ─── 3. Beam Search 长度验证 ────────────────────────────────────────────────
print()
print("=" * 50)
print("3. Beam Search 长度验证")

# 构造一个极简假模型（不用真实 Transformer，用固定 logits）
class FakeModel:
    """
    教学用假模型：forward 返回固定随机 logits，形状 [seq_len, vocab_size]。
    兼容 TinyTransformerWithKVCache 的签名：(token_ids, past_key_values=None)
    """
    def __init__(self, vocab_size: int = 30):
        self.vocab_size = vocab_size
        torch.manual_seed(0)
        self._logits = torch.randn(vocab_size)

    def __call__(self, token_ids, past_key_values=None):
        seq_len = token_ids.size(0)
        logits = self._logits.unsqueeze(0).expand(seq_len, -1)  # [seq_len, vocab_size]
        return logits, None

model = FakeModel(vocab_size=30)

prompt_len = 5
max_new = 10
beam_width = 3

prompt_ids = torch.arange(prompt_len)  # [0, 1, 2, 3, 4]
result = beam_search(model, prompt_ids, beam_width=beam_width, max_new=max_new)

expected_len = prompt_len + max_new
actual_len = result.size(0)

print(f"   prompt_len={prompt_len}, max_new={max_new}, beam_width={beam_width}")
print(f"   期望输出长度: {expected_len}, 实际输出长度: {actual_len}")
assert actual_len == expected_len, (
    f"Beam Search 输出长度错误: 期望 {expected_len}, 实际 {actual_len}"
)

# 验证 prompt 部分未被修改
assert torch.equal(result[:prompt_len], prompt_ids), \
    "Beam Search 输出的 prompt 部分应与输入一致"
print(f"   输出序列前 {prompt_len} 个 token 与 prompt 一致 ✓")
print(f"   生成的新 token: {result[prompt_len:].tolist()}")
print("   [PASS] Beam Search 长度验证通过")

# ─── 最终总结 ─────────────────────────────────────────────────────────────────
print()
print("✅ adv02_sampling_advanced 通过")
