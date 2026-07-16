"""
adv13 run.py — Linear Attention / SSM 验证脚本

断言:
  ① linear_attention 与 standard_attention 输出形状一致
  ② linear_attention_matrix(矩阵形式) 与 linear_attention_noncausal(递推形式) 结果 allclose
  ③ LinearAttentionLayer 前向通过

打印复杂度对比说明。
"""

import torch
from linear_attn import (
    standard_attention,
    linear_attention,
    linear_attention_matrix,
    linear_attention_noncausal,
    LinearAttentionLayer,
)

torch.manual_seed(42)

SEQ = 16
D = 32

q = torch.randn(SEQ, D)
k = torch.randn(SEQ, D)
v = torch.randn(SEQ, D)

# ─────────────────────────────────────────────
# 断言 ①: 输出形状一致
# ─────────────────────────────────────────────
out_std = standard_attention(q, k, v)
out_lin = linear_attention(q, k, v)

assert out_std.shape == out_lin.shape, (
    f"形状不一致: standard={out_std.shape}, linear={out_lin.shape}"
)
print(f"[断言①] 形状检查通过: standard={out_std.shape}, linear={out_lin.shape}")
print(f"        注意:数值不同——linear_attention 用 φ(x)=elu(x)+1 近似 softmax,")
print(f"        是无归一化的线性近似,数值差异属预期行为。")

diff = (out_std - out_lin).abs().max().item()
print(f"        最大绝对误差 (仅供参考): {diff:.4f}")

# ─────────────────────────────────────────────
# 断言 ②: 矩阵形式 vs 递推形式 allclose
#   两者均为非因果(使用全部序列),数学上严格等价:
#   递推: S = Σ_t outer(φ(k_t), v_t),  out_t = φ(q_t) @ S
#   矩阵: S = φ(K)ᵀ @ V,               out   = φ(Q) @ S
# ─────────────────────────────────────────────
out_matrix = linear_attention_matrix(q, k, v)
out_recur  = linear_attention_noncausal(q, k, v)

max_err = (out_matrix - out_recur).abs().max().item()
print(f"\n[断言②] 矩阵形式 vs 非因果递推形式 最大绝对误差: {max_err:.2e}")
assert torch.allclose(out_matrix, out_recur, atol=1e-5), (
    f"矩阵形式与递推形式不 allclose! 最大误差={max_err:.2e}"
)
print(f"        allclose 通过 (atol=1e-5) ✓")
print(f"        数学等价性: φ(Q)(φ(K)ᵀV) = Σ_t [φ(q_t) @ Σ_i outer(φ(k_i),v_i)]")

# ─────────────────────────────────────────────
# 断言 ③: LinearAttentionLayer 前向
# ─────────────────────────────────────────────
x = torch.randn(SEQ, D)
layer = LinearAttentionLayer(D)
out_layer = layer(x)
assert out_layer.shape == (SEQ, D), f"LinearAttentionLayer 输出形状错误: {out_layer.shape}"
print(f"\n[断言③] LinearAttentionLayer 前向通过: 输出形状={out_layer.shape}")

# ─────────────────────────────────────────────
# 复杂度对比说明
# ─────────────────────────────────────────────
print(f"""
┌─────────────────────────────────────────────────────────────────┐
│               复杂度对比 (seq={SEQ}, d={D})
├──────────────────────┬──────────────────┬────────────────────────┤
│ 方法                 │ 时间复杂度        │ 关键显存               │
├──────────────────────┼──────────────────┼────────────────────────┤
│ standard_attention   │ O(seq² × d)      │ O(seq²) 注意力矩阵     │
│ linear_attention     │ O(seq × d²)      │ O(d²)  状态矩阵 S      │
├──────────────────────┴──────────────────┴────────────────────────┤
│ 当 seq >> d 时,线性 attention 大幅节省显存和计算               │
│ 例: seq=4096, d=64 → 注意力矩阵 16M vs 状态矩阵 4K (4000x)    │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n✅ adv13_linear_attention 通过")
