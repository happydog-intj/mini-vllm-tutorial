"""
adv07: Sequence Parallel 验证脚本

验证内容：
  1. sp_attention 与标准 attention 数值 allclose（改进版 SP：Q 切分，KV 完整）
  2. all_gather 形状：shard [seq/2, d] → full [seq, d]
  3. reduce_scatter 形状：full [seq, d] → shard [seq/2, d]
  4. reduce_scatter 数值语义：规约后取第 0 卡分片，值 = full * world_size 的前半段
"""

import torch
from sp_sim import sp_attention, standard_attention, all_gather, reduce_scatter

torch.manual_seed(42)

# ──────────────────────────────────────────────
# 超参
# ──────────────────────────────────────────────
SEQ = 16       # 序列长度（可被 seq_splits 和 world_size 整除）
D = 32         # 注意力头维度
SEQ_SPLITS = 4 # 模拟卡数（Q 的切分份数）
WORLD_SIZE = 2 # all_gather / reduce_scatter 模拟卡数

# ──────────────────────────────────────────────
# 1. sp_attention vs 标准 attention
# ──────────────────────────────────────────────
print("=" * 56)
print("验证 1：sp_attention 与标准 attention 数值一致性")
print("=" * 56)

q = torch.randn(SEQ, D)
k = torch.randn(SEQ, D)
v = torch.randn(SEQ, D)

out_sp  = sp_attention(q, k, v, seq_splits=SEQ_SPLITS)
out_std = standard_attention(q, k, v)

print(f"  输入 q/k/v shape : {list(q.shape)}")
print(f"  sp_attention  out: {list(out_sp.shape)}")
print(f"  standard_attn out: {list(out_std.shape)}")
max_diff = (out_sp - out_std).abs().max().item()
print(f"  最大绝对误差      : {max_diff:.2e}")

assert out_sp.shape == out_std.shape, "形状不一致！"
assert torch.allclose(out_sp, out_std, atol=1e-5), (
    f"数值不一致，最大误差 {max_diff:.2e}"
)
print("  ✅ sp_attention 与标准 attention 数值 allclose\n")

# ──────────────────────────────────────────────
# 2. all_gather 形状验证
# ──────────────────────────────────────────────
print("=" * 56)
print("验证 2：all_gather 形状（沿 dim=0 拼接）")
print("=" * 56)

local_shard = torch.randn(SEQ // WORLD_SIZE, D)   # 每卡持有 seq/2 行
full = all_gather(local_shard, world_size=WORLD_SIZE, dim=0)

print(f"  local_shard shape : {list(local_shard.shape)}")
print(f"  all_gather   out  : {list(full.shape)}")
expected_full_shape = [SEQ, D]
assert list(full.shape) == expected_full_shape, (
    f"all_gather 形状错误：期望 {expected_full_shape}，得到 {list(full.shape)}"
)
# 验证内容：前半段 == 后半段（两份相同的 shard 拼接）
assert torch.equal(full[:SEQ // WORLD_SIZE], full[SEQ // WORLD_SIZE:]), \
    "all_gather 内容错误：前后两段应相等"
print("  ✅ all_gather 形状正确，前后分片内容一致\n")

# ──────────────────────────────────────────────
# 3. reduce_scatter 形状验证
# ──────────────────────────────────────────────
print("=" * 56)
print("验证 3：reduce_scatter 形状（沿 dim=0 切分）")
print("=" * 56)

full_tensor = torch.randn(SEQ, D)  # 完整张量（每卡都持有一份）
shard = reduce_scatter(full_tensor, world_size=WORLD_SIZE, dim=0)

print(f"  full_tensor shape : {list(full_tensor.shape)}")
print(f"  reduce_scatter out: {list(shard.shape)}")
expected_shard_shape = [SEQ // WORLD_SIZE, D]
assert list(shard.shape) == expected_shard_shape, (
    f"reduce_scatter 形状错误：期望 {expected_shard_shape}，得到 {list(shard.shape)}"
)
print("  ✅ reduce_scatter 形状正确\n")

# ──────────────────────────────────────────────
# 4. reduce_scatter 数值语义验证
# ──────────────────────────────────────────────
print("=" * 56)
print("验证 4：reduce_scatter 数值语义")
print("=" * 56)

# rank 0 得到的分片应等于 (full * world_size) 的前 seq/2 行
expected_shard_values = (full_tensor * WORLD_SIZE).narrow(0, 0, SEQ // WORLD_SIZE)
assert torch.allclose(shard, expected_shard_values), \
    "reduce_scatter 数值语义错误"
print(f"  rank-0 shard == full * {WORLD_SIZE} 的前 {SEQ // WORLD_SIZE} 行")
print("  ✅ reduce_scatter 数值语义正确\n")

# ──────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────
print("=" * 56)
print("\n✅ adv07_sequence_parallel 通过")
