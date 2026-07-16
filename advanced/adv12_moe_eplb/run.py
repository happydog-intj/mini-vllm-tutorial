"""
adv12 演示脚本: MoE 路由 + EPLB 专家负载均衡

验证内容:
  1. MoELayer 正常前向传播,输出形状正确
  2. expert_imbalance(load) > 1.0 —— 验证天然路由不均衡
  3. eplb_rebalance 后各设备负载方差 < 朴素均分时 —— 验证 EPLB 确实改善了均衡
"""

import torch
import statistics

from moe_layer import MoELayer, expert_imbalance, eplb_rebalance

# ─────────────────────────────────────────
# 超参数
# ─────────────────────────────────────────
torch.manual_seed(42)          # 固定随机数,结果可复现
SEQ_LEN    = 64
D_MODEL    = 8
NUM_EXPERTS = 4
TOP_K       = 2
NUM_DEVICES = 2

# ─────────────────────────────────────────
# 构造模型 & 输入
# ─────────────────────────────────────────
model = MoELayer(d_model=D_MODEL, num_experts=NUM_EXPERTS, top_k=TOP_K)
x = torch.randn(SEQ_LEN, D_MODEL)

# ─────────────────────────────────────────
# 前向传播
# ─────────────────────────────────────────
out, load = model(x)

# 断言输出形状正确
assert out.shape == (SEQ_LEN, D_MODEL), \
    f"输出形状错误: 期望 ({SEQ_LEN}, {D_MODEL}),实际 {tuple(out.shape)}"

# ─────────────────────────────────────────
# 打印路由分布
# ─────────────────────────────────────────
print("=" * 50)
print("adv12: MoE + EPLB 专家负载均衡")
print("=" * 50)
print(f"\n[路由分布] 每个专家被路由到的 token 数 (共 {SEQ_LEN*TOP_K} 次路由):")
for e in range(NUM_EXPERTS):
    bar = "█" * load[e].item()
    print(f"  专家 {e}: {load[e].item():3d}  {bar}")

# ─────────────────────────────────────────
# 断言 1: 不均衡度 > 1
# ─────────────────────────────────────────
imbalance = expert_imbalance(load)
print(f"\n[不均衡度] max/min 负载比 = {imbalance:.3f}")
assert imbalance > 1.0, (
    f"期望负载不均衡(imbalance > 1.0),实际 imbalance = {imbalance:.3f}。"
    "请换一个随机种子或增大 seq_len。"
)
print("  ✓ 天然路由确实不均衡 (imbalance > 1.0)")

# ─────────────────────────────────────────
# EPLB 重均衡
# ─────────────────────────────────────────
assignment, device_load_after = eplb_rebalance(load, num_devices=NUM_DEVICES)

print(f"\n[EPLB 重均衡] 专家 -> 设备映射:")
for e in range(NUM_EXPERTS):
    print(f"  专家 {e}(负载 {load[e].item():3d}) -> 设备 {assignment[e]}")

# 朴素均分:前 NUM_EXPERTS//2 个专家分给设备 0,后半分给设备 1
half = NUM_EXPERTS // NUM_DEVICES
naive_device_load = []
for d in range(NUM_DEVICES):
    naive_device_load.append(
        sum(load[d * half + i].item() for i in range(half))
    )

print(f"\n[各设备负载对比]")
print(f"  {'设备':<6}{'朴素均分':>10}{'EPLB':>10}")
for d in range(NUM_DEVICES):
    print(f"  设备 {d}  {naive_device_load[d]:>10.1f}{device_load_after[d]:>10.1f}")

max_load_naive = max(naive_device_load)
max_load_eplb  = max(device_load_after)

var_naive = statistics.variance(naive_device_load) if len(naive_device_load) > 1 else 0.0
var_eplb  = statistics.variance(device_load_after) if len(device_load_after) > 1 else 0.0

print(f"\n  朴素均分 -> 最大设备负载: {max_load_naive:.1f}, 方差: {var_naive:.2f}")
print(f"  EPLB     -> 最大设备负载: {max_load_eplb:.1f}, 方差: {var_eplb:.2f}")

# ─────────────────────────────────────────
# 断言 2: EPLB 后更均衡(最大设备负载 ≤ 朴素方案,或方差更小)
# ─────────────────────────────────────────
assert max_load_eplb <= max_load_naive or var_eplb <= var_naive, (
    f"EPLB 未能改善均衡:\n"
    f"  朴素最大负载={max_load_naive:.1f}, 方差={var_naive:.2f}\n"
    f"  EPLB 最大负载={max_load_eplb:.1f}, 方差={var_eplb:.2f}"
)
print("  ✓ EPLB 后各设备负载更均衡")

print("\n✅ adv12_moe_eplb 通过")
