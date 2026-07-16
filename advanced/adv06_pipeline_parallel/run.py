"""
run.py — adv06 Pipeline Parallel 调度对比演示

场景: 4 个 GPU stage × 4 microbatch
对比指标:
  - 串行模拟总时间（体现调度顺序差异）
  - 理论峰值驻留 microbatch 数（体现显存气泡差异）

⚠️  显存优势说明:
    本脚本运行在单机串行环境，总时间不能反映真实多 GPU 吞吐差异。
    "峰值驻留数"使用理论推导（compute_theoretical_peak），
    而非实测——串行模拟无法复现多 GPU 并行时的激活驻留并发情况。
    断言基于理论值: 1F1B 峰值 (p-1) < GPipe 峰值 (n)。
"""

from pp_sim import (
    Device,
    gpipe_schedule,
    onef_oneb_schedule,
    compute_theoretical_peak,
)

# ─── 配置 ─────────────────────────────────────────────────────────────────────
NUM_STAGES = 4
NUM_MICROBATCHES = 4

devices = [
    Device(name=f"GPU{i}", layers=[f"L{i*4}", f"L{i*4+1}", f"L{i*4+2}", f"L{i*4+3}"],
           comm_latency=0.01, fwd_time=0.02, bwd_time=0.04)
    for i in range(NUM_STAGES)
]

# ─── 运行两种调度 ──────────────────────────────────────────────────────────────
gpipe_time, gpipe_events  = gpipe_schedule(devices, NUM_MICROBATCHES)
onefb_time, onefb_events  = onef_oneb_schedule(devices, NUM_MICROBATCHES)

# ─── 理论峰值显存（驻留 microbatch 数） ─────────────────────────────────────────
gpipe_peak = compute_theoretical_peak('gpipe', NUM_MICROBATCHES, NUM_STAGES)
onefb_peak = compute_theoretical_peak('1f1b',  NUM_MICROBATCHES, NUM_STAGES)

# ─── 报告 ──────────────────────────────────────────────────────────────────────
print("=" * 62)
print(f"  Pipeline Parallel 调度对比 (stages={NUM_STAGES}, microbatches={NUM_MICROBATCHES})")
print("=" * 62)

print(f"\n【GPipe 调度】")
print(f"  串行模拟总时间          : {gpipe_time:.4f} s")
print(f"  理论峰值驻留 mb 数       : {gpipe_peak}  （= n，全部 microbatch 同驻）")

print(f"\n【1F1B 调度】")
print(f"  串行模拟总时间          : {onefb_time:.4f} s")
print(f"  理论峰值驻留 mb 数       : {onefb_peak}  （= p-1，仅 warmup depth 个 mb）")

print("\n" + "-" * 62)
print(f"  显存峰值对比  GPipe={gpipe_peak} mb  vs  1F1B={onefb_peak} mb")
savings_pct = (gpipe_peak - onefb_peak) / gpipe_peak * 100
print(f"  1F1B 节省峰值激活显存    : {savings_pct:.1f}%")
print("-" * 62)

print("\n  ⚠️  模拟说明:")
print("    - 串行仿真，总时间不代表真实多 GPU 加速效果。")
print("    - 真实场景下 1F1B 吞吐 ≈ GPipe，但显存峰值从 n 降至 p-1。")
print("    - 当 n >> p 时（如 n=32, p=4），1F1B 节省 ~90% 激活显存。")

# ─── 额外展示: 不同 n 下的峰值对比 ────────────────────────────────────────────
print("\n  不同 n 下 GPU0 理论峰值驻留数对比 (p=4):")
print(f"  {'n':>6} | {'GPipe':>8} | {'1F1B':>8} | {'节省':>8}")
print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
for n in [4, 8, 16, 32]:
    g = compute_theoretical_peak('gpipe', n, NUM_STAGES)
    b = compute_theoretical_peak('1f1b',  n, NUM_STAGES)
    s = (g - b) / g * 100
    print(f"  {n:>6} | {g:>8} | {b:>8} | {s:>7.1f}%")

# ─── 断言：验证 1F1B 显存优势 ─────────────────────────────────────────────────
assert onefb_peak < gpipe_peak, (
    f"1F1B 理论峰值 ({onefb_peak}) 应 < GPipe ({gpipe_peak})"
)

print("\n✅ adv06_pipeline_parallel 通过")
