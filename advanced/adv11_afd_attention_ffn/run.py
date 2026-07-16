"""
adv11_afd_attention_ffn/run.py

AFD (Attention-FFN Disaggregation) 均衡配比对比实验。

场景:
  - Attention 单设备耗时 0.02s,FFN 单设备耗时 0.05s (FFN 更重)
  - 朴素部署: A/F 各 1 台设备, FFN 成为瓶颈
  - AFD 均衡部署: balanced_config 算出 A:F = 2:5, 两端实际耗时接近

断言:
  均衡后两端实际耗时之差 < 10% (相对误差)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from afd_sim import AttentionDevice, FFNDevice, run_layer, balanced_config

# ---------- 实验参数 ----------
ATTN_TIME = 0.02   # Attention 单设备耗时 (s)
FFN_TIME  = 0.05   # FFN      单设备耗时 (s)
SEQ_LEN   = 128    # 序列长度 (占位)
TOLERANCE = 0.10   # 允许的相对差 (10%)

print("=" * 58)
print("  adv11: AFD Attention-FFN 分离 对比实验")
print("=" * 58)
print(f"  Attention 单设备耗时 : {ATTN_TIME * 1000:.0f} ms")
print(f"  FFN       单设备耗时 : {FFN_TIME  * 1000:.0f} ms")
print("-" * 58)

# ---------- 朴素部署: A/F 各 1 台 ----------
naive_attn = AttentionDevice(n=1, t=ATTN_TIME)
naive_ffn  = FFNDevice(n=1, t=FFN_TIME)

t0 = time.perf_counter()
run_layer(SEQ_LEN, naive_attn, naive_ffn)
naive_total = time.perf_counter() - t0

naive_attn_actual = ATTN_TIME / naive_attn.n   # = 0.020 s
naive_ffn_actual  = FFN_TIME  / naive_ffn.n    # = 0.050 s
naive_imbalance   = abs(naive_attn_actual - naive_ffn_actual) / max(naive_attn_actual, naive_ffn_actual)

print(f"  [朴素部署] A 设备数: {naive_attn.n},  F 设备数: {naive_ffn.n}")
print(f"    Attention 实际耗时 : {naive_attn_actual * 1000:.1f} ms")
print(f"    FFN       实际耗时 : {naive_ffn_actual  * 1000:.1f} ms")
print(f"    不均衡度           : {naive_imbalance * 100:.1f}%  ← FFN 是瓶颈")
print(f"    一层总耗时(实测)   : {naive_total * 1000:.1f} ms")
print("-" * 58)

# ---------- AFD 均衡部署 ----------
a_units, f_units = balanced_config(ATTN_TIME, FFN_TIME)

afd_attn = AttentionDevice(n=a_units, t=ATTN_TIME)
afd_ffn  = FFNDevice(n=f_units, t=FFN_TIME)

t0 = time.perf_counter()
run_layer(SEQ_LEN, afd_attn, afd_ffn)
afd_total = time.perf_counter() - t0

afd_attn_actual = ATTN_TIME / a_units
afd_ffn_actual  = FFN_TIME  / f_units
afd_imbalance   = abs(afd_attn_actual - afd_ffn_actual) / max(afd_attn_actual, afd_ffn_actual)

print(f"  [AFD 均衡] A 设备数: {a_units},  F 设备数: {f_units}")
print(f"    Attention 实际耗时 : {afd_attn_actual * 1000:.1f} ms")
print(f"    FFN       实际耗时 : {afd_ffn_actual  * 1000:.1f} ms")
print(f"    不均衡度           : {afd_imbalance * 100:.1f}%  ← 两端均衡")
print(f"    一层总耗时(实测)   : {afd_total * 1000:.1f} ms")
print("-" * 58)

# ---------- 利用率分析 ----------
bottleneck_naive = max(naive_attn_actual, naive_ffn_actual)
attn_util_naive  = naive_attn_actual / bottleneck_naive * 100
ffn_util_naive   = naive_ffn_actual  / bottleneck_naive * 100

bottleneck_afd  = max(afd_attn_actual, afd_ffn_actual)
attn_util_afd   = afd_attn_actual / bottleneck_afd * 100
ffn_util_afd    = afd_ffn_actual  / bottleneck_afd * 100

print(f"  利用率对比:")
print(f"    朴素  → Attention: {attn_util_naive:.0f}%  FFN: {ffn_util_naive:.0f}%")
print(f"    AFD   → Attention: {attn_util_afd:.0f}%  FFN: {ffn_util_afd:.0f}%")
print("=" * 58)

# ---------- 断言: 均衡后两端耗时差 < TOLERANCE ----------
assert afd_imbalance < TOLERANCE, (
    f"AFD 均衡失败!两端耗时差 {afd_imbalance * 100:.1f}% 超出阈值 {TOLERANCE * 100:.0f}%。\n"
    f"  Attention 实际耗时: {afd_attn_actual * 1000:.2f} ms\n"
    f"  FFN       实际耗时: {afd_ffn_actual  * 1000:.2f} ms\n"
    f"  配比 a_units={a_units}, f_units={f_units}"
)

print("\n✅ adv11_afd_attention_ffn 通过")
