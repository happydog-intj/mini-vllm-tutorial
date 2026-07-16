"""
adv10_pd_disaggregation/run.py

对比 colocated（合并部署）vs disaggregated（分离部署）在长 prompt 场景的吞吐差异。

实验设计：
  - 场景1：基准配比（p_speed=d_speed=1.0），合并 vs 分离（无配比优势）
  - 场景2：分离部署对 P 引擎加速（p_speed=2.0），展示独立配比的吞吐提升
  - 场景3：长 prompt + 高 p_speed，展示分离部署在算力密集 Prefill 上的流水线优势

断言：分离部署（最优配比）的 wall_time 小于合并部署基准。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pd_sim import colocated, disaggregated

# ---------------------------------------------------------------------------
# 请求负载：长 prompt（512~2048 tokens），中等解码步数
# ---------------------------------------------------------------------------
REQS = [
    (2048, 50),   # 超长 prompt，Prefill 极重
    (1024, 80),
    (1536, 60),
    (2048, 40),
    (512,  120),
    (1024, 100),
    (1800, 50),
    (2048, 30),
]

KV_LATENCY = 0.001   # 1ms KV 迁移延迟（常数近似）

SEP = "=" * 60

print(SEP)
print("  adv10: PD Disaggregation — Prefill/Decode 分离对比实验")
print(SEP)
print(f"  请求数量  : {len(REQS)}")
print(f"  Prompt 长度分布 : {[r[0] for r in REQS]}")
print(f"  KV 迁移延迟     : {KV_LATENCY*1000:.1f} ms/请求")
print(SEP)

# ---------------------------------------------------------------------------
# 场景 A：合并部署基准（p_speed=d_speed=1.0）
# ---------------------------------------------------------------------------
wall_co, p_busy_co, d_busy_co = colocated(REQS, p_speed=1.0, d_speed=1.0)
total_co = p_busy_co + d_busy_co   # 合并部署：串行，total = wall_time
print("\n[A] 合并部署 (colocated, p_speed=1.0, d_speed=1.0)")
print(f"    wall_time  : {wall_co*1000:.2f} ms")
print(f"    P 忙碌时间 : {p_busy_co*1000:.2f} ms")
print(f"    D 忙碌时间 : {d_busy_co*1000:.2f} ms")

# ---------------------------------------------------------------------------
# 场景 B：分离部署（同等配比 1.0），无配比优势，但流水线重叠
# ---------------------------------------------------------------------------
wall_dis_eq, p_busy_eq, d_busy_eq = disaggregated(
    REQS, p_speed=1.0, d_speed=1.0, kv_latency=KV_LATENCY
)
print("\n[B] 分离部署 (disaggregated, p_speed=1.0, d_speed=1.0)")
print(f"    wall_time  : {wall_dis_eq*1000:.2f} ms  ← 流水线重叠收益")
print(f"    P 忙碌时间 : {p_busy_eq*1000:.2f} ms")
print(f"    D 忙碌时间 : {d_busy_eq*1000:.2f} ms")

# ---------------------------------------------------------------------------
# 场景 C：分离部署（P 节点加速 2x），独立配比最优化
# ---------------------------------------------------------------------------
wall_dis_opt, p_busy_opt, d_busy_opt = disaggregated(
    REQS, p_speed=2.0, d_speed=1.0, kv_latency=KV_LATENCY
)
print("\n[C] 分离部署 (disaggregated, p_speed=2.0, d_speed=1.0)  ← 独立配比优化")
print(f"    wall_time  : {wall_dis_opt*1000:.2f} ms")
print(f"    P 忙碌时间 : {p_busy_opt*1000:.2f} ms")
print(f"    D 忙碌时间 : {d_busy_opt*1000:.2f} ms")

# ---------------------------------------------------------------------------
# 汇总与利用率
# ---------------------------------------------------------------------------
print("\n" + SEP)
print("  汇总对比")
print(SEP)
print(f"  {'方案':<40} {'wall_time (ms)':>14}")
print(f"  {'-'*40} {'-'*14}")
print(f"  {'[A] 合并部署 (1.0/1.0)':<40} {wall_co*1000:>14.2f}")
print(f"  {'[B] 分离部署 (1.0/1.0, 流水线)':<40} {wall_dis_eq*1000:>14.2f}")
print(f"  {'[C] 分离部署 (2.0/1.0, 配比优化)':<40} {wall_dis_opt*1000:>14.2f}")

speedup_b = wall_co / wall_dis_eq
speedup_c = wall_co / wall_dis_opt
print(f"\n  [B] vs [A] 加速比（流水线）   : {speedup_b:.2f}x")
print(f"  [C] vs [A] 加速比（配比优化）  : {speedup_c:.2f}x")

print("\n  各引擎利用率（忙碌/wall_time）：")
print(f"  [A] P 利用率: {p_busy_co/wall_co*100:.1f}%  D 利用率: {d_busy_co/wall_co*100:.1f}%  (共享, 互相阻塞)")
print(f"  [C] P 利用率: {p_busy_opt/wall_dis_opt*100:.1f}%  D 利用率: {d_busy_opt/wall_dis_opt*100:.1f}%  (独立节点, 流水线重叠)")

print(SEP)

# ---------------------------------------------------------------------------
# 断言：分离部署（配比优化）吞吐必须优于合并部署
# ---------------------------------------------------------------------------
assert wall_dis_opt < wall_co, (
    f"分离部署（p_speed=2.0）应比合并部署更快！\n"
    f"  disaggregated={wall_dis_opt*1000:.2f}ms  colocated={wall_co*1000:.2f}ms"
)

assert wall_dis_eq < wall_co, (
    f"分离部署（等速）在长 prompt 场景下也应比合并部署更快（流水线收益）！\n"
    f"  disaggregated={wall_dis_eq*1000:.2f}ms  colocated={wall_co*1000:.2f}ms"
)

assert wall_dis_opt < wall_dis_eq, (
    f"提升 P 节点算力（p_speed=2.0）应比等速分离更快！\n"
    f"  opt={wall_dis_opt*1000:.2f}ms  eq={wall_dis_eq*1000:.2f}ms"
)

print("\n✅ adv10_pd_disaggregation 通过")
