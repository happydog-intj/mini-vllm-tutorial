"""
adv08 run.py: 对比 RoundRobin vs LeastLoad(DPLB) 在异构副本下的负载分布。

场景设计原理:
  - replica-B 速率为 0.3，即处理能力仅 30 token/步（模拟 GPU 热限速或显存带宽瓶颈）
  - 突发期到达速率约 83 token/步（每副本），远超 B 的处理能力(30)，低于 A/C 的处理能力(100)
  - RoundRobin 朴素轮询：B 收到 1/3 流量 → 严重积压，A/C 有大量空余容量
  - LeastLoad  DPLB   ：检测到 B 积压 → 将流量重定向至 A/C → B 可迅速排空，整体更均衡
"""

import random
import statistics

from dp_sim import LeastLoadLB, Replica, RoundRobinLB, simulate

# ─── 场景参数 ─────────────────────────────────────────────────────────────────
SEED = 42
TOTAL_TIME = 100
BURST_START = 10     # 突发流量起始时刻
BURST_END = 40       # 突发流量结束时刻（持续 30 步）

random.seed(SEED)

# 生成请求到达序列
arrivals: list[tuple[int, int]] = []
for t in range(TOTAL_TIME):
    if BURST_START <= t < BURST_END:
        # 突发期：每步 4~6 个请求，每个 40~60 token
        n = random.randint(4, 6)
        req_size = lambda: random.randint(40, 60)
    else:
        # 正常期：每步 0~1 个请求
        n = random.randint(0, 1)
        req_size = lambda: random.randint(20, 40)
    for _ in range(n):
        arrivals.append((t, req_size()))

# 副本配置：异构速率
#   A = 1.0  标准 GPU（处理 ~100 token/步）
#   B = 0.3  严重降速副本（处理 ~30  token/步，模拟热限速/显存带宽瓶颈）
#   C = 1.2  高性能副本（处理 ~100 token/步，速率上限与 in_flight 上限相同）
REPLICA_CONFIGS = [
    ("replica-A", 1.0),
    ("replica-B", 0.3),   # 严重降速：RoundRobin 下将积压大量 token
    ("replica-C", 1.2),
]


def make_replicas() -> list[Replica]:
    return [Replica(name, speed) for name, speed in REPLICA_CONFIGS]


# ─── 运行两种策略 ──────────────────────────────────────────────────────────────

rr_replicas = make_replicas()
rr_log = simulate(rr_replicas, RoundRobinLB(), arrivals, TOTAL_TIME)

ll_replicas = make_replicas()
ll_log = simulate(ll_replicas, LeastLoadLB(), arrivals, TOTAL_TIME)

# ─── 统计分析 ──────────────────────────────────────────────────────────────────

def avg_cross_replica_variance(log: list[dict[str, int]]) -> float:
    """计算每个时间步各副本负载方差，再取平均，衡量负载不均衡程度。"""
    variances = []
    for snapshot in log:
        loads = list(snapshot.values())
        if len(loads) > 1:
            variances.append(statistics.variance(loads))
    return statistics.mean(variances) if variances else 0.0


def peak_cross_replica_variance(log: list[dict[str, int]]) -> float:
    """峰值方差，反映最坏时刻的不均衡程度。"""
    variances = []
    for snapshot in log:
        loads = list(snapshot.values())
        if len(loads) > 1:
            variances.append(statistics.variance(loads))
    return max(variances) if variances else 0.0


def total_load_per_replica(log: list[dict[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for snapshot in log:
        for name, load in snapshot.items():
            totals[name] = totals.get(name, 0) + load
    return totals


rr_avg_var = avg_cross_replica_variance(rr_log)
ll_avg_var = avg_cross_replica_variance(ll_log)
rr_peak_var = peak_cross_replica_variance(rr_log)
ll_peak_var = peak_cross_replica_variance(ll_log)

rr_totals = total_load_per_replica(rr_log)
ll_totals = total_load_per_replica(ll_log)

# ─── 输出报告 ──────────────────────────────────────────────────────────────────

print("=" * 65)
print("adv08: Data Parallel + DPLB 负载均衡对比")
print("=" * 65)
print(f"\n场景: {len(REPLICA_CONFIGS)} 副本 | 请求总数: {len(arrivals)}")
print(f"      突发窗口: t={BURST_START}~{BURST_END} ({BURST_END-BURST_START} 步)")
print(f"      副本速率: A=1.0, B=0.3(降速副本), C=1.2 | 总时间步: {TOTAL_TIME}\n")

print("─" * 65)
print("策略 1: RoundRobin (朴素轮询)")
print("─" * 65)
for name, total in rr_totals.items():
    bar = "█" * (total // 2000)
    print(f"  {name}: 累计负载={total:7d}  {bar}")
print(f"  平均跨副本负载方差: {rr_avg_var:,.0f}")
print(f"  峰值跨副本负载方差: {rr_peak_var:,.0f}")

print()
print("─" * 65)
print("策略 2: LeastLoad / DPLB (最小负载路由)")
print("─" * 65)
for name, total in ll_totals.items():
    bar = "█" * (total // 2000)
    print(f"  {name}: 累计负载={total:7d}  {bar}")
print(f"  平均跨副本负载方差: {ll_avg_var:,.0f}")
print(f"  峰值跨副本负载方差: {ll_peak_var:,.0f}")

print()
print("─" * 65)
print("对比结论")
print("─" * 65)
avg_improvement = (rr_avg_var - ll_avg_var) / rr_avg_var * 100 if rr_avg_var > 0 else 0
peak_improvement = (rr_peak_var - ll_peak_var) / rr_peak_var * 100 if rr_peak_var > 0 else 0
print(f"  RoundRobin 平均方差: {rr_avg_var:,.0f}  峰值方差: {rr_peak_var:,.0f}")
print(f"  LeastLoad  平均方差: {ll_avg_var:,.0f}  峰值方差: {ll_peak_var:,.0f}")
print(f"  平均方差改善: {avg_improvement:.1f}%  |  峰值方差改善: {peak_improvement:.1f}%")
print()
print("  结论: B 副本在 RoundRobin 下严重积压（处理速率仅 0.3），")
print("        DPLB 检测到其负载升高后将流量重定向至 A/C，")
print("        使各副本负载更均衡，降低端到端请求延迟方差。")
print()

# ─── 断言：DPLB 平均方差 < RoundRobin 平均方差 ─────────────────────────────────
assert ll_avg_var < rr_avg_var, (
    f"断言失败: LeastLoad 平均方差({ll_avg_var:.0f}) 应 < RoundRobin 平均方差({rr_avg_var:.0f})\n"
    "请检查模拟参数：需确保突发强度在 B 处理上限以上、A/C 处理上限以下。"
)

print("\n✅ adv08_data_parallel_dplb 通过")
