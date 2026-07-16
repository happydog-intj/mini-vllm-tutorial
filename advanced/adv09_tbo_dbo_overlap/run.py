"""
adv09_tbo_dbo_overlap/run.py

运行 TBO 计算通信重叠对比实验。
  - 8 个 microbatch
  - 朴素串行 vs TBO 重叠
  - 断言 TBO 耗时 < 朴素串行耗时
"""

from overlap_sim import no_overlap, tbo_overlap

MICROBATCHES = list(range(8))   # 8 个 microbatch（内容仅作占位）
CT = 0.05   # compute 时间 (s)
MT = 0.03   # comm    时间 (s)
N = len(MICROBATCHES)

print("=" * 52)
print("  adv09: TBO / DBO 计算-通信重叠 对比实验")
print("=" * 52)
print(f"  microbatch 数量 : {N}")
print(f"  compute 时间    : {CT * 1000:.0f} ms / microbatch")
print(f"  comm    时间    : {MT * 1000:.0f} ms / microbatch")
print(f"  理论朴素耗时    : {N * (CT + MT) * 1000:.0f} ms")
print(f"  理论 TBO  耗时  : ~{(N * max(CT, MT) + min(CT, MT)) * 1000:.0f} ms")
print("-" * 52)

t_no  = no_overlap(MICROBATCHES,  ct=CT, mt=MT)
t_tbo = tbo_overlap(MICROBATCHES, ct=CT, mt=MT)

print(f"  no_overlap  实测: {t_no  * 1000:.1f} ms")
print(f"  tbo_overlap 实测: {t_tbo * 1000:.1f} ms")
speedup = t_no / t_tbo
print(f"  加速比          : {speedup:.2f}x")
print("-" * 52)

assert t_tbo < t_no, (
    f"TBO 应比朴素串行快！"
    f"tbo={t_tbo*1000:.1f}ms no_overlap={t_no*1000:.1f}ms"
)

print("\n✅ adv09_tbo_dbo_overlap 通过")
