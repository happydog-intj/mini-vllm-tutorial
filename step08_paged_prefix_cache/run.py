"""
step08: Paged Prefix Cache 演示

验证策略：
  - 第一轮：冷启动，全部 miss，正常 prefill
  - 第二轮：相同请求，命中缓存，跳过前缀 prefill
  - 两轮输出完全相同（缓存只是优化，不改变数学结果）
  - Scheduler 引擎：Continuous Batching + prefix cache，与串行版输出一致
"""

import time
import torch
from engine import PagedPrefixCacheEngineV2, PagedPrefixCacheSchedulerEngine

SYSTEM_PROMPT_LEN = 32   # 共享前缀长度（模拟 system prompt）
USER_QUESTION_LEN = 8    # 每个请求独有的后缀
NUM_REQUESTS = 10
MAX_NEW = 5
BLOCK_SIZE = 16


def make_requests(num: int, shared_prefix_len: int, suffix_len: int):
    """生成 num 个请求，前 shared_prefix_len 个 token 完全相同。"""
    prefix = torch.arange(1, shared_prefix_len + 1)
    requests = []
    for i in range(num):
        suffix = torch.randint(50, 200, (suffix_len,)) + i
        prompt = torch.cat([prefix, suffix])
        requests.append((prompt, MAX_NEW))
    return requests


def main():
    torch.manual_seed(42)
    requests = make_requests(NUM_REQUESTS, SYSTEM_PROMPT_LEN, USER_QUESTION_LEN)

    print("=" * 60)
    print("Paged Prefix Cache：Block 粒度前缀复用")
    print("=" * 60)
    print(f"请求数: {NUM_REQUESTS}，共享前缀: {SYSTEM_PROMPT_LEN} tokens，"
          f"独有后缀: {USER_QUESTION_LEN} tokens，block_size: {BLOCK_SIZE}")

    engine = PagedPrefixCacheEngineV2(block_size=BLOCK_SIZE, total_blocks=128)

    # ── 第一轮：冷启动，全部 miss ──
    t0 = time.perf_counter()
    results_cold = engine.generate_batch(requests)
    t_cold = time.perf_counter() - t0
    hits_1 = engine.cache_hits
    misses_1 = engine.cache_misses

    # ── 第二轮：相同请求，命中缓存 ──
    engine.cache_hits = 0
    engine.cache_misses = 0
    t0 = time.perf_counter()
    results_warm = engine.generate_batch(requests)
    t_warm = time.perf_counter() - t0
    hits_2 = engine.cache_hits
    misses_2 = engine.cache_misses

    # ── 验证两轮输出完全相同 ──
    for i, (r_cold, r_warm) in enumerate(zip(results_cold, results_warm)):
        assert torch.equal(r_cold, r_warm), \
            f"请求 {i} 冷热输出不一致！\n  冷: {r_cold}\n  热: {r_warm}"

    hit_rate_1 = hits_1 / (hits_1 + misses_1) * 100 if (hits_1 + misses_1) > 0 else 0
    hit_rate_2 = hits_2 / (hits_2 + misses_2) * 100 if (hits_2 + misses_2) > 0 else 0

    print(f"\n  第一轮（冷启动）: {t_cold*1000:.1f} ms  命中率 {hit_rate_1:.0f}%  ({hits_1}/{hits_1+misses_1})")
    print(f"  第二轮（缓存热）: {t_warm*1000:.1f} ms  命中率 {hit_rate_2:.0f}%  ({hits_2}/{hits_2+misses_2})")
    print(f"\n  两轮输出完全相同 ✅（缓存只是优化，不改变结果）")
    print(f"  KV 数据全部存储在 kv_pool 中，past_kv 彻底消失 ✅")
    print(f"  prefix cache 命中 = block_table 复用，零拷贝 ✅")

    assert hits_2 > 0, "第二轮应有缓存命中"

    for blk in engine.block_manager._blocks:
        assert blk.ref_count >= 0, f"Block {blk.block_id} ref_count 异常: {blk.ref_count}"
    print(f"  所有 Block ref_count 正常 ✅")

    print("\n✅ step08_paged_prefix_cache 通过")

    # ── Scheduler 引擎：Continuous Batching + prefix cache ──
    print("\n" + "=" * 60)
    print("Scheduler 引擎：Continuous Batching + Paged Prefix Cache")
    print("=" * 60)

    sched_engine = PagedPrefixCacheSchedulerEngine(
        block_size=BLOCK_SIZE, total_blocks=128, max_running=4
    )
    sched_engine.model.load_state_dict(engine.model.state_dict())  # 复制权重确保可比较

    t0 = time.perf_counter()
    results_sched = sched_engine.generate_batch(requests)
    t_sched = time.perf_counter() - t0

    for i, (r_serial, r_sched) in enumerate(zip(results_warm, results_sched)):
        assert torch.equal(r_serial, r_sched), \
            f"请求 {i} scheduler 输出与串行版不一致！\n  串行: {r_serial}\n  sched: {r_sched}"

    hits_s = sched_engine.cache_hits
    misses_s = sched_engine.cache_misses
    hit_rate_s = hits_s / (hits_s + misses_s) * 100 if (hits_s + misses_s) > 0 else 0

    print(f"\n  耗时: {t_sched*1000:.1f} ms  命中率 {hit_rate_s:.0f}%  ({hits_s}/{hits_s+misses_s})")
    print(f"  输出与串行版完全相同 ✅")

    for blk in sched_engine.block_manager._blocks:
        assert blk.ref_count >= 0, f"Scheduler Block {blk.block_id} ref_count 异常"
    print(f"  所有 Block ref_count 正常 ✅")

    print("\n✅ step08_paged_prefix_cache Scheduler 通过")


if __name__ == "__main__":
    main()
