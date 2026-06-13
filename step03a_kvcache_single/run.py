import time
import torch
from engine import NaiveEngine, KVCacheEngine

def bench(engine_cls, prompt_ids, max_new_tokens):
    """运行一次生成，返回总耗时(ms)"""
    engine = engine_cls()
    # 预热
    _ = engine.generate(torch.tensor([65, 66, 67]), max_new_tokens=3)
    # 正式计时
    t0 = time.perf_counter()
    result = engine.generate(prompt_ids, max_new_tokens=max_new_tokens)
    return (time.perf_counter() - t0) * 1000, result

def main():
    torch.manual_seed(42)
    prompt_ids = torch.tensor([72, 101, 108, 108, 111])  # prompt len=5

    print("=" * 60)
    print("KV Cache 效果 — NaiveEngine vs KVCacheEngine")
    print("=" * 60)
    print(f"{'生成长度':>10}  {'NaiveEngine':>14}  {'KVCacheEngine':>14}  {'加速比':>8}")
    print("-" * 60)

    for max_new in [10, 30, 50]:
        naive_ms, naive_out = bench(NaiveEngine, prompt_ids, max_new)
        kvcache_ms, kvcache_out = bench(KVCacheEngine, prompt_ids, max_new)
        speedup = naive_ms / kvcache_ms
        print(f"{max_new:>9}tokens  {naive_ms:>12.0f}ms  {kvcache_ms:>12.0f}ms  {speedup:>7.1f}×")
        assert kvcache_ms < naive_ms, f"KV Cache 应该更快: {kvcache_ms:.1f}ms < {naive_ms:.1f}ms"

    print("\n→ 序列越长，KV Cache 加速越明显 ✅")

    # 验证两种引擎生成结果一致（相同种子）
    torch.manual_seed(0)
    naive_engine = NaiveEngine()
    torch.manual_seed(0)
    kv_engine = KVCacheEngine()

    out_naive = naive_engine.generate(prompt_ids, max_new_tokens=10)
    out_kv = kv_engine.generate(prompt_ids, max_new_tokens=10)

    assert torch.equal(out_naive, out_kv), (
        f"两引擎应输出相同结果\n  naive:   {out_naive.tolist()}\n  kvcache: {out_kv.tolist()}"
    )
    print("两种引擎生成结果完全一致 ✅")
    print("\n✅ step03a_kvcache_single 通过")

if __name__ == "__main__":
    main()
