import time
import torch
from engine import NoPrefixCacheEngine, PrefixCacheEngine

def main():
    torch.manual_seed(42)

    # 场景：系统提示词（100 token）+ 10 个用户问题（各 10~30 token）
    system_prompt = torch.randint(1, 255, (100,))
    user_questions = []
    for i in range(10):
        q_len = 10 + (i * 2) % 20  # 10~28 tokens
        user_questions.append(torch.randint(1, 255, (q_len,)))

    # 每个请求 = 系统提示词 + 用户问题
    requests = [
        (torch.cat([system_prompt, q]), 5)
        for q in user_questions
    ]

    # 统计 token 数
    no_cache_prefill_tokens = sum(len(p) for p, _ in requests)
    system_tokens = len(system_prompt)
    user_tokens = sum(len(q) for q in user_questions)
    cache_prefill_tokens = system_tokens + user_tokens  # 系统提示词只算一次

    savings = (no_cache_prefill_tokens - cache_prefill_tokens) / no_cache_prefill_tokens

    # 实际计时
    no_cache_engine = NoPrefixCacheEngine()
    t0 = time.perf_counter()
    no_cache_engine.generate_batch(requests)
    no_cache_ms = (time.perf_counter() - t0) * 1000

    cache_engine = PrefixCacheEngine(block_size=16)
    t0 = time.perf_counter()
    cache_engine.generate_batch(requests)
    cache_ms = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print("Prefix Caching：系统提示词场景")
    print("=" * 60)
    print(f"系统提示词: {system_tokens} tokens（所有请求共享）")
    avg_q = sum(len(q) for q in user_questions) // len(user_questions)
    print(f"用户问题平均: {avg_q} tokens")
    print(f"\n  无前缀缓存: prefill 总计 {no_cache_prefill_tokens} tokens，耗时 {no_cache_ms:.0f}ms")
    print(f"  有前缀缓存: prefill 总计 {cache_prefill_tokens} tokens，耗时 {cache_ms:.0f}ms")
    print(f"  节省计算: {savings*100:.0f}% ✅")
    print(f"\n  缓存命中次数: {cache_engine.cache_hits}/{len(requests)}")

    assert savings > 0.5, f"前缀缓存应节省 >50% prefill，实际 {savings*100:.0f}%"
    assert cache_engine.cache_hits >= len(requests) - 1, \
        f"除第1个请求，其余应命中缓存，实际命中 {cache_engine.cache_hits}"

    print("\n✅ step07_prefix_cache 通过")

if __name__ == "__main__":
    main()
