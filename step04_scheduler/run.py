import time
import torch
from engine import StaticBatchingEngine, ContinuousBatchingEngine

def simulate_requests(n=8, seed=42):
    torch.manual_seed(seed)
    requests = []
    for _ in range(n):
        prompt_len = torch.randint(5, 20, ()).item()
        max_new = torch.randint(10, 50, ()).item()
        prompt_ids = torch.randint(1, 255, (prompt_len,))
        requests.append((prompt_ids, max_new))
    return requests

def main():
    requests = simulate_requests(n=8)

    static_engine = StaticBatchingEngine()
    t0 = time.perf_counter()
    static_engine.generate_batch(requests)
    static_ms = (time.perf_counter() - t0) * 1000

    cb_engine = ContinuousBatchingEngine()
    t0 = time.perf_counter()
    cb_engine.generate_batch(requests)
    cb_ms = (time.perf_counter() - t0) * 1000

    speedup = static_ms / cb_ms

    print("=" * 55)
    print("8 个并发请求：Static vs Continuous Batching")
    print("=" * 55)
    print(f"  Static Batching:     {static_ms:>8.0f} ms")
    print(f"  Continuous Batching: {cb_ms:>8.0f} ms  提升 {speedup:.1f}×")

    assert cb_ms < static_ms, (
        f"Continuous Batching 应该更快: {cb_ms:.0f}ms < {static_ms:.0f}ms"
    )
    print("\n✅ step04_scheduler 通过")

if __name__ == "__main__":
    main()
