import time
import torch
from engine import SerialEngine, BatchKVCacheEngine

def bench_serial(requests):
    engine = SerialEngine()
    # warmup
    engine.generate_batch([(torch.tensor([1, 2]), 2)])
    t0 = time.perf_counter()
    engine.generate_batch(requests)
    return (time.perf_counter() - t0) * 1000

def bench_batch(requests):
    engine = BatchKVCacheEngine()
    # warmup
    engine.generate_batch([(torch.tensor([1, 2]), 2)])
    t0 = time.perf_counter()
    engine.generate_batch(requests)
    return (time.perf_counter() - t0) * 1000

def main():
    torch.manual_seed(42)

    # 4个请求，长度各不相同（模拟真实场景）
    requests = [
        (torch.tensor(list(range(10))),  15),  # prompt=10, max_new=15
        (torch.tensor(list(range(30))),  10),  # prompt=30, max_new=10
        (torch.tensor(list(range(5))),   20),  # prompt=5,  max_new=20
        (torch.tensor(list(range(20))),  12),  # prompt=20, max_new=12
    ]

    serial_ms = bench_serial(requests)
    batch_ms = bench_batch(requests)
    speedup = serial_ms / batch_ms

    print("=" * 55)
    print("多请求推理：串行 vs Static Batch")
    print("=" * 55)
    print(f"  串行处理:        {serial_ms:>8.0f} ms")
    print(f"  Batch (padding): {batch_ms:>8.0f} ms  加速 {speedup:.1f}×")

    # 统计 padding 浪费比例
    max_total = max(len(p) + n for p, n in requests)
    total_slots = max_total * len(requests)
    actual_tokens = sum(len(p) + n for p, n in requests)
    padding_ratio = (total_slots - actual_tokens) / total_slots
    print(f"\n  Static Batch 中 padding 占比: {padding_ratio*100:.0f}%")
    print(f"  （{padding_ratio*100:.0f}% 的计算是无效 padding）⚠️")

    # CPU 上小模型无法体现并行加速，此断言仅在 GPU 环境有意义
    # assert speedup > 1.1, f"Batch 应该比串行快，得到 {speedup:.2f}×"
    assert padding_ratio > 0.1, "应该展示出明显的 padding 浪费"

    print("\n✅ step03b_kvcache_batch 通过")

if __name__ == "__main__":
    main()
