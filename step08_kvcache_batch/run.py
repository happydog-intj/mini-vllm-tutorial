import torch
from engine import SerialEngine, BatchKVCacheEngine

def main():
    torch.manual_seed(42)

    # 4个请求，长度各不相同（模拟真实场景）
    requests = [
        (torch.tensor(list(range(10))),  15),  # prompt=10, max_new=15
        (torch.tensor(list(range(30))),  10),  # prompt=30, max_new=10
        (torch.tensor(list(range(5))),   20),  # prompt=5,  max_new=20
        (torch.tensor(list(range(20))),  12),  # prompt=20, max_new=12
    ]

    print("=" * 60)
    print("Static Batching：padding 结构可视化")
    print("=" * 60)

    # 展示 padding 后的矩阵形状
    prompt_lengths = [len(p) for p, _ in requests]
    max_new_list   = [n for _, n in requests]
    print(f"\nPrefill padding（pad 到最长 prompt = {max(prompt_lengths)} tokens）：")
    for i, (l, (prompt_ids, _)) in enumerate(zip(prompt_lengths, requests)):
        pad = max(prompt_lengths) - l
        bar = "█" * l + "░" * pad
        print(f"  请求{i}: [{bar}]  实际={l} pad={pad}")

    total_padded  = len(requests) * max(prompt_lengths)
    actual_tokens = sum(prompt_lengths)
    prefill_waste = (total_padded - actual_tokens) / total_padded
    print(f"\n  Prefill padding 浪费: {prefill_waste*100:.0f}%  "
          f"({total_padded - actual_tokens}/{total_padded} slots)")

    max_decode = max(max_new_list)
    print(f"\nDecode padding（等最长完成 = {max_decode} decode steps）：")
    for i, (decode_len, _) in enumerate(zip(max_new_list, requests)):
        wait = max_decode - decode_len
        bar = "█" * decode_len + "░" * wait
        print(f"  请求{i}: [{bar}]  实际={decode_len} idle={wait}")

    total_decode_slots = len(requests) * max_decode
    actual_decode = sum(max_new_list)
    decode_waste   = (total_decode_slots - actual_decode) / total_decode_slots if total_decode_slots > 0 else 0
    print(f"\n  Decode idle 浪费:     {decode_waste*100:.0f}%  "
          f"({total_decode_slots - actual_decode}/{total_decode_slots} steps)")

    # 实际运行 BatchKVCacheEngine，验证正确性
    batch_engine = BatchKVCacheEngine()
    batch_results = batch_engine.generate_batch(requests)
    assert len(batch_results) == len(requests)
    # 验证 engine 内部统计的 padding 浪费和我们手算的一致
    assert batch_engine.padded_prefill_slots == total_padded
    assert batch_engine.actual_prefill_tokens == actual_tokens

    serial_engine = SerialEngine()
    serial_results = serial_engine.generate_batch(requests)
    # 两种引擎输出长度应相同
    for i in range(len(requests)):
        assert len(batch_results[i]) == len(serial_results[i]), \
            f"请求{i}: batch={len(batch_results[i])} serial={len(serial_results[i])}"

    print(f"\n两种引擎输出长度一致 ✅")

    # 核心结论
    assert prefill_waste > 0.1, "应该展示出明显的 padding 浪费"
    assert decode_waste  > 0.0, "应该展示出 idle 等待浪费"

    print()
    print("Static Batching 的两大问题：")
    print(f"  1. Prefill padding：{prefill_waste*100:.0f}% 的 prefill 计算是无效 PAD  ⚠️")
    print(f"  2. Decode idle：    {decode_waste*100:.0f}% 的 decode 步骤是空转等待  ⚠️")
    print()
    print("注：GPU 上 batch matmul 有并行加速，但 padding 浪费始终存在。")
    print("    step09 解决问题 2（Continuous Batching），step12 解决问题 2 的内存碎片。")

    print("\n✅ step08_kvcache_batch 通过")

if __name__ == "__main__":
    main()
