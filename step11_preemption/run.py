import torch
from engine import NoPreemptionEngine, PreemptionEngine

def main():
    torch.manual_seed(42)

    # 8 个请求，故意设置很小的 KV 槽位（max_kv_slots=20）触发内存不足
    requests = [
        (torch.randint(1, 255, (5,)), 8)
        for _ in range(8)
    ]

    print("=" * 55)
    print("Preemption：KV Cache 满时优雅降级 vs 崩溃")
    print("=" * 55)

    # 无抢占：内存满了直接报错
    try:
        engine = NoPreemptionEngine(max_kv_slots=20)
        engine.generate_batch(requests)
        print("  无 Preemption: 运行成功（KV 槽位充足）")
    except RuntimeError as e:
        print(f"  无 Preemption: RuntimeError: {str(e)[:60]} 💥")

    # 有抢占：内存满了驱逐低优先级请求，继续运行
    engine = PreemptionEngine(max_kv_slots=20)
    results = engine.generate_batch(requests)
    print(f"  有 Preemption: 全部 {len(results)} 个请求成功完成 ✅")
    print(f"  驱逐发生次数: {engine.preempt_count}")

    assert len(results) == len(requests), "所有请求都应完成"
    assert engine.preempt_count > 0, "应该发生过抢占（KV 槽位故意设小）"

    print("\n✅ step11_preemption 通过")

if __name__ == "__main__":
    main()
