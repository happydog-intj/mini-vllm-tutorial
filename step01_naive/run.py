import time
import torch
from engine import NaiveEngine

def main():
    torch.manual_seed(42)
    engine = NaiveEngine(vocab_size=256, d_model=512, num_heads=8, num_layers=6)

    print("=" * 50)
    print("朴素自回归推理 — 速度随序列长度下降")
    print("=" * 50)

    prompt_ids = torch.tensor([72, 101, 108, 108, 111])  # "Hello"
    max_steps = 200
    times = []

    input_ids = prompt_ids.clone()
    for step in range(max_steps):
        t0 = time.perf_counter()
        next_id = engine.decode_one_step(input_ids)
        dt = (time.perf_counter() - t0) * 1000  # ms
        times.append(dt)
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)])

        if step in (4, 9, 49, 99, 149, 199):
            total_len = len(input_ids)
            print(f"  Step {step+1:3d}: {dt:.1f}ms/token | 序列总长: {total_len} | "
                  f"重算 KV 次数: {total_len}")

    # 验证：后20步平均时间 > 前20步平均时间
    early_avg = sum(times[:20]) / 20
    late_avg = sum(times[-20:]) / 20
    assert late_avg > early_avg * 1.05, (
        f"朴素推理应该越来越慢: early={early_avg:.2f}ms late={late_avg:.2f}ms"
    )
    print(f"\n前20步平均: {early_avg:.1f}ms  后20步平均: {late_avg:.1f}ms")
    print("→ 速度随序列长度线性下降 ⚠️")
    print("\n✅ step01_naive 通过")

if __name__ == "__main__":
    main()
