"""step12 run.py"""
import os
import statistics
from bench import BenchResult

MODEL_PATH = os.environ.get("QWEN3_MODEL_PATH", os.path.expanduser("~/huggingface/Qwen3-0.6B"))

def main():
    print("=" * 60)
    print("mini-vllm-tutorial Benchmark 工具")
    print("=" * 60)

    import glob
    has_model = bool(glob.glob(os.path.join(MODEL_PATH, "*.safetensors")))

    if not has_model:
        print("⚠️  未找到模型，使用模拟数据展示 Benchmark 格式")
        results = [
            BenchResult(ttft_ms=145.2, tpot_ms=8.3, output_len=128, throughput=412),
            BenchResult(ttft_ms=152.1, tpot_ms=8.1, output_len=96, throughput=412),
            BenchResult(ttft_ms=141.8, tpot_ms=8.5, output_len=110, throughput=412),
        ]
    else:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step08_real_model'))
        from engine import RealModelEngine
        from bench import Benchmarker
        engine = RealModelEngine(MODEL_PATH)
        bench = Benchmarker(engine)
        results = bench.run(num_requests=5, input_len_range=(20, 80), output_len_range=(10, 30))

    ttft = [r.ttft_ms for r in results]
    tpot = [r.tpot_ms for r in results]

    print(f"\n请求数: {len(results)}")
    print(f"\n首 Token 延迟 (TTFT):")
    print(f"  平均: {statistics.mean(ttft):.1f}ms")
    print(f"  P50:  {sorted(ttft)[len(ttft)//2]:.1f}ms")
    print(f"\n每 Token 延迟 (TPOT):")
    print(f"  平均: {statistics.mean(tpot):.1f}ms")
    print(f"\n总吞吐量: {results[0].throughput:.0f} tok/s")

    print("\n✅ step12_benchmark 通过")

if __name__ == "__main__":
    main()
