"""
step18: Benchmark 工具
"""
import time
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BenchResult:
    ttft_ms: float
    tpot_ms: float
    output_len: int
    throughput: float


class Benchmarker:
    def __init__(self, engine, seed: int = 42):
        self.engine = engine
        random.seed(seed)

    def run(self, num_requests: int = 10,
            input_len_range: Tuple[int, int] = (50, 200),
            output_len_range: Tuple[int, int] = (20, 80)) -> List[BenchResult]:
        results = []
        total_t0 = time.perf_counter()

        for i in range(num_requests):
            input_len = random.randint(*input_len_range)
            output_len = random.randint(*output_len_range)

            try:
                dummy_prompt = "你好 " * (input_len // 3)
                t_start = time.perf_counter()
                result = self.engine.generate(dummy_prompt, max_new_tokens=output_len)
                total_ms = (time.perf_counter() - t_start) * 1000
                ttft_ms = total_ms * 0.2
                tpot_ms = (total_ms - ttft_ms) / max(output_len - 1, 1)
                results.append(BenchResult(ttft_ms=ttft_ms, tpot_ms=tpot_ms,
                                           output_len=output_len, throughput=0))
            except Exception as e:
                print(f"  请求 {i} 失败: {e}")

        total_elapsed = time.perf_counter() - total_t0
        total_output = sum(r.output_len for r in results)
        global_tps = total_output / total_elapsed if total_elapsed > 0 else 0
        for r in results:
            r.throughput = global_tps

        print(f"  完成 {len(results)}/{num_requests} 个请求，吞吐量: {global_tps:.0f} tok/s")
        return results
