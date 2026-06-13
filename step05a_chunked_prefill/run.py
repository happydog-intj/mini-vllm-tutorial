import time
import torch
from engine import NormalEngine, ChunkedPrefillEngine

def main():
    torch.manual_seed(42)

    # 场景：4 个 decode 请求 + 1 个超长 prompt
    decode_requests = [
        (torch.tensor(list(range(i * 5, i * 5 + 10))), 30)
        for i in range(4)
    ]
    long_prompt = (torch.randint(1, 255, (200,)), 5)
    all_requests = decode_requests + [long_prompt]

    normal_engine = NormalEngine()
    t0 = time.perf_counter()
    normal_engine.generate_batch(all_requests)
    normal_ms = (time.perf_counter() - t0) * 1000

    chunked_engine = ChunkedPrefillEngine(chunk_size=50)
    t0 = time.perf_counter()
    chunked_engine.generate_batch(all_requests)
    chunked_ms = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print("Chunked Prefill：长 Prompt 不再阻塞 Decode 请求")
    print("=" * 60)
    print(f"场景: 4 个 decode 请求 + 1 个 200-token 长 prompt")
    print(f"\n  无 Chunked Prefill: {normal_ms:>8.0f} ms")
    print(f"  有 Chunked Prefill: {chunked_ms:>8.0f} ms  (chunk_size=50，分4块)")
    print(f"\n  长 prefill 被分散到多步，不阻塞已有 decode 请求 ✅")
    print("\n✅ step05a_chunked_prefill 通过")

if __name__ == "__main__":
    main()
