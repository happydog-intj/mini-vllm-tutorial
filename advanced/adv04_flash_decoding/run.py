"""
adv04_flash_decoding/run.py

验证 flash_decode_splitk 与 naive_decode_attention 数值等价。
纯 CPU / PyTorch 实现，无 CUDA 依赖。
"""

import torch
from flash_decode import naive_decode_attention, flash_decode_splitk


def run_test(seq: int, heads: int, d_head: int, num_splits: int, seed: int = 42) -> None:
    torch.manual_seed(seed)
    q = torch.randn(heads, d_head)
    K = torch.randn(seq, heads, d_head)
    V = torch.randn(seq, heads, d_head)

    ref = naive_decode_attention(q, K, V)
    out = flash_decode_splitk(q, K, V, num_splits=num_splits)

    max_diff = (ref - out).abs().max().item()
    allclose = torch.allclose(ref, out, atol=1e-5)
    status = "PASS" if allclose else "FAIL"
    print(
        f"  seq={seq:5d}  heads={heads}  d_head={d_head}"
        f"  splits={num_splits}  max_diff={max_diff:.2e}  [{status}]"
    )
    assert allclose, (
        f"flash_decode_splitk 与 naive 结果不一致！max_diff={max_diff:.2e}"
    )


def main() -> None:
    print("=" * 60)
    print("adv04_flash_decoding 正确性验证")
    print("=" * 60)

    print("\n[1] 基础用例（seq 可整除 splits）")
    run_test(seq=128,  heads=8, d_head=64, num_splits=4)
    run_test(seq=512,  heads=8, d_head=64, num_splits=4)
    run_test(seq=1024, heads=8, d_head=64, num_splits=8)

    print("\n[2] seq 不能整除 splits（最后一段更短）")
    run_test(seq=100,  heads=4, d_head=32, num_splits=3)
    run_test(seq=1000, heads=4, d_head=32, num_splits=7)

    print("\n[3] 极端情况：splits=1（退化为单段，等同于 naive）")
    run_test(seq=256, heads=4, d_head=64, num_splits=1)

    print("\n[4] 极端情况：splits > seq（部分段为空，跳过）")
    run_test(seq=5, heads=2, d_head=16, num_splits=8)

    print("\n[5] 较长序列，模拟 decode 长 KV Cache 场景")
    run_test(seq=4096, heads=16, d_head=128, num_splits=16)
    run_test(seq=8192, heads=16, d_head=128, num_splits=32)

    print("\n" + "=" * 60)
    print("\n✅ adv04_flash_decoding 通过")


if __name__ == "__main__":
    main()
