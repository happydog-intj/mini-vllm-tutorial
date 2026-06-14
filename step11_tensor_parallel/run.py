import torch
from linear import ColumnParallelLinear, RowParallelLinear

def main():
    torch.manual_seed(42)
    in_features, out_features = 256, 512
    tp_size = 1

    print("Tensor Parallelism 线性层验证")
    print("=" * 45)

    batch_size, seq_len = 4, 16
    x = torch.randn(batch_size, seq_len, in_features)

    col = ColumnParallelLinear(in_features, out_features, tp_size=tp_size)
    std = torch.nn.Linear(in_features, out_features, bias=False)
    std.weight.data = col.weight.data.clone()
    out_col = col(x)
    out_std = std(x)
    assert out_col.shape == out_std.shape
    assert torch.allclose(out_col, out_std, atol=1e-4)
    print(f"ColumnParallelLinear (tp_size=1): {x.shape} → {out_col.shape}  ✅")

    row = RowParallelLinear(out_features, in_features, tp_size=tp_size)
    std2 = torch.nn.Linear(out_features, in_features, bias=False)
    std2.weight.data = row.weight.data.clone()
    out_row = row(out_col)
    out_std2 = std2(out_std)
    assert out_row.shape == out_std2.shape
    assert torch.allclose(out_row, out_std2, atol=1e-4)
    print(f"RowParallelLinear (tp_size=1):    {out_col.shape} → {out_row.shape}  ✅")

    print(f"\n切分策略说明 (tp_size=2 时):")
    print(f"  ColumnParallel: weight [{in_features}×{out_features}] → [{in_features}×{out_features//2}] × 2 GPU")
    print(f"  RowParallel:    weight [{out_features}×{in_features}] → [{out_features//2}×{in_features}] × 2 GPU")
    print(f"    + all_reduce 跨 GPU 求和")

    print("\n✅ step11_tensor_parallel 通过")

if __name__ == "__main__":
    main()
