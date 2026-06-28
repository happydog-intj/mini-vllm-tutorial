import torch
from attention import MultiHeadAttention, scaled_dot_product_attention

def main():
    torch.manual_seed(42)
    seq_len, d_model, num_heads = 4, 8, 2
    d_head = d_model // num_heads  # 4

    # 1. 测试 scaled_dot_product_attention
    Q = torch.randn(seq_len, d_head)
    K = torch.randn(seq_len, d_head)
    V = torch.randn(seq_len, d_head)
    output, weights = scaled_dot_product_attention(Q, K, V, causal=True)
    assert output.shape == (seq_len, d_head), f"输出 shape 错误: {output.shape}"
    assert weights.shape == (seq_len, seq_len)
    # 因果 mask：上三角（未来位置）权重必须为0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert weights[i, j].item() < 1e-6, f"未来位置 [{i},{j}] 权重非零"
    print("注意力权重矩阵 (因果 mask 生效):")
    print("        t0    t1    t2    t3")
    for i in range(seq_len):
        row = "  ".join(f"{weights[i,j].item():.2f}" for j in range(seq_len))
        print(f"  t{i}  [{row}]")

    # 2. 测试 MultiHeadAttention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(seq_len, d_model)
    out = mha(x)
    assert out.shape == (seq_len, d_model), f"MHA 输出 shape 错误: {out.shape}"
    print(f"\nMultiHeadAttention: 输入 {x.shape} → 输出 {out.shape}  ✅")

    print("\n✅ step03_attention 通过")

if __name__ == "__main__":
    main()
