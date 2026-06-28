import torch
from attention import standard_attention, flash_attention, is_flash_attn_available

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"设备: {device}")
    print(f"FlashAttention 可用: {is_flash_attn_available()}")

    torch.manual_seed(0)
    seq_len, num_heads, head_dim = 64, 8, 64
    q = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    out_std = standard_attention(q, k, v)
    out_flash = flash_attention(q, k, v)
    max_diff = (out_std.float() - out_flash.float()).abs().max().item()
    print(f"\n正确性验证: max_diff = {max_diff:.6f}  （< 0.02 即通过）")
    assert max_diff < 0.02, f"输出差异过大: {max_diff}"
    print("两者输出一致 ✅")

    print("\n✅ step09_flash_attention 通过")

if __name__ == "__main__":
    main()
