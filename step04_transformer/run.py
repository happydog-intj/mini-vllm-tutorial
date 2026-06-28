import torch
from transformer import TinyTransformer

def main():
    torch.manual_seed(42)
    vocab_size, d_model, num_heads, num_layers = 256, 128, 4, 2

    model = TinyTransformer(vocab_size, d_model, num_heads, num_layers)
    param_count = sum(p.numel() for p in model.parameters())

    # 验证前向传播
    seq_len = 10
    token_ids = torch.randint(0, vocab_size, (seq_len,))
    logits = model(token_ids)

    assert logits.shape == (seq_len, vocab_size), f"输出 shape 错误: {logits.shape}"
    print(f"TinyTransformer: {num_layers}层, d_model={d_model}, heads={num_heads}, vocab={vocab_size}")
    print(f"参数量: {param_count:,}  (~{param_count/1e6:.1f}M)")
    print(f"输入: {token_ids.shape}  → 输出 logits: {logits.shape}")

    # 验证因果性：修改 token[-1] 后，前面位置的 logits 不变
    token_ids2 = token_ids.clone()
    token_ids2[-1] = (token_ids[-1] + 1) % vocab_size
    logits2 = model(token_ids2)
    assert torch.allclose(logits[:-1], logits2[:-1], atol=1e-5), "因果性违反！"
    print("\n因果性验证：修改 token[-1] 后，前面位置的 logits 不变 ✅")

    print("\n✅ step04_transformer 通过")

if __name__ == "__main__":
    main()
