import torch
from sampler import greedy_sample, temperature_sample, top_k_sample, top_p_sample, gumbel_max_sample

def main():
    torch.manual_seed(42)
    vocab_size = 256

    # 构造一个有尖峰的 logits 分布
    logits = torch.randn(vocab_size) * 2.0
    logits[65] = 10.0  # token 65 ('A') 打高分

    print(f"logits 峰值在 token 65 ('A'), 值=10.0\n")

    # 1. Greedy：必须选 token 65
    g = greedy_sample(logits)
    assert g.item() == 65, f"Greedy 应选 token 65，得到 {g.item()}"
    print(f"Greedy:          token {g.item():3d}  (确定性)")

    # 2. Temperature sampling
    torch.manual_seed(0)
    t_low = temperature_sample(logits, temperature=0.1)
    torch.manual_seed(0)
    t_high = temperature_sample(logits, temperature=2.0)
    print(f"Temperature=0.1: token {t_low.item():3d}  (低温→集中在高概率Token)")
    print(f"Temperature=2.0: token {t_high.item():3d}  (高温→更随机)")

    # 3. Top-k sampling
    torch.manual_seed(1)
    tk = top_k_sample(logits, k=10, temperature=1.0)
    print(f"Top-k (k=10):    token {tk.item():3d}  (从概率最高的10个里选)")

    # 4. Top-p (Nucleus) sampling
    torch.manual_seed(2)
    tp = top_p_sample(logits, p=0.9, temperature=1.0)
    print(f"Top-p (p=0.9):   token {tp.item():3d}  (从累积概率90%的Token里选)")

    # 5. Gumbel-Max
    torch.manual_seed(3)
    gm = gumbel_max_sample(logits, temperature=1.0)
    print(f"Gumbel-Max:      token {gm.item():3d}  (等价于temperature采样)")

    # 验证所有函数返回标量
    for fn, args in [
        (greedy_sample, (logits,)),
        (temperature_sample, (logits, 1.0)),
        (top_k_sample, (logits, 10, 1.0)),
        (top_p_sample, (logits, 0.9, 1.0)),
        (gumbel_max_sample, (logits, 1.0)),
    ]:
        result = fn(*args)
        assert result.shape == (), f"{fn.__name__} 应返回标量，得到 shape={result.shape}"

    print("\n✅ step02_sampler 通过")

if __name__ == "__main__":
    main()
