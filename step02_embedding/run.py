import torch
from embedding import Embedding, cosine_similarity

def main():
    vocab_size = 256
    d_model = 8
    emb = Embedding(vocab_size, d_model)

    # 基础：token_id → 向量
    token_id = 65  # 'A'
    vec = emb(torch.tensor([token_id]))
    assert vec.shape == (1, d_model), f"shape 错误: {vec.shape}"
    print(f"token_id={token_id} ('A') → 向量 shape: {vec.shape}")
    print(f"  向量值: {[round(x, 2) for x in vec[0].tolist()]}")

    # 批量 embedding
    ids = torch.tensor([72, 101, 108, 108, 111])  # 'Hello'
    vecs = emb(ids)
    assert vecs.shape == (5, d_model)
    print(f"\n'Hello' token IDs {ids.tolist()} → 矩阵 shape: {vecs.shape}")

    # 余弦相似度：自身相似度必须为1
    sim_aa = cosine_similarity(emb(torch.tensor([65])), emb(torch.tensor([65])))
    sim_ab = cosine_similarity(emb(torch.tensor([65])), emb(torch.tensor([66])))
    print(f"\n余弦相似度('A','A') = {sim_aa:.4f}  （应为1.0）")
    print(f"余弦相似度('A','B') = {sim_ab:.4f}  （随机初始化接近0）")
    assert abs(sim_aa - 1.0) < 1e-5, "self-similarity 应为1"

    print("\n✅ step02_embedding 通过")

if __name__ == "__main__":
    main()
