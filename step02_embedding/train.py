"""
step02: 训练演示 — 用字符级语言模型证明 Embedding 是可学习的

教学要点:
  - Embedding 的 weight 是 nn.Parameter，梯度会随 loss 反向传播更新
  - 用极简 next-token 预测任务：输入当前字符，预测下一个字符
  - 训练后：语义/用法相近的字符在向量空间中距离更近
"""

import torch
import torch.nn as nn
from embedding import Embedding, cosine_similarity


def train():
    torch.manual_seed(42)
    vocab_size = 256
    d_model = 16

    # 极简语言模型：Embedding → 线性层 → 预测下一个 token
    emb = Embedding(vocab_size, d_model)
    head = nn.Linear(d_model, vocab_size, bias=False)  # LM head
    optimizer = torch.optim.Adam(
        list(emb.parameters()) + list(head.parameters()), lr=0.01
    )
    loss_fn = nn.CrossEntropyLoss()

    # 训练数据：两组随机序列，组内随机采样，组间完全隔离
    # 小写组：随机从 a-j 中采样 → 组内所有字符共现模式相似
    # 大写组：随机从 A-J 中采样 → 组内所有字符共现模式相似
    # 关键：交替训练，两组之间没有任何 token 边界接触 → 跨组向量不相似
    import random
    random.seed(42)
    lower = [ord(c) for c in "abcdefghij"]
    upper = [ord(c) for c in "ABCDEFGHIJ"]
    lower_ids = torch.tensor([random.choice(lower) for _ in range(2000)], dtype=torch.long)
    upper_ids = torch.tensor([random.choice(upper) for _ in range(2000)], dtype=torch.long)

    # 训练前：随机初始化，同组字符相似度接近0
    sim_before = cosine_similarity(
        emb(torch.tensor([ord("a")])), emb(torch.tensor([ord("b")]))
    )
    print(f"训练前  sim('a','b') = {sim_before:.4f}  （随机初始化）")
    print()

    # 训练：两组交替取窗口，组内 next-token prediction
    for step in range(500):
        src = lower_ids if step % 2 == 0 else upper_ids
        i = torch.randint(0, len(src) - 64, (1,)).item()
        x = src[i : i + 64]      # 输入：当前 token
        y = src[i + 1 : i + 65]  # 目标：下一个 token（同组内）

        logits = head(emb(x))       # [64, vocab_size]
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"  step {step+1:4d}  loss={loss.item():.4f}")

    print()

    # 训练后对比
    sim_ab = cosine_similarity(
        emb(torch.tensor([ord("a")])), emb(torch.tensor([ord("b")]))
    )
    sim_aA = cosine_similarity(
        emb(torch.tensor([ord("a")])), emb(torch.tensor([ord("A")]))
    )
    print(f"训练后  sim('a','b') = {sim_ab:.4f}  （同组小写，应 > 训练前）")
    print(f"训练后  sim('a','A') = {sim_aA:.4f}  （跨组，应 < sim('a','b')）")

    assert sim_ab > sim_before, "训练后同组字符相似度应上升"
    assert sim_ab > sim_aA, "同组字符应比跨组字符更相似"

    print("\n✅ Embedding 可学习性验证通过")


if __name__ == "__main__":
    train()
