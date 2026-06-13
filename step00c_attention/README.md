# step00c — Attention：手写注意力机制

## 教学目标

完全理解 Scaled Dot-Product Attention 和多头注意力的计算过程。

## Q/K/V 的含义

图书馆检索类比：

```
Q (Query)：你的检索关键词      e.g. "neural network"
K (Key)：  每本书的标签/索引   e.g. ["deep learning", "python", ...]
V (Value)：每本书的实际内容

scores[i,j] = Q[i] · K[j]  ← 当前token与位置j的相关性
output[i]   = Σ_j softmax(scores[i]) * V[j]  ← 加权汇总
```

## 计算流程

```
输入 x: [seq_len, d_model]
    │
    ├─→ W_q → Q: [seq_len, d_head]
    ├─→ W_k → K: [seq_len, d_head]
    └─→ W_v → V: [seq_len, d_head]

scores = Q·Kᵀ / √d_head     [seq_len, seq_len]
    │
    ├─→ 因果 mask: 上三角置 -∞
    │
    ▼
weights = softmax(scores)    [seq_len, seq_len]
    │
    ▼
output = weights · V         [seq_len, d_head]
```

## 因果 Mask 为什么必须有？

生成时：模型预测 t3 时不能用 t4, t5... 的信息（因为还没生成）：

```
scores[3,4] = -∞  → softmax → 0
scores[3,5] = -∞  → softmax → 0
```

未来位置权重强制为0。

## 运行

```bash
python run.py
```
