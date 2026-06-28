# step03 — Attention：从零理解注意力机制

## 为什么需要注意力机制？

在 Transformer 出现之前，处理序列的主流方式是 RNN（循环神经网络）。RNN 按时间步依次处理 token，每个时间步把"记忆"压缩进一个固定大小的隐向量后传给下一步。

这带来了一个根本性问题：**长距离依赖很难捕获**。

```
RNN 的信息传递：

t1 → h1 → t2 → h2 → t3 → h3 → ... → t100 → h100
           ↑           ↑
       压缩成向量   再压缩一次

问："t1 的信息到 t100 时还剩多少？"
答："经过 99 次压缩，几乎没了。"
```

注意力机制的思路完全不同：**让每个 token 直接和序列中所有其他 token 交互**，不经过中间压缩，距离远近不影响信息传递质量。

```
Attention 的信息传递：

t1  t2  t3  t4  ... t100
 ↘  ↘  ↘  ↘      ↙
       t50 直接和所有位置交互
```

这就是为什么 Transformer 在处理长文本时远优于 RNN。

---

## Q/K/V 的设计直觉

注意力的核心问题是：**当前 token 应该从序列的哪些位置"收集"信息？收集多少？**

三个矩阵各司其职：

- **Q（Query，查询）**：当前 token"我想要什么信息"的表示
- **K（Key，键）**：每个位置"我能提供什么"的标签
- **V（Value，值）**：每个位置实际携带的内容

点积 `Q[i] · K[j]` 度量"位置 i 的需求"和"位置 j 的供给"的匹配程度。匹配得越好，位置 j 的 V 在输出中占的权重越大。

```
位置 i 的 output：
  output[i] = Σ_j  softmax(Q[i]·K[j] / √d) · V[j]
               ↑              ↑                  ↑
          对所有 j 加权    权重（0~1，和为1）    j 的内容
```

为什么 Q、K、V 是通过线性投影得到的，而不是直接用输入？
因为原始输入 `x` 是同一个向量，用不同的权重矩阵（`W_q`、`W_k`、`W_v`）投影，可以让模型**学习到不同的表示视角**：查询时用一种视角，被检索时用另一种视角，提供内容时用第三种视角。

---

## 为什么要除以 √d？

直觉上，点积 `Q·K` 的值是 `d_head` 个维度乘积之和。假设 Q 和 K 的每个分量都是均值 0、方差 1 的随机变量，则点积的方差是 `d_head`，标准差是 `√d_head`。

当 `d_head` 较大（比如 64、128），未缩放的点积绝对值会很大：

```
d_head = 64 时，点积的典型量级 ~ √64 = 8
d_head = 128 时，点积的典型量级 ~ √128 ≈ 11
```

把这些大数值送入 softmax 会发生什么？

```
softmax([-8, 0, 8])  →  [接近0,  接近0,  接近1]
softmax([-1, 0, 1])  →  [0.09,   0.47,   0.44]
```

当点积绝对值很大时，softmax 输出会趋向 one-hot（一个位置接近 1，其余接近 0）。这意味着**梯度几乎为零**，模型无法从大多数位置学到有效信号。

除以 `√d_head` 把点积的量级压回 ~1 的范围，使 softmax 的梯度保持健康，训练时梯度能正常传播。

---

## 计算流程

```
输入 x: [seq_len, d_model]
    │
    ├─→ W_q (线性投影) → Q: [seq_len, d_head]
    ├─→ W_k (线性投影) → K: [seq_len, d_head]
    └─→ W_v (线性投影) → V: [seq_len, d_head]
            │
            ▼
    scores = Q · Kᵀ / √d_head      形状: [seq_len, seq_len]
            │
            ├─→ 因果 mask（上三角置 -∞）
            │
            ▼
    weights = softmax(scores, dim=-1)  形状: [seq_len, seq_len]
            │                          每行和为 1
            ▼
    output = weights · V               形状: [seq_len, d_head]
```

---

## 因果 Mask 为什么对生成至关重要？

语言模型是**自回归**的：生成第 t 个 token 时，第 t+1、t+2... 个 token 还不存在。

如果不加掩码，模型在训练时会"作弊"——直接看到答案再预测答案，根本学不到真正的生成能力：

```
预测 t3 时看到的应该是：
  ✅ t0, t1, t2 → 合法（已生成的历史）
  ❌ t4, t5, ... → 非法（未来，还没生成）
```

因果 mask 把注意力矩阵的上三角（未来位置）全部置为 `-∞`，经过 softmax 后这些位置的权重变为 0：

```
scores 矩阵（4个 token）：

        t0    t1    t2    t3
  t0  [ s00  -inf  -inf  -inf ]   t0 只能看自己
  t1  [ s10   s11  -inf  -inf ]   t1 能看 t0, t1
  t2  [ s20   s21   s22  -inf ]   t2 能看 t0, t1, t2
  t3  [ s30   s31   s32   s33 ]   t3 能看所有历史

softmax 后（-inf → 0）：

        t0    t1    t2    t3
  t0  [1.00  0.00  0.00  0.00]
  t1  [w10   w11   0.00  0.00]
  t2  [w20   w21   w22   0.00]
  t3  [w30   w31   w32   w33]
```

注意：这个约定来自 PyTorch 的 `torch.triu(..., diagonal=1)` 配合 `masked_fill(..., float("-inf"))`——是实现层面的标准做法，不是任意选择的。

---

## 多头注意力（Multi-Head Attention）

单头注意力在整个 `d_model` 维空间里计算一次注意力。多头注意力的想法是：**把维度切成若干份，每份独立做注意力**。

```
d_model = 8, num_heads = 2, d_head = 4

输入 x: [seq_len, 8]
    │
    ├─ W_q → Q: [seq_len, 8]
    │         ↓ 切成 2 个头
    │    Head0: Q[..., :4]    Head1: Q[..., 4:]
    │
   (K, V 同理)
    │
    ├─ Head 0: attention(Q0, K0, V0) → out0: [seq_len, 4]
    ├─ Head 1: attention(Q1, K1, V1) → out1: [seq_len, 4]
    │
    └─ concat([out0, out1]) → [seq_len, 8]
                │
               W_o → 输出: [seq_len, 8]
```

为什么这样设计有用？不同的头可以学习不同类型的依赖关系——有的头可能关注语法结构，有的关注语义相似性，有的关注位置邻近性。这是模型能力的来源之一，而不只是参数量的堆叠。

代价是：多头注意力比单头有更多线性投影（`W_q`、`W_k`、`W_v`、`W_o` 各一个），但 `d_head` 更小，每头的计算量与单头 `d_head` 版本相当。

---

## 运行

```bash
python run.py
```

预期输出（seed=42，seq_len=4，d_model=8，num_heads=2）：

```
注意力权重矩阵 (因果 mask 生效):
        t0    t1    t2    t3
  t0  [1.00  0.00  0.00  0.00]
  t1  [...]  [...]  0.00  0.00]
  t2  [...]  [...]  [...]  0.00]
  t3  [...]  [...]  [...]  [...]]

MultiHeadAttention: 输入 torch.Size([4, 8]) → 输出 torch.Size([4, 8])  ✅

✅ step03_attention 通过
```

上三角位置（未来 token）的权重全部为 0，因果 mask 生效。

---

## 关键代码对照

`attention.py` 中 `scaled_dot_product_attention` 的四步与计算流程一一对应：

```python
# Step 1: 相似度分数（除以 √d 防止梯度消失）
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)

# Step 2: 因果 mask（上三角 → -inf）
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))

# Step 3: softmax 归一化
weights = torch.softmax(scores, dim=-1)

# Step 4: 加权汇总 V
output = torch.matmul(weights, V)
```

`MultiHeadAttention.forward` 的切头操作：

```python
# [seq_len, d_model] → [num_heads, seq_len, d_head]
def split_heads(t):
    return t.view(seq_len, num_heads, d_head).transpose(0, 1)
```

---

## 下一步

至此，我们手写了注意力机制的核心计算，理解了 Q/K/V、缩放因子和因果掩码的原理。

但实际推理中面临新问题：**每次生成一个 token 时，K 和 V 都需要重新计算整个历史序列**，当序列很长时，这是极大的浪费。

下一步（Transformer Decoder 层：组装完整计算单元）将引入另外两个组件——**MLP（SwiGLU）** 和 **归一化（RMSNorm）**，并把它们与注意力一起组装成一个完整的 Transformer Decoder 层。
