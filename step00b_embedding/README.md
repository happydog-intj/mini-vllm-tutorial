# step00b — Embedding：向量空间

## 为什么需要 Embedding？

神经网络的计算单元（矩阵乘法、加法、激活函数）只能处理连续数值，而 token 本质上是整数 ID：`'A'` 是 65，`'B'` 是 66，`'你'` 可能是 15267。

直接把整数 ID 塞进网络会引发两个问题：

1. **数值大小毫无意义**：65 和 66 相差 1，并不意味着 'A' 和 'B' 在语义上更相近；而 'king'（假设 ID=1000）和 'queen'（假设 ID=1001）如果碰巧相邻，只是词表排序的巧合，与语义无关。

2. **无法学习语义关系**：网络需要一种表示方式，使得训练后"语义相近的词在向量空间中也相近"。整数 ID 做不到这一点。

解决方案：用一张可学习的查找表，把每个 token ID 映射到一个连续向量。这张表就是 **Embedding 矩阵**。

## 为什么不用 one-hot 编码？

one-hot 是最朴素的"把类别变成向量"方案：

```
词表大小 = 256（byte-level tokenizer）

one-hot('A') = [0, 0, ..., 1, ..., 0]
                             ↑ 第65位
               ← 共 256 维，只有1个1，其余全0 →
```

one-hot 有两个根本缺陷：

**问题1：维度灾难**

实际 LLM 的词表通常有 3~10 万个 token（GPT-2 是 50257，LLaMA 是 32000）。
用 one-hot 表示每个 token 就需要一个 5 万维的稀疏向量，绝大多数位置都是 0，计算和存储极度浪费。

**问题2：无法表示语义关系**

用欧氏距离衡量任意两个不同的 one-hot 向量：

```
‖one-hot(i) - one-hot(j)‖ = √2   （i ≠ j 时恒成立）
```

所有词对之间的距离完全相同。网络从这种表示里学不到"cat 和 dog 都是动物，而 cat 和 airplane 差异更大"这样的语义信息。

## Embedding：可学习的查找表

Embedding 矩阵的形状是 `[vocab_size, d_model]`：

```
                  d_model=8（训练中会学习到语义）
             ┌──────────────────────────────┐
token_id=0   │  0.23  -0.11   0.45  ...    │  ← '\x00' 的向量
token_id=1   │  0.87   0.45  -0.32  ...    │
    ...      │           ...               │
token_id=65  │  0.34  -1.21   0.87  ...    │  ← 'A' 的向量
token_id=66  │  0.12   0.93  -0.54  ...    │  ← 'B' 的向量
    ...      │           ...               │
token_id=255 │ -0.56   0.33   0.71  ...    │
             └──────────────────────────────┘
              ← vocab_size=256 行 →
```

**查表操作极其简单**：

```python
output = weight[token_id]   # 取矩阵的第 token_id 行
```

对于一个序列（多个 token），就是批量取多行：

```python
# token_ids: [72, 101, 108, 108, 111]  ('H','e','l','l','o')
output = weight[token_ids]  # 形状: [5, d_model]
```

## 实现细节

核心代码在 `embedding.py`：

```python
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # 初始化为标准正态分布，训练后学习语义信息
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, token_ids: Tensor) -> Tensor:
        # 输入: [seq_len]，值域 [0, vocab_size)
        # 输出: [seq_len, d_model]
        return self.weight[token_ids]   # 行索引，等价于 nn.Embedding
```

这与 PyTorch 的 `nn.Embedding` 等价（`nn.Embedding` 还做了输入校验和 padding_idx 支持，但核心操作相同）。

**初始化时**：权重是随机的，'A' 和 'B' 的向量没有任何有意义的关系。

**训练后**：梯度通过行索引反传到对应行，语义相近的词会被"拉近"。余弦相似度可以用来衡量两个向量的方向是否一致（取值 -1 到 1，越接近 1 越相似）。

注意：`run.py` 中演示的相似度是在**随机初始化**权重上算的，此时 `sim('A','B')` 是随机值，不代表语义——这是正常的，语义关系需要训练才能出现。

## 运行

```bash
python run.py
```

输出示例：

```
token_id=65 ('A') → 向量 shape: torch.Size([1, 8])
  向量值: [0.34, -1.21, 0.87, ...]   ← 随机初始化值，每次运行不同

'Hello' token IDs [72, 101, 108, 108, 111] → 矩阵 shape: torch.Size([5, 8])

余弦相似度('A','A') = 1.0000  （应为1.0）
余弦相似度('A','B') = 0.xxxx  （随机初始化，接近0）

✅ step00b_embedding 通过
```

## 下一步

现在每个 token 有了一个向量，但这些向量还不知道自己在序列中的**位置**。第 1 个 'l' 和第 2 个 'l'（在 "Hello" 中）会得到完全相同的向量——Transformer 无法区分它们。

下一步（step00c）将引入**位置编码（Positional Encoding）**，把序列位置信息注入向量，让模型知道"谁在前、谁在后"。
