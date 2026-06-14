# step01 — 朴素自回归推理

## 为什么从这里开始？

vLLM 解决的核心问题是：**大语言模型推理太慢**。但"太慢"是相对的——要理解为什么慢、慢在哪里，必须先看最朴素的实现。

这一步展示不做任何优化的基准实现。后续每一步优化都是在这个基础上解决某一个具体问题。

---

## 自回归的本质

大语言模型生成文本的数学基础是**概率链式分解**：

```
P(t1, t2, t3, ..., tN)
  = P(t1) × P(t2 | t1) × P(t3 | t1, t2) × ... × P(tN | t1...t(N-1))
```

每个 token 的生成概率，依赖于它之前所有 token 的历史。这个性质决定了模型**只能一个 token 一个 token 地顺序生成**，无法并行生成整个序列——这就是"自回归"的含义。

生成过程如下：

```
输入 prompt: [t1, t2, t3, t4, t5]
                        ↓
               model.forward(全部5个token)
                        ↓
              取最后位置的 logits → argmax → t6
                        ↓
输入: [t1, t2, t3, t4, t5, t6]
                        ↓
               model.forward(全部6个token)
                        ↓
              取最后位置的 logits → argmax → t7
                        ↓
              ... 循环直到生成结束 ...
```

注意：每一步都把**完整的历史序列**传入模型，包括之前已经计算过的 token。

---

## O(n²) 的具体来源

计算量随序列长度增长，原因在于 Transformer 中的**注意力机制（Attention）**。

在每一层，Attention 需要计算序列中每个位置的 Query 向量与所有位置的 Key 向量的相似度，然后用这些相似度加权求和 Value 向量。设当前序列长度为 n：

```
Q, K, V 矩阵各有 n 行（每行对应一个 token 的向量表示）

注意力分数矩阵:
        K0   K1   K2  ...  Kn
   Q0 [  ·    ·    ·  ...   · ]
   Q1 [  ·    ·    ·  ...   · ]
   Q2 [  ·    ·    ·  ...   · ]
   ...
   Qn [  ·    ·    ·  ...   · ]

矩阵大小: n × n
```

每步 Decode 需要计算这个 n×n 的矩阵，而且要从头重新计算所有 token 的 K 和 V。

朴素实现中，生成 m 个新 token 的总计算量：

```
Step 1:  model(5 tokens)   → 计算 5×5  的注意力矩阵，重算 5  个 K/V
Step 2:  model(6 tokens)   → 计算 6×6  的注意力矩阵，重算 6  个 K/V  (Step1的5个被重算)
Step 3:  model(7 tokens)   → 计算 7×7  的注意力矩阵，重算 7  个 K/V  (Step1,2的全被重算)
...
Step m:  model((5+m) tokens)

Attention 计算量之和:
  5² + 6² + 7² + ... + (5+m)²  ≈  O(m³)  （严格来说是 O(n·m²)，n 为 prompt 长度）

K/V 重复计算量之和:
  5 + 6 + 7 + ... + (5+m)  ≈  O(m²)
```

简而言之：**每生成一个新 token，就要把之前所有 token 的 K/V 从头计算一遍**。已经做过的计算被反复抛弃重做。

---

## 核心代码

`engine.py` 的 `decode_one_step` 展示了问题所在：

```python
@torch.no_grad()
def decode_one_step(self, input_ids: Tensor) -> Tensor:
    # 全量前向：把当前完整序列传入模型
    # 模型内部会重新计算所有 token 的 Q/K/V
    logits = self.model(input_ids)  # [seq_len, vocab_size]

    # 只取最后一个位置的 logits（预测下一个 token）
    last_logits = logits[-1]        # [vocab_size]

    # Greedy 采样：选概率最大的 token
    next_id = torch.argmax(last_logits)
    return next_id
```

每次调用，`input_ids` 比上次多一个 token，但模型对前面所有 token 的计算都是重复的。

`generate` 方法将这个循环串联起来：

```python
def generate(self, prompt_ids: Tensor, max_new_tokens: int) -> Tensor:
    input_ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        next_id = self.decode_one_step(input_ids)          # 全量前向
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)])  # 追加新 token
    return input_ids
```

逻辑简单清晰，但计算浪费是结构性的。

---

## Prefill 与 Decode 的隐含区别

在朴素实现中，两个阶段混在一起，但值得提前认识：

```
Prefill 阶段（处理 prompt）:
  输入: [t1, t2, t3, t4, t5]  （所有 prompt token 并行计算）
  特点: 一次前向，计算量大但 GPU 利用充分

Decode 阶段（生成新 token）:
  Step 1: 新增 t6，但要带上 [t1..t5] 重算
  Step 2: 新增 t7，但要带上 [t1..t6] 重算
  特点: 每步只产出 1 个 token，但计算量随步数增加
        GPU 每步计算量较少，大量算力被用于重复计算
```

朴素实现没有区分这两个阶段——每步都把全部 token 传入，Prefill 的工作在 Decode 的每一步都被重做。step03a 会专门解决这个问题。

---

## 运行

```bash
python run.py
```

输出示例（具体时间因硬件不同而异）：

```
==================================================
朴素自回归推理 — 速度随序列长度下降
==================================================
  Step   5: x.xms/token | 序列总长:  10 | 重算 KV 次数: 10
  Step  10: x.xms/token | 序列总长:  15 | 重算 KV 次数: 15
  Step  50: x.xms/token | 序列总长:  55 | 重算 KV 次数: 55
  Step 100: x.xms/token | 序列总长: 105 | 重算 KV 次数: 105
  Step 150: x.xms/token | 序列总长: 155 | 重算 KV 次数: 155
  Step 200: x.xms/token | 序列总长: 205 | 重算 KV 次数: 205

前20步平均: x.xms  后20步平均: x.xms
→ 速度随序列长度线性下降 ⚠️

✅ step01_naive 通过
```

运行结束时会断言后 20 步的平均时间明显大于前 20 步，验证"越来越慢"的现象。

---

## 这个实现的价值

朴素实现有两个作用：

1. **正确性基准**：后续每步优化后，生成结果应与朴素实现完全一致（相同随机种子下）。如果不一致，说明优化引入了 bug。

2. **性能基准**：每步优化后，运行时间对比朴素实现应该更短，且随序列长度的增长应该更平缓。

---

## 下一步

**step02** 解决采样策略问题：`argmax`（贪心解码）每次选概率最高的 token，生成结果单调、缺乏多样性。Temperature 采样、Top-k、Top-p 等策略控制如何从概率分布中抽取 token，影响生成质量。

**step03a** 解决 O(n²) 的根本问题：引入 KV Cache，把每个 token 的 K/V 向量缓存起来，Decode 阶段每步只计算新增 token 的 K/V，消除重复计算。
