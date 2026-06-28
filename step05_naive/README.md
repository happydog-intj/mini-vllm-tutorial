# step05 — 朴素自回归推理

## 为什么叫“自回归”？
因为每个新输出回归（依赖）于自己之前生成的结果 —— 就像写一句话：写了“I”才能写“love”，写了“love”才能写“China”。一旦中间某步错了（比如写了“like”），后续都会跟着跑偏，这就是自回归的误差累积特点。

## 为什么从自回归开始介绍？

vLLM 解决的核心问题是：**大语言模型推理太慢**。但"太慢"是相对的——要理解为什么慢、慢在哪里，必须先看最朴素的实现。

这一步展示不做任何优化的基准实现。后续每一步优化都是在这个基础上解决某一个具体问题。

---

## 自回归的本质

大语言模型生成文本的数学基础是**概率链式分解**：

```
P(t1, t2, t3, ..., tN)
  = P(t1) × P(t2 | t1) × P(t3 | t1, t2) × ... × P(tN | t1...t(N-1))
```

## 通俗的解释就是**文字接龙**：

我们用“机器翻译”（中文→英文）来举例，模型要一步步把“我爱中国”译成“I love China”。

自回归生成过程（每一步）
初始输入：<start>（起始标记）

第1步：模型根据<start>预测第一个英文单词 → 输出概率最高的是 I

第2步：输入变为 <start> I，预测下一个 → 输出 love

第3步：输入变为 <start> I love，预测下一个 → 输出 China

第4步：输入变为 <start> I love China，预测下一个 → 输出 <end>（结束标记），生成停止。

每一步的输入都包含之前所有已生成的token，但绝不包含未来的正确单词（比如预测第2步时，不能提前看到love或China）。

## 我们可以用 GPT 模型（比如 ChatGPT 背后的生成式预训练 Transformer）来举例，展示它的自回归机制如何工作。

场景：你输入一段话的开头，让 GPT 继续写下去
假设你给 GPT 的 prompt 是：
“今天天气真好，我决定”

GPT 会一个字一个字地（更准确说是 token 一个接一个地）往下生成，每一步都依赖已经写出的内容。

自回归生成过程（简化成 token 序列）
| 步骤 | 当前输入（已生成的 token 序列）                       | GPT 预测的下一个 token |
|------|------------------------------------------------|------------------|
| 1    | `[今天] [天气] [真好] [，] [我] [决定]`            | `去`             |
| 2    | `... [决定] [去]`                               | `公园`           |
| 3    | `... [去] [公园]`                               | `散步`           |
| 4    | `... [公园] [散步]`                             | `。`             |
| 5    | `... [散步] [。]`                               | `<eos>`（结束）   |

最终输出：“今天天气真好，我决定去公园散步。”

每一步，GPT 都会：

把到目前为止所有已生成的 token 作为输入（包括 prompt 原始 token）。

通过 Transformer 解码器中的因果掩码（causal mask），保证在计算每个位置的注意力时，只能看到它左边的 token，不能“偷看”右边还未生成的词。

从词汇表的概率分布中采样/选择下一个 token。

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

```
生成的开始 ──▶ input_ids = [101, 2054, 2003]    # 初始 prompt
                      ↓
        ┌─────────────────────────────────────────────┐
        │          自回归生成循环 (generate)           │
        │  ┌─────────────────────────────────────┐    │
        │  │  模型前向 (model.forward)            │    │
        │  │          ↓                          │    │
        │  │  因果注意力 (CausalAttention)        │    │
        │  │  • 创建上三角掩码                    │    │
        │  │  • masked_fill_ 屏蔽未来位置         │    │
        │  │  • Softmax 只关注历史 token         │    │
        │  │          ↓                          │    │
        │  │  输出 logits → softmax → 采样       │    │
        │  └─────────────────────────────────────┘    │
        └─────────────────────────────────────────────┘
                      ↓
               预测下一个 token
                      ↓
          input_ids.append(next_token)   # 添加到输入序列
                      ↓
               (循环，直到生成为止)
                      ↓
                最终生成完成
```

对应到 `engine.py` 的实际代码，每一步的操作就是：

```python
# 初始输入：prompt 的 token_ids，比如 [72, 101, 108, 108, 111]（"Hello"）
input_ids = prompt_ids.clone()   # tensor([72, 101, 108, 108, 111])

for step in range(max_new_tokens):

    # ── 关键操作：把完整的历史序列整个传入模型 ──
    logits = self.model(input_ids)
    # logits 形状: [当前序列长度, vocab_size]
    # 例如第一步: [5, 256]，第二步: [6, 256]，...
    # 注意：模型内部对 input_ids[0]、input_ids[1]、... 的 K/V 全部重新算了一遍！

    # 只取最后一个位置的 logits（该位置预测"下一个 token"）
    last_logits = logits[-1]                         # [vocab_size]，如 [256]

    # 选概率最大的 token（贪心采样）
    next_id = torch.argmax(last_logits)              # 标量，如 tensor(119)

    # 把新 token 追加到序列末尾，下一步作为输入
    input_ids = torch.cat([input_ids, next_id.unsqueeze(0)])
    # 第一步后: [72, 101, 108, 108, 111, 119]  ← 长度从5变成6
    # 第二步后: [72, 101, 108, 108, 111, 119, ?]  ← 再追加一个

# 最终 input_ids = [prompt tokens] + [生成的 tokens]
```

可以看到每一步 `self.model(input_ids)` 的 `input_ids` 都比上一步多一个 token，
但模型对前面所有 token 的 K/V 计算是从零开始重做的——
`input_ids[0]` 到 `input_ids[-2]` 的 K/V 在上一步已经算过，这一步白白重算了一遍。

注意：每一步都把**完整的历史序列**传入模型，包括之前已经计算过的 token。

---

## self.model(input_ids) 内部：QKV 计算详解

上面的循环每次调用 `self.model(input_ids)`，模型内部究竟做了什么？
以序列长度 `n=6`（5 个 prompt token + 1 个已生成 token）为例，逐层展开。

### 第一步：Embedding 查表

```python
# TinyTransformer.forward() 第一行：
x = self.embed(token_ids)   # [6] → [6, 128]
# token_ids = [72, 101, 108, 108, 111, 119]  ← 6个token ID
# x[i] = embed.weight[token_ids[i]]          ← 查表，每个ID变成128维向量
# x 形状: [6, 128]   seq_len=6，d_model=128
```

### 第二步：经过每个 TransformerDecoderLayer

TinyTransformer 有 2 层，每层做同样的事：

```python
# TransformerDecoderLayer.forward():
x = x + self.attn(self.norm1(x))   # 注意力子层（含残差）
x = x + self.mlp(self.norm2(x))    # MLP 子层（含残差）
```

展开注意力子层 `self.attn(self.norm1(x))`：

```python
# MultiHeadAttention.forward(x):   x 形状 [6, 128]
seq_len = 6

# 1. 三个线性投影：把 128 维向量分别投影为 Q、K、V
Q = self.W_q(x)   # [6, 128] @ [128, 128] → [6, 128]
K = self.W_k(x)   # [6, 128] @ [128, 128] → [6, 128]
V = self.W_v(x)   # [6, 128] @ [128, 128] → [6, 128]
# W_q、W_k、W_v 是可学习的权重矩阵，形状都是 [128, 128]

# 2. 切分成多头（num_heads=4，每头 d_head=32）
# [6, 128] → [6, 4, 32] → [4, 6, 32]
Q = Q.view(6, 4, 32).transpose(0, 1)   # [4, 6, 32]
K = K.view(6, 4, 32).transpose(0, 1)   # [4, 6, 32]
V = V.view(6, 4, 32).transpose(0, 1)   # [4, 6, 32]
# 现在每个头有自己的 [6, 32] 的 Q/K/V

# 3. 每个头独立做 Scaled Dot-Product Attention
for h in range(4):
    q_h = Q[h]   # [6, 32]  ← 头 h 的查询矩阵
    k_h = K[h]   # [6, 32]  ← 头 h 的键矩阵
    v_h = V[h]   # [6, 32]  ← 头 h 的值矩阵

    # 计算注意力分数：每个位置的 Q 与所有位置的 K 做点积
    scores = q_h @ k_h.T / sqrt(32)   # [6, 32] @ [32, 6] → [6, 6]
    # scores[i, j] = Q[i] · K[j] / √32
    # 表示位置 i 对位置 j 的"注意力程度"

    # 因果掩码：token i 不能看到 token j > i（未来的 token）
    # 把上三角（j > i 的位置）置为 -inf，softmax 后变为 0
    mask = [[0, -inf, -inf, -inf, -inf, -inf],
            [0,    0, -inf, -inf, -inf, -inf],
            [0,    0,    0, -inf, -inf, -inf],
            [0,    0,    0,    0, -inf, -inf],
            [0,    0,    0,    0,    0, -inf],
            [0,    0,    0,    0,    0,    0]]
    scores = scores + mask   # [6, 6]

    # Softmax：每行归一化，得到注意力权重
    weights = softmax(scores, dim=-1)   # [6, 6]，每行和为 1
    # weights[i, j] = token i 分配给 token j 的注意力权重

    # 用权重加权求和 V：
    out_h = weights @ v_h   # [6, 6] @ [6, 32] → [6, 32]
    # out_h[i] = Σ_j weights[i,j] * V[j]
    # = 用注意力权重把所有位置的 V 加权混合

# 4. 拼接所有头的输出
concat = cat([out_0, out_1, out_2, out_3], dim=-1)   # [6, 128]

# 5. 输出投影
output = self.W_o(concat)   # [6, 128] @ [128, 128] → [6, 128]
```

### 第三步：MLP 子层

```python
# MLP.forward(x):   x 形状 [6, 128]
gate = SiLU(self.W_gate(x))   # [6, 128] → [6, 512]
up   = self.W_up(x)           # [6, 128] → [6, 512]
output = self.W_down(gate * up)  # [6, 512] → [6, 128]
# d_ff = 128 × 4 = 512，先升维再降维
```

### 第四步：LM Head 输出 logits

经过 2 层 TransformerDecoderLayer 和最终的 RMSNorm 后：

```python
x = self.norm(x)           # [6, 128] → [6, 128]
logits = self.lm_head(x)   # [6, 128] @ [128, 256] → [6, 256]
# logits[i] = 位置 i 处对词表（256个token）的打分
# 我们只用 logits[-1]（最后一个位置），预测下一个 token
```

### 关键总结：哪里在重复计算？

```
第 k 步 decode（序列长度 n = prompt_len + k）：

  token 0 → Embedding → Q0,K0,V0 ← 第 k-1 步已经算过了，这步重算！
  token 1 → Embedding → Q1,K1,V1 ← 第 k-1 步已经算过了，这步重算！
  ...
  token n-2 → Embedding → Q(n-2),K(n-2),V(n-2) ← 上步算过，重算！
  token n-1 → Embedding → Q(n-1),K(n-1),V(n-1) ← 新 token，第一次算

  scores = Q @ K^T   ← [n, n] 矩阵，包含了所有历史 token 的重新计算
```

每步新增 1 个 token，却要重算所有 n 个 token 的 K/V。
K/V 不会因为后续 token 的存在而改变（K_i = f(token_i)），
所以这种重算是纯粹的浪费——**这正是 step07 KV Cache 要解决的问题**。

---



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

朴素实现没有区分这两个阶段——每步都把全部 token 传入，Prefill 的工作在 Decode 的每一步都被重做。step07 会专门解决这个问题。

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

✅ step05_naive 通过
```

运行结束时会断言后 20 步的平均时间明显大于前 20 步，验证"越来越慢"的现象。

---

## 这个实现的价值

朴素实现有两个作用：

1. **正确性基准**：后续每步优化后，生成结果应与朴素实现完全一致（相同随机种子下）。如果不一致，说明优化引入了 bug。

2. **性能基准**：每步优化后，运行时间对比朴素实现应该更短，且随序列长度的增长应该更平缓。

---

## 下一步

**step06** 解决采样策略问题：`argmax`（贪心解码）每次选概率最高的 token，生成结果单调、缺乏多样性。Temperature 采样、Top-k、Top-p 等策略控制如何从概率分布中抽取 token，影响生成质量。

**step07** 解决 O(n²) 的根本问题：引入 KV Cache，把每个 token 的 K/V 向量缓存起来，Decode 阶段每步只计算新增 token 的 K/V，消除重复计算。
