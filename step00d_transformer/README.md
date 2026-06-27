# step00d — Transformer：完整 Decoder 层

## 教学目标

在 step00c 的注意力机制基础上，补全另外两个核心组件（MLP、归一化），并把三者组装成一个完整的 Transformer Decoder 层，理解每个设计决策背后的原因：

- 为什么用 Pre-Norm 而不是 Post-Norm？
- 残差连接解决什么问题？
- SwiGLU 比普通 ReLU MLP 好在哪里？
- RMSNorm 为什么逐渐取代 LayerNorm？

## 问题背景：为什么需要这些设计？

最早的 Transformer（2017 年 Attention Is All You Need）用的是 Post-Norm + ReLU MLP + LayerNorm，能工作，但训练深层网络时不稳定。现代 LLM（LLaMA、Qwen、Mistral 等）换成了 Pre-Norm + SwiGLU + RMSNorm 的组合，训练更稳定，效果也更好。这一步就是实现这套现代标准结构。

## Pre-Norm 结构（现代 LLM 标准）

完整 Decoder 层的数据流：

```
输入 x  [seq_len, d_model]
  │
  ├──→ RMSNorm ──→ MultiHeadAttention ──→ (+) ──→ x'
  │                                         ↑
  │                                    残差连接（加回原始 x）
  │
  ├──→ RMSNorm ──→ MLP (SwiGLU) ──────→ (+) ──→ x''
  │                                        ↑
  │                                   残差连接（加回 x'）
  │
输出 x'' [seq_len, d_model]
```

对应代码（`transformer.py` 中 `TransformerDecoderLayer.forward`）：

```python
x = x + self.attn(self.norm1(x))   # 注意力子层（Pre-Norm + 残差）
x = x + self.mlp(self.norm2(x))    # MLP 子层（Pre-Norm + 残差）
```

### Pre-Norm vs Post-Norm：为什么换了？

**Post-Norm（原版 Transformer）：**

```
x → Attention → (+x) → LayerNorm → 输出
```

**Pre-Norm（现代 LLM）：**

```
x → LayerNorm → Attention → (+x) → 输出
```

区别在于 Norm 的位置。Post-Norm 把归一化放在残差加法之后，意味着每一层输出都经过归一化，但梯度在反向传播时必须穿过 Norm 层才能到达前面的层，深层时梯度不稳定，需要仔细的学习率预热。

Pre-Norm 把归一化移到子层之前，残差路径（直连的那条）上没有 Norm，梯度可以直接流到前面的层，训练更稳定，对学习率不那么敏感。这是现代大模型能训练几十上百层的关键之一。

## 残差连接：梯度的高速公路

残差连接的形式是：

```python
x = x + F(norm(x))   # 不是 x = F(norm(x))
```

为什么需要这个"加回去"的操作？

**没有残差连接时的问题：**

```
输入 → 层1 → 层2 → 层3 → ... → 层N → 输出

反向传播时梯度要经过每一层的权重矩阵相乘。
如果某层的梯度 < 1，经过 N 层后梯度接近 0（梯度消失）。
如果某层的梯度 > 1，经过 N 层后梯度爆炸。
GPT-2 之前，超过几十层的 Transformer 很难稳定训练。
```

**有残差连接时：**

```
梯度 = d(loss)/dx = d(loss)/d(x + F(x)) 
     = d(loss)/d_output * (1 + dF/dx)

即使 dF/dx 接近 0（子层学到了恒等映射），梯度仍然可以
通过"1"这一项直接流过去，不会消失。
```

残差连接本质上给梯度开了一条高速公路，让它可以绕过子层直接到达更早的层。这使得训练数十甚至数百层的深层网络成为可能。

## MLP：SwiGLU 比 ReLU 好在哪里

### 普通 ReLU MLP（原版 Transformer）

```
x → W1 → ReLU → W2 → output

W1: [d_model, d_ff]
W2: [d_ff, d_model]
```

ReLU 在输入 < 0 时输出恒为 0，意味着网络中大量神经元在任意给定输入下是"死"的（输出为零），表达能力受限。

### SwiGLU（现代 LLM 标准）

```
          ┌─ W_gate ─→ SiLU(·) ─┐
x ──┤                              × ──→ W_down ──→ output
    └─ W_up ─────────────────────┘

W_gate: [d_model, d_ff]
W_up:   [d_model, d_ff]
W_down: [d_ff, d_model]
```

SwiGLU 有两条并行路径：`W_gate` 经过 SiLU 激活后作为门控（gate），与 `W_up` 路径的输出做逐元素相乘。SiLU（Sigmoid Linear Unit）是平滑的激活函数，没有 ReLU 的"硬截断"问题。

门控机制的直觉：`W_gate` 学会"哪些特征是重要的"，`W_up` 学会"特征的值是什么"，乘法把两者结合，网络可以动态地压制不相关的特征。

**代价**：比两矩阵的 ReLU MLP 多一个矩阵（`W_gate`），参数量和计算量约多 50%。实践中通常把 `d_ff` 缩小来补偿，总参数量接近不变。

本步骤实现（`d_model=128, d_ff=512`）：

```python
def forward(self, x):
    return self.W_down(self.act(self.W_gate(x)) * self.W_up(x))
    # act = SiLU
```

## RMSNorm：为什么比 LayerNorm 更常用

### LayerNorm

```
输入 x，形状 [seq_len, d_model]

mean = mean(x, dim=-1)          # 每个 token 的均值
var  = var(x, dim=-1)           # 每个 token 的方差
y    = (x - mean) / sqrt(var + ε) * γ + β

参数：γ（scale）、β（shift），各 d_model 个
```

LayerNorm 需要计算均值和方差两个统计量，并且有可学习的偏置 β。

### RMSNorm

```
输入 x，形状 [seq_len, d_model]

rms = sqrt(mean(x², dim=-1) + ε)   # 只算均方根
y   = x / rms * γ

参数：γ（scale），d_model 个；没有 β
```

RMSNorm 去掉了均值中心化（不减 mean）和偏置参数 β，只保留均方根归一化。

**为什么这样做有意义？**

原始 Transformer 引入 LayerNorm 是为了稳定训练，其中最重要的操作是缩放（除以标准差），而不是中心化（减均值）。实验表明，去掉均值中心化对模型效果影响很小，但计算更简单，在相同精度下计算速度更快。LLaMA、Qwen 等主流模型均使用 RMSNorm。

代码实现：

```python
def forward(self, x):
    rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
    return x / rms * self.weight
```

## 完整模型结构：TinyTransformer

本步骤还实现了一个完整的小型语言模型，演示多层堆叠的完整推理流程：

```
token_ids [seq_len]
    │
    ▼
Embedding                          [seq_len] → [seq_len, d_model=128]
    │
    ▼
TransformerDecoderLayer × 2        每层：Pre-Norm + Attention + Pre-Norm + MLP
    │
    ▼
RMSNorm（最终归一化）
    │
    ▼
LM Head（线性层）                  [seq_len, d_model] → [seq_len, vocab_size=256]
    │
    ▼
logits [seq_len, vocab_size]
```

参数量分布（d_model=128, num_heads=4, num_layers=2, vocab_size=256）：

```
Embedding:             vocab_size × d_model   = 256 × 128    = 32,768
每个 DecoderLayer:
  - Attention（Q/K/V/O）: 4 × d_model²       = 4 × 128²     = 65,536
  - MLP（gate/up/down）:  3 × d_model × d_ff = 3 × 128 × 512= 196,608
  - RMSNorm × 2:          2 × d_model        = 256
  合计 per layer: ≈ 262,400
× 2 层:                                                        524,800
最终 RMSNorm:           d_model              = 128
LM Head:                d_model × vocab_size = 128 × 256    = 32,768
─────────────────────────────────────────────────────────────────────
总参数量:                                                    ≈ 590,464
```

## 运行

```bash
python run.py
```

预期输出：

```
TinyTransformer: 2层, d_model=128, heads=4, vocab=256
参数量: 590,464  (~0.6M)
输入: torch.Size([10])  → 输出 logits: torch.Size([10, 256])

因果性验证：修改 token[-1] 后，前面位置的 logits 不变 ✅

✅ step00d_transformer 通过
```

因果性验证说明：修改序列最后一个 token，前面所有位置的 logits 不应该改变——这验证了因果注意力掩码正确工作，模型只能看到当前及之前的 token。

## 设计权衡总结

| 选择 | 现代做法 | 原版做法 | 改变的原因 |
|------|----------|----------|-----------|
| Norm 位置 | Pre-Norm | Post-Norm | 训练更稳定，梯度流更顺畅 |
| 激活函数 | SwiGLU（SiLU + 门控）| ReLU | 表达能力更强，避免"死神经元" |
| 归一化方式 | RMSNorm | LayerNorm | 计算更简单，效果相近 |
| 偏置 | 大多数线性层无偏置 | 有偏置 | 减少参数，训练更稳定 |

## 下一步

完整的 Transformer Decoder 层已经就绪。下一步（step01）将用这个结构搭建最朴素的自回归推理循环：每次生成一个 token，把它追加到序列末尾，再喂入模型预测下一个——这是理解所有后续优化的基准起点。
