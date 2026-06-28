# Step 09: Real Model — Qwen3ForCausalLM

## 本节目标

前几步我们用的是 TinyTransformer——一个手写的、参数量只有几千的玩具模型。
本节接入 **Qwen3-0.6B**，一个真实的、可下载的开源语言模型。

这一步要解决的问题是：**如何让一个从零实现的推理引擎，直接加载并运行 HuggingFace 格式的权重？**

完成本节后，你能用自己实现的引擎跑出真实的文本生成结果。

---

## 从 TinyTransformer 到真实模型，差在哪里？

不是代码复杂度，是几个具体的工程差距：

| 维度 | TinyTransformer | Qwen3-0.6B |
|------|-----------------|------------|
| 参数量 | ~几千 | 6 亿（约 1.2 GB BF16） |
| 词表大小 | 自定义（小） | 151,936 |
| 位置编码 | 绝对位置编码或无 | RoPE（旋转位置编码） |
| 注意力 | MHA（多头注意力） | GQA（分组查询注意力） |
| 归一化 | LayerNorm | RMSNorm（去掉均值中心化） |
| 权重格式 | PyTorch `.pt` | safetensors 分片格式 |
| Tokenizer | 无或简单字符级 | Byte-Pair Encoding，需要 transformers 库 |

每一行都代表一个需要专门处理的工程问题。下面逐一说明。

---

## safetensors 格式：为什么不用 pickle？

HuggingFace 的旧格式是 `.bin`（本质是 `torch.save`，基于 Python pickle）。
现在的模型几乎都改用 **safetensors**，原因有三：

**1. 安全性**
pickle 在反序列化时可以执行任意 Python 代码。下载一个陌生模型，加载 `.bin` 文件，理论上可以被植入恶意代码。safetensors 只存储张量数据，没有可执行逻辑。

**2. 支持 lazy loading（按需加载）**
safetensors 在文件头部存储了所有张量的名称、形状、数据类型和偏移量。这意味着可以只读取文件头，然后按名称精确定位并加载某个张量，而不必把整个文件读入内存。对于多 GPU 分布式推理，这个特性非常重要（不同 GPU 各自加载自己负责的分片）。

**3. 支持模型分片**
大模型（几十 GB）不能放进单个文件。safetensors 用一个 `model.safetensors.index.json` 记录哪个张量在哪个分片文件里，加载器按需读取。

```
model.safetensors.index.json          ← 索引：name → shard 文件名
model-00001-of-00004.safetensors      ← 分片 1
model-00002-of-00004.safetensors      ← 分片 2
...
```

Qwen3-0.6B 比较小，使用单文件 `model.safetensors`。`loader.py` 两种格式都支持：

```python
index_file = model_path / "model.safetensors.index.json"
if index_file.exists():
    # 大模型：读索引，按序加载每个分片
    ...
else:
    # 小模型：直接加载单文件
    single = model_path / "model.safetensors"
```

还有一个细节：HuggingFace 的权重 key 全部带 `"model."` 前缀（如 `model.layers.0.self_attn.q_proj.weight`），而我们的 `Qwen3ForCausalLM` 的 state_dict 里没有这个前缀。`load_weights` 里用 `name[len("model."):]` 统一去掉。

---

## BF16 精度：为什么不用 FP16 或 FP32？

推理时需要选一个浮点精度。三个选项的权衡：

```
FP32: 符号1位 + 指数8位 + 尾数23位   → 精度高，但占用是 BF16 的 2 倍
FP16: 符号1位 + 指数5位 + 尾数10位   → 指数位少，表示范围小（最大约 65504）
BF16: 符号1位 + 指数8位 + 尾数7位    → 指数位与 FP32 一样多，尾数精度较低
```

FP16 的问题是**数值溢出**：大型语言模型中 logits（输出层的未归一化分数）经常超过 65504，直接变成 `inf`，生成结果乱掉。

BF16 的指数范围与 FP32 相同，不会溢出。代价是精度（尾数只有 7 位），但推理时的精度损失通常在可接受范围内——模型在 BF16 下训练，推理也用 BF16，分布一致。

代码中的体现：

```python
self.model = Qwen3ForCausalLM(config).to(torch.bfloat16).to(self.device)
# ...
state_dict[name] = tensor.to(torch.bfloat16)   # 权重加载时也转换
```

注意：BF16 需要硬件支持（NVIDIA A100/H100/RTX 30xx 以上，或 Apple Silicon）。旧显卡（如 V100、GTX 10xx）不支持原生 BF16 运算，会自动回退到 FP32 计算，速度下降但结果正确。

---

## RMSNorm：去掉均值中心化

Transformer 标准用 LayerNorm：

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

Qwen3 和很多现代 LLM（LLaMA、Mistral 等）改用 **RMSNorm**：

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

去掉了两样东西：
- 均值中心化（`x - mean(x)`）
- bias 参数

动机：实验发现去掉均值中心化对模型质量影响很小，但计算量更少，且没有 bias 意味着参数更少。

```python
def forward(self, x: Tensor) -> Tensor:
    rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
    return x / rms * self.weight
```

Qwen3 在注意力模块内也对每个 Q 和 K 的 head 单独做了 RMSNorm（`q_norm`、`k_norm`），这是 Qwen3 相比 LLaMA 的一处改动，有助于训练稳定性。

---

## GQA：KV Cache 太大怎么办？

先理解问题来源。

**标准 MHA（多头注意力）** 中，每个 attention head 都有独立的 K 和 V。假设有 32 个 head，每个 head 维度 128，序列长度 4096，BF16 精度，则单层 KV Cache 大小约为：

```
单层 KV = 2（K+V）× 序列长度 × head数 × head维度 × 2字节（BF16）
        = 2 × 4096 × 32 × 128 × 2 = 67 MB
28层模型合计 ≈ 1.8 GB
```

随着序列变长，KV Cache 线性增长，很快就会成为显存瓶颈。

**GQA（Grouped Query Attention）** 的解决思路：多个 Q head 共享同一组 K/V head。

```
MHA:  Q[32 heads] → K[32 heads], V[32 heads]
GQA:  Q[32 heads] → K[ 8 heads], V[ 8 heads]  (4 个 Q head 共享 1 个 KV head)
MQA:  Q[32 heads] → K[ 1 head],  V[ 1 head]   (极端情况，所有 Q 共享)
```

Qwen3-0.6B 的配置（`config.json` 中）：

```
num_attention_heads    = 16   ← Q head 数
num_key_value_heads    = 8    ← KV head 数
num_groups             = 16 / 8 = 2
```

推理时，用 `repeat_interleave` 把 KV 展开，使维度与 Q 对齐：

```python
k_exp = k.repeat_interleave(self.num_groups, dim=1)
v_exp = v.repeat_interleave(self.num_groups, dim=1)
```

```
原始 K:  [seq_len, 8, head_dim]
展开后:  [seq_len, 16, head_dim]   ← 每个 KV head 复制 2 份
```

展开在计算上等价于 MHA，但 **存储的 KV Cache 只有 MHA 的 1/2**（按 KV head 数量比例）。

代价是：Q 和 K/V 来自不同的表示空间，理论上表达能力略有下降。但实践中，GQA 在参数量相近的情况下，质量损失很小，而 KV Cache 节省显著，已成为现代 LLM 的标准配置。

---

## RoPE：位置编码的泛化问题

绝对位置编码（如原始 Transformer 的 sinusoidal 编码或可学习编码）的问题：训练时见过的最大序列长度是上限，超出这个长度，位置编码没有对应的表示，模型行为不可预测。

**RoPE（旋转位置编码）** 的核心思想：不直接编码"这个 token 在位置 N"，而是编码"这两个 token 的相对距离"。通过对 Q 和 K 施加旋转变换，使得点积 `QᵀK` 自然包含相对位置信息：

```
位置 m 的 Q：q_m = rotate(q, m * θ)
位置 n 的 K：k_n = rotate(k, n * θ)

点积 q_m · k_n  只依赖于 (m - n)，即相对距离
```

实现上，将每个 head 的向量拆成前半和后半，做"复数乘法"：

```python
x1 = x[..., :head_dim // 2]   # 前半
x2 = x[..., head_dim // 2:]   # 后半
rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

其中 `cos` 和 `sin` 是按位置预计算的旋转频率，`theta=1000000.0` 是 Qwen3 使用的基础频率（比原始 RoPE 的 10000 大得多，有助于编码更长的序列）。

```python
freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
positions = torch.arange(max_seq_len)
freqs_matrix = torch.outer(positions, freqs)  # [seq_len, head_dim/2]
cos, sin = torch.cos(freqs_matrix), torch.sin(freqs_matrix)
```

RoPE 相对绝对编码的优势：对训练时未见过的更长序列，仍有合理的外推能力（各种 RoPE 变体如 YaRN、LongRoPE 进一步增强了这一点）。

---

## 推理流程全貌

```
输入 prompt: "你好，请介绍一下你自己。"
         │
         ▼
   [Tokenizer.encode]
   → token ids: [108386, 3837, ...]   (151936 词表)
         │
         ▼
   ┌─────────────────────────────────────────────┐
   │           Prefill（一次并行处理全部 token）        │
   │                                             │
   │  embed_tokens → [seq_len, 1024]             │
   │         ↓                                   │
   │  Layer 0..27:                               │
   │    RMSNorm → GQA(RoPE) → RMSNorm → MLP     │
   │    每层输出 (K, V) → 存入 past_key_values    │
   │         ↓                                   │
   │  final RMSNorm → lm_head                    │
   │  → logits[-1]: 下一个 token 的概率分布        │
   └─────────────────────────────────────────────┘
         │
         ▼ argmax → token id → 新 token
         │
   ┌─────────────────────────────────────────────┐
   │           Decode（逐步，每步 1 个 token）        │
   │                                             │
   │  input: [1]  positions: [prompt_len + step] │
   │  past_key_values: 上一步的 KV Cache          │
   │         ↓                                   │
   │  embed → 28层(复用 KV Cache) → lm_head       │
   │  → 下一个 token                              │
   └─────────────────────────────────────────────┘
         │
   重复直到 eos_token 或达到 max_new_tokens
         │
         ▼
   [Tokenizer.decode] → 输出文本
```

Prefill 和 Decode 的关键区别：

- **Prefill**：`is_causal=True`，用因果掩码，并行处理整个输入序列，`past_kv=None`
- **Decode**：每步只输入 1 个 token，`past_kv` 里有之前所有 token 的 K/V，`is_causal=False`（单 token 不需要掩码）

---

## 文件说明

| 文件 | 功能 |
|------|------|
| `model.py` | Qwen3ForCausalLM 完整实现（RMSNorm、RoPE、GQA、MLP、解码层） |
| `loader.py` | safetensors 权重加载器，支持单文件和分片两种格式 |
| `engine.py` | 封装模型 + Tokenizer，实现 prefill/decode 两阶段推理 |
| `run.py` | 测试入口，无模型时自动跳过 |

---

## 运行

```bash
# 下载模型（需要 huggingface-cli 或 modelscope）
# huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B

# 设置模型路径（默认 ~/huggingface/Qwen3-0.6B）
export QWEN3_MODEL_PATH=/path/to/Qwen3-0.6B

# 安装依赖
pip install safetensors transformers

python run.py
```

有模型时的输出示例：

```
模型路径: /path/to/Qwen3-0.6B
  使用设备: cuda
  初始化模型结构...
  加载权重...
  权重加载完成，338 个参数张量
  加载完成 ✅

Prompt: '你好，请介绍一下你自己。'
Output: '我是Qwen，由阿里云开发的AI助手...'

  速度: xxx tok/s

✅ step15_real_model 通过
```

没有模型时：

```
⚠️  未找到模型：/path/to/Qwen3-0.6B
✅ step15_real_model 通过（跳过推理测试，未找到模型）
```

---

## 硬件说明

- **CUDA GPU（推荐）**：BF16 在 Ampere 架构（A100、RTX 3xxx）以上原生加速；Volta（V100）不支持 BF16，但代码会正常运行（回退 FP32 计算）
- **Apple Silicon（MPS）**：支持 BF16，代码自动选择 MPS 设备
- **CPU**：可以运行，但 0.6B 模型在 CPU 上速度较慢，不适合交互式测试

BF16 对显存的要求：0.6B 参数 × 2 字节 ≈ 1.2 GB，加上 KV Cache，总显存需求在 2 GB 以内，集成显卡也能运行。

---

## 下一步

本节的引擎每次只处理一个请求，注意力计算也是标准实现。下一步（FlashAttention：SRAM-aware 注意力计算）将引入 **FlashAttention**：通过分块计算消除标准注意力的内存带宽瓶颈，同时支持变长序列（varlen）接口，为后续 Continuous Batching 做好基础。
