# Step 10: FlashAttention 封装

## 教学目标

理解标准注意力的内存带宽瓶颈，以及 FlashAttention 如何通过分块计算消除它。
同时，理解变长序列（varlen）接口——这是 Continuous Batching 下高效处理混合长度请求的基础。

## 为什么需要 FlashAttention？

先回顾注意力计算的公式：

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

这个公式看起来简洁，但朴素实现有一个严重的问题：**`QK^T` 的结果必须完整写回显存**。

```
标准注意力的内存访问（seq_len = 2048，head_dim = 128）：

  步骤1: 从显存读 Q、K  → 计算 QK^T
  步骤2: QK^T 矩阵 [2048, 2048] 写回显存   ← 每层约 32MB（float16）
  步骤3: 从显存读 QK^T  → 计算 softmax
  步骤4: softmax 结果写回显存              ← 再写 32MB
  步骤5: 从显存读 softmax 结果、V → 计算输出

  每层的 HBM（显存）读写量：O(seq_len²)
  seq_len 翻倍 → 读写量翻 4 倍
```

这不是算力（FLOPS）的瓶颈，而是**显存带宽（memory bandwidth）的瓶颈**。
GPU 的算术单元在等数据，而不是在计算。

## GPU 内存层次结构：为什么分块有效

GPU 有两级内存：

```
HBM（High Bandwidth Memory，就是"显存"）
├── A100: 容量 80GB，带宽约 2TB/s
└── 存放所有模型权重、KV Cache、中间激活

SRAM（片上缓存，即 GPU 的 Shared Memory）
├── A100: 每个 SM（流多处理器）约 192KB
└── 带宽约 19TB/s  ← 比 HBM 快约 10×，但容量极小
```

FlashAttention 的核心思路：**把 Q/K/V 切成小块，每块装进 SRAM，
在 SRAM 内部完成所有计算，只把最终结果写回 HBM。**

```
FlashAttention 分块计算示意：

  HBM                         SRAM（片上缓存）
  ┌─────────────────────┐     ┌───────────────┐
  │ Q: [N, d]           │     │               │
  │ K: [N, d]     加载→ │────→│ Q_i: [Br, d]  │
  │ V: [N, d]    一小块 │     │ K_j: [Bc, d]  │
  │                     │     │ V_j: [Bc, d]  │
  │ Output: [N, d]      │     │               │
  └─────────────────────┘     │ 在片上完成：   │
           ↑                  │  S_ij = Q_i·Kⱼᵀ│
           │ 只写最终输出      │  softmax       │
           └──────────────────│  O_i += softmax│
                              │        × V_j   │
                              └───────────────┘

  关键：QK^T 的中间结果不写回 HBM
  HBM 读写量：O(N)  而非  O(N²)
```

**在线 softmax（online softmax）** 是让分块等价的数学技巧：
softmax 需要知道一行的最大值才能稳定计算，但分块时看不到整行。
FlashAttention 通过维护一个滚动的最大值和归一化因子，
确保分块计算的最终结果与完整矩阵计算**数值等价**。

## 标准注意力 vs FlashAttention

```
标准注意力（PyTorch 朴素实现）：

  Q, K, V ──→ QK^T ──(写HBM)──→ softmax ──(写HBM)──→ ×V ──→ Output
                  ↑                    ↑
               大量显存读写          大量显存读写

FlashAttention（分块 + 片上计算）：

  Q, K, V ──→ [分成小块，在SRAM内循环] ──→ Output
                  ↑
               只有最终结果写显存

  内存复杂度：O(N²) → O(N)
  速度：受益于显存读写减少，序列越长收益越大
```

## 变长序列（varlen）接口：Continuous Batching 的实际需求

Continuous Batching 调度器 提到，Continuous Batching 下同一 batch 内不同请求的序列长度各不相同：

```
某一时刻的 batch：

  请求A：正在 decode，已生成 47 个 token
          → 注意力计算：Q=[1, d]  与 KV=[47, d] 做注意力

  请求B：正在 decode，已生成 312 个 token
          → 注意力计算：Q=[1, d]  与 KV=[312, d] 做注意力

  请求C：刚进来做 prefill，prompt = 128 个 token
          → 注意力计算：Q=[128, d] 与 KV=[128, d] 做注意力

  三个请求的形状完全不同，无法直接拼成一个矩阵。
```

**传统做法：padding 补齐**

```
把所有序列补到最长那个的长度：

  [A的47个token + 265个PAD][B的312个token][C的128个token + 184个PAD]
  形状: [3, 312, head_dim]

  问题：PAD 越多，计算越浪费；
        且 B（长序列）决定了整个 tensor 的形状
```

**FlashAttention varlen 做法：拼接 + cu_seqlens**

```
直接把所有序列拼成一维：

  [A的47个token | B的312个token | C的128个token]
  形状: [487, head_dim]   ← 无 PAD，零浪费

  cu_seqlens: [0, 47, 359, 487]
              ↑  ↑   ↑    ↑
              A起 A止 B止  C止
              （cumulative sequence lengths，累积序列长度）

  GPU kernel 内部：
    for i in range(batch_size):
        start = cu_seqlens[i]    # = 0, 47, 359
        end   = cu_seqlens[i+1]  # = 47, 359, 487
        处理第 i 个序列的 token [start:end]
```

### cu_seqlens 的具体含义

`cu_seqlens` 是累积序列长度数组（cumulative sequence lengths），长度为 `batch_size + 1`：

```python
# 示例：3个序列，长度分别为 47, 312, 128
seqlens = [47, 312, 128]
cu_seqlens = torch.tensor([0, 47, 359, 487], dtype=torch.int32)
#                                ^    ^    ^
#                     0+47=47  47+312=359  359+128=487

# 调用 flash_attn_varlen_func
from flash_attn import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q,              # [total_tokens, num_heads, head_dim]  ← 三个序列拼在一起
    k,              # [total_tokens, num_kv_heads, head_dim]
    v,              # [total_tokens, num_kv_heads, head_dim]
    cu_seqlens_q,   # [batch_size + 1]
    cu_seqlens_k,   # [batch_size + 1]
    max_seqlen_q,   # 最长序列的长度（用于 kernel 分块尺寸决策）
    max_seqlen_k,
    causal=True,
)
# output: [total_tokens, num_heads, head_dim]
```

### varlen 是接口约定，不是硬件要求

这里有一个容易混淆的地方：

**`flash_attn_varlen_func` 要求的拼接 + `cu_seqlens` 格式，是 FlashAttention 库的接口约定，
而不是 GPU 硬件要求的。** 选择这个格式是因为一块连续内存对 GPU kernel 最友好——
kernel 内部用 `cu_seqlens[i]` 和 `cu_seqlens[i+1]` 就能定位序列边界，不需要额外跳转。

FlashAttention 的速度收益（减少 HBM 读写）来自分块算法，这部分依赖 GPU 的片上缓存（SRAM）；
`varlen` 接口本身对 GPU 硬件没有特殊要求，任何支持 CUDA 的 GPU 都能跑。

## Prefill 用 varlen，Decode 用 kvcache 接口

实际推理引擎中，Prefill 和 Decode 阶段调用的 FlashAttention 接口不同：

```
Prefill（处理 prompt，Q/K 等长）：

  使用 flash_attn_varlen_func
  原因：一个 batch 里多个请求的 prompt 长度各不相同
        varlen 接口避免 padding，节省计算

Decode（生成阶段，每次只有 1 个新 token）：

  使用 flash_attn_with_kvcache
  原因：新 token 的 Q 只有 1 行，
        但 K/V 来自整个 KV Cache（可能存在 paged memory 里）
        这个接口针对 1 对多的注意力做了专门优化
        同时支持直接传入 page table（block_table）读取分页 KV Cache
```

**本教程 FlashAttention：SRAM-aware 注意力计算 的实现是教学简化版**，使用 `flash_attn_func`（非 varlen，非 kvcache），
演示 FlashAttention 的基本封装和正确性验证。
完整的 varlen + kvcache 分发逻辑在 nano-vllm 等完整推理引擎中实现。

## 不同硬件的支持情况

```
NVIDIA GPU（CUDA）：
  flash-attn 库完整支持，varlen 和 kvcache 接口均可用
  安装：pip install flash-attn（需要 CUDA 编译环境）
  注意：仅支持 float16 / bfloat16，不支持 float32

AMD GPU（ROCm）：
  有移植版（ROCm 官方维护，hipFlashAttention）
  主流 GPU（MI200/MI300 系列）均支持
  部分接口可能稍落后 NVIDIA 版本

Apple Silicon（M 系列，MPS 后端）：
  flash-attn 库不支持 MPS
  PyTorch 内置的 scaled_dot_product_attention 有类似 IO 优化
  本教程 FlashAttention：SRAM-aware 注意力计算 的 flash_attention() 在非 CUDA 设备上自动回退到 SDPA

CPU：
  无片上缓存优化，用标准矩阵乘法
  flash-attn 不支持 CPU，回退到 SDPA
```

## 本步实现

本步不实现完整的 varlen / kvcache 接口，而是聚焦于：

1. **封装 FlashAttention**：统一的 `flash_attention()` 函数，内部处理形状转换
2. **自动回退**：CUDA + flash-attn 可用时用 FlashAttention，否则回退到 PyTorch SDPA
3. **正确性验证**：对比两种实现的输出，确认数值等价（bfloat16 精度下差异极小）

### 形状约定

```
本教程约定（与 HuggingFace 一致）：
  输入:  [batch, num_heads, seq_len, head_dim]

flash_attn_func 要求：
  输入:  [batch, seq_len, num_heads, head_dim]

flash_attention() 内部做了 transpose(1, 2) 转换：
  q_fa = q.transpose(1, 2)   # [batch, num_heads, seq_len, d] → [batch, seq_len, num_heads, d]
  ...
  out.transpose(1, 2)         # 输出转回 [batch, num_heads, seq_len, d]
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `attention.py` | FlashAttention 封装 + SDPA 回退 |
| `run.py` | 正确性验证（max_diff < 0.02） |

## 运行

```bash
# 安装 flash-attn（CUDA 环境）
pip install flash-attn

# 运行验证
python run.py
```

示例输出（CUDA 环境）：

```
设备: cuda
FlashAttention 可用: True

正确性验证: max_diff = 0.001234  （< 0.02 即通过）
两者输出一致 ✅

✅ step16_flash_attention 通过
```

非 CUDA 环境（CPU/MPS）时，`flash_attention()` 自动回退到 SDPA，
两个函数输出完全相同，差异为 0。

## 代价与限制

FlashAttention 减少了显存读写，但也有限制：

```
精度限制：
  仅支持 float16 / bfloat16
  float32 不支持（原因：SRAM 太小，float32 的块塞不下）

安装成本：
  flash-attn 需要编译 CUDA 扩展，安装慢（约 5~30 分钟）
  版本与 PyTorch / CUDA 版本强绑定，升级时可能需要重新编译

序列长度限制：
  序列很短时（< 64 token），FlashAttention 的 kernel 启动开销
  可能超过节省的读写量，收益减小
  序列越长，收益越明显
```

## 下一步

CUDA Graph：录制重放，跳过调度层：CUDA Graph——GPU kernel 每次启动都有调度开销，
Decode 阶段每步只处理 1 个 token，计算量极小但 kernel 启动次数多，
调度开销反而成了瓶颈。CUDA Graph 把固定形状的计算图录制下来，
后续重复执行时绕过 CPU 调度，大幅降低 Decode 的延迟。
