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

## flash_attn_varlen_func 实现原理

### 接口签名

```python
from flash_attn import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q,              # [total_tokens, num_heads, head_dim]
    k,              # [total_tokens, num_kv_heads, head_dim]
    v,              # [total_tokens, num_kv_heads, head_dim]
    cu_seqlens_q,   # [batch_size + 1]，int32
    cu_seqlens_k,   # [batch_size + 1]，int32
    max_seqlen_q,   # int，最长 q 序列长度
    max_seqlen_k,   # int，最长 k/v 序列长度
    causal=True,    # 是否使用因果掩码
)
# output: [total_tokens, num_heads, head_dim]
```

### 为什么比 padded 版本快

**padded 版本的浪费**：

```
batch 里 3 个序列，长度 [8, 256, 32]
padding 后全部补到 256：
  实际 token 数：8 + 256 + 32 = 296
  padding 后：256 × 3 = 768
  浪费：(768 - 296) / 768 = 61% 的计算是无效 PAD
```

**varlen 的做法**：把所有 token 直接拼成 `[296, num_heads, d_head]`，kernel 内部用 `cu_seqlens` 定位每条序列的边界。

### kernel 内部执行流程

首先理解「为什么是在 kernel 内部执行」：

**Python 调用 → CUDA kernel 的边界**

```
Python：flash_attn_varlen_func(q, k, v, cu_seqlens_q, ...)
                    ↓ 一次 Python → C++ → CUDA 调用
CUDA kernel 启动（1 次 launch）
                    ↓ 以下全部在 GPU 上执行，CPU 不参与
    GPU 有数千个 CUDA core，同时处理所有 block
                    ↓
返回 output tensor 到 Python
```

一旦 kernel 启动，CPU 就不参与了。所有的循环、分块、矩阵乘法都在 GPU 上的 CUDA core 里执行，用的是 GPU 的 SRAM（片上缓存）和寄存器，不回 CPU，不经过 Python 解释器。

**kernel 内部的两层并行**：

```
第 1 层：序列级并行
  不同序列分配给不同的 CUDA Thread Block
  序列A 和 序列B 在 GPU 上同时处理

第 2 层：块级并行（FlashAttention tiling）
  每条序列内部，Q 的不同 tile（行块）分配给不同的 Warp
  同一序列的不同 Q tile 也在 GPU 上并行处理
```

**伪代码（对应 GPU 上真实执行逻辑）**：

```
# 每个 CUDA Thread Block 处理一条序列的一个 Q tile
# GPU 同时启动 (batch_size × num_q_tiles) 个 Thread Block

thread_block_i_j:                   # 序列 i，Q 的第 j 个 tile
    q_tile = q[cu_seqlens[i] + j*Br : cu_seqlens[i] + (j+1)*Br]  # 从 HBM 读一次
    # 加载到 SRAM（片上缓存，速度是 HBM 的 10×）

    m = -inf    # online softmax 的最大值（维护在寄存器里）
    acc = 0     # 累积输出（维护在寄存器里）

    for k_tile in k[cu_seqlens[i] : cu_seqlens[i+1]]:  # 遍历所有 K tile
        k_block = load_to_sram(k_tile)   # 从 HBM 读一次
        v_block = load_to_sram(v_tile)   # 从 HBM 读一次

        s = q_tile @ k_block.T           # 在 SRAM 内计算，不写 HBM
        m_new = max(m, row_max(s))       # 更新 online softmax 状态
        acc = acc * exp(m - m_new) + softmax(s) @ v_block  # 累积
        m = m_new

    output_tile = acc / normalizer       # 最终结果
    write_to_HBM(output_tile)           # 只写一次 HBM
```

**关键**：`QK^T` 的中间结果（完整的 score 矩阵）**从未写回 HBM**，全程在 SRAM 里处理完就丢弃。这就是 HBM 读写从 O(N²) 降到 O(N) 的根本原因。

注意：这个循环在 GPU kernel 内部展开，不是 Python 循环——所有序列在 GPU 上并行处理。

### causal mask 在 varlen 中的处理

`causal=True` 时，kernel 只需对每个序列内部施加下三角掩码。由于序列边界由 `cu_seqlens` 明确标记，**跨序列的 token 天然不会互相 attend**，不需要额外的跨序列 mask。

```
序列A的 token 只能 attend 序列A内的历史 token
序列B的 token 只能 attend 序列B内的历史 token
                ↑
    cu_seqlens 保证了这个隔离，无需显式跨序列 mask
```

---

## flash_attn_with_kvcache 实现原理

### 接口签名

```python
from flash_attn import flash_attn_with_kvcache

output = flash_attn_with_kvcache(
    q,                  # [batch, seqlen_q, num_heads, head_dim]，decode 时 seqlen_q=1
    k_cache,            # [batch_size, seqlen_k, num_kv_heads, head_dim] 或分页格式
    v_cache,            # 同上
    cache_seqlens=None, # [batch_size]，每条序列在 cache 里实际有效的长度
    block_table=None,   # [batch_size, max_num_blocks]，分页 KV Cache 的 block table
    causal=True,
)
# output: [batch, seqlen_q, num_heads, head_dim]
```

### decode 阶段为什么不用 varlen？

Decode 时每条序列只有 1 个新 token（Q 的 seq_len=1），但需要 attend 到所有历史 K/V（K/V 的 seq_len 可能是几千）。这是一个典型的**1 对多**（asymmetric attention）场景：

```
Q:   [batch, 1, num_heads, d_head]    ← 1 个新 token
K/V: [batch, N, num_heads, d_head]    ← N 个历史 token（来自 KV Cache）

seq_len_q ≠ seq_len_k，varlen 接口设计是 q 和 k 等长的（prefill），
不适合这个 1 对多的场景。
```

`flash_attn_with_kvcache` 专门为这种场景设计，内部对 1 对多的访问模式做了优化。

### 分页 KV Cache 的直接支持

最关键的特性是 `block_table` 参数：

```python
# block_table: [batch_size, max_num_blocks_per_seq]
# 每行是一条序列的物理 block ID 列表
block_table = torch.tensor([
    [3, 7, 12, 0, 0],   # 序列A：物理 block 3, 7, 12（后面是 padding）
    [1, 5, 0, 0, 0],    # 序列B：物理 block 1, 5
], dtype=torch.int32)

output = flash_attn_with_kvcache(
    q,
    k_cache,      # [total_blocks, block_size, num_kv_heads, d_head]
    v_cache,      # 全局物理 KV pool
    block_table=block_table,
    cache_seqlens=torch.tensor([48, 32]),  # 序列A有48个有效token，序列B有32个
)
```

**kernel 内部执行流程**：

同样地，`flash_attn_with_kvcache` 也是一次 kernel launch，CPU 不参与内部循环：

```
Python：flash_attn_with_kvcache(q, k_cache, v_cache, block_table, ...)
                    ↓ 一次 launch
CUDA kernel 启动
                    ↓ 以下全部在 GPU 上执行
    每个 Thread Block 负责一条序列的输出
```

**伪代码（对应 GPU 上真实执行逻辑）**：

```
# 每个 CUDA Thread Block 处理一条序列（batch 里的第 i 条）
thread_block_i:
    q_i = q[i, 0]          # decode：1 个新 token 的 Q，从 HBM 读一次
                            # 加载到 SRAM

    m = -inf               # online softmax 状态（在寄存器里）
    acc = 0

    # 遍历该序列在 KV Cache 里的所有物理 block
    for block_idx in range(cache_seqlens[i] // block_size + 1):
        physical_block = block_table[i, block_idx]

        # 直接从分页 KV pool 按物理地址读取（无需 gather 到连续内存）
        k_block = k_cache[physical_block]   # [block_size, kv_heads, d_head]，从 HBM 读一次
        v_block = v_cache[physical_block]   # 同上
        # 加载到 SRAM

        # 在 SRAM 内完成 attention score 累积（中间结果不写回 HBM）
        s = q_i @ k_block.T                 # [1, block_size]
        m_new = max(m, max(s))
        acc = acc * exp(m - m_new) + softmax(s) @ v_block
        m = m_new

    output[i, 0] = acc / normalizer         # 只写一次 HBM
```

**为什么可以直接按 `block_table` 访问 paged KV Cache？**

CUDA kernel 可以拿到任意 GPU 显存地址并直接读取，`block_table[i, block_idx]` 给出物理 block ID，kernel 用它计算出 `k_cache` 里的偏移量，直接寻址——不需要先把散落的 block gather 成连续内存，省去了整个 gather 步骤。

这就是 step14_2 中 `gather_kv_from_blocks` 在真实 vLLM 里被完全消除的原因。

### 与 step14 系列的对比

| 步骤 | 我们的实现 | flash_attn_with_kvcache |
|------|-----------|------------------------|
| KV 写入 | step14_1 的 advanced indexing scatter | kernel 内部直接写入 |
| KV gather | step14_2 的 advanced indexing gather | **消除**，kernel 直接按 block_table 访问 |
| Attention 计算 | step14_3 的 bmm / 本章的 SDPA | kernel 内部 FlashAttention 分块 |
| Causal mask | step14_4 的 broadcast 构造 | **消除**，kernel 内部隐式处理 |

`flash_attn_with_kvcache` 把这四步全部融合进一个 kernel，这是 nano-vllm 相比本教程快 28× 的核心来源之一。

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

✅ step15_flash_attention 通过
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
