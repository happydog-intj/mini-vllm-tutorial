# step04 — 连续批处理 Scheduler

## 教学目标

理解 Static Batching 的根本缺陷，以及 Continuous Batching 如何解决它。

## step03b 的遗留问题

step03b 实现了 Static Batching：把多个请求合并成一个 batch，
利用 GPU 并行矩阵乘法提升吞吐量。

但是，**batch 是在处理开始前就固定的**：

```
时刻 0：收到请求 A、B、C、D，凑成一个 batch，开始处理
时刻 T：请求 B 生成完毕（它比较短）
时刻 T：GPU 槽位2 空着——但系统不知道可以放新请求进来
时刻 2T：请求 A 生成完毕（最长的那个）
时刻 2T：整个 batch 才算结束，才能接受新请求
```

这意味着：

```
GPU 时间线:
  ══════════════════════════════════════════════
  批次1: [A B C D 处理中 ........... 全部结束]
                    ↑
                    B 早就完成了，但这个 batch 必须等 A
  ──────────────────────────────────────────────
  批次2:                                        [E F G H 处理中 ...]
                                                ↑
                                                B 完成到 E 开始，中间这段时间 GPU 在等
  ══════════════════════════════════════════════
```

**问题的根源：GPU 有空闲算力，但系统不允许插入新请求。**

## 为什么 Static Batching 必须等整批完成？

原因在于早期推理系统（比如 NVIDIA Triton Inference Server 的早期版本）
把 batch 当成一个整体来处理：

```
                     ┌──────────────────────────┐
新请求 ──→ 等待队列 ──→│  凑够 N 个请求或超时     │──→ 一起 prefill
                     │  一起 decode（同步推进）   │
                     │  全部完成才释放 batch 槽位 │
                     └──────────────────────────┘
```

这种设计简单，容易实现，但代价是：
- 短请求完成后必须等长请求，GPU 有空转
- 新到来的请求必须等当前 batch 全部完成，排队延迟高
- batch 越大，最长请求越长，等待越严重

## 实际有多浪费？

2022 年 Orca 论文（Continuous Batching 的提出者）测量了这个浪费：

```
实际负载中，一个 batch 里请求的输出长度差异很大：
  最短请求：生成 10 个 token
  最长请求：生成 500 个 token

Static Batching 下，短请求的 GPU 槽位在 90% 的时间里都在空转。
整体 GPU 利用率：约 20%~40%
```

## Continuous Batching 的核心思想

**不要等整批完成，哪个请求完成了就立刻换一个新请求进来。**

```
GPU 时间线（Continuous Batching）:
  ══════════════════════════════════════════════════════
  槽位0: [请求A ........][请求E .....][请求I ....][...]
  槽位1: [请求B ....][请求F .......][请求J ....][...]
  槽位2: [请求C ......][请求G ....][请求K .......][...]
  槽位3: [请求D ..][请求H ..][请求L ....][请求M ......][...]
  ══════════════════════════════════════════════════════
  → GPU 槽位从不空转，新请求随时可以插入
```

关键变化：**每完成一个 decode step，调度器就重新检查是否有请求完成，
有的话立刻把新请求补进来做 prefill，再继续 decode。**

## Sequence 状态机

每个请求用一个 `Sequence` 对象跟踪状态：

```
新请求到来
    │
    ▼
 WAITING  ──────────────────→  RUNNING
（等待调度）  schedule() 选中      │
    ↑                            │ 每步 decode，追加一个 token
    │                            │
    │                            ▼ is_done? (达到 max_new_tokens 或生成 EOS)
    │                         FINISHED
    │                            │
    └── 立即补充新请求 ←──────────┘
        (Continuous Batching 核心！)
```

## 调度器的工作流程

```python
while scheduler.has_work:
    prefill_seqs, decode_seqs = scheduler.schedule()
    # schedule() 做了三件事：
    #   1. 把刚完成的请求从 running 移到 finished
    #   2. 从 waiting 取新请求加入 running（有多少空位就填多少）
    #   3. 区分哪些需要 prefill（新进来的），哪些需要 decode（已有 KV Cache 的）

    for seq in prefill_seqs:
        logits, seq.past_kv = model(seq.prompt)      # prefill：一次算完整个 prompt
        seq.append_token(argmax(logits[-1]))

    for seq in decode_seqs:
        logits, seq.past_kv = model(seq.last_token,  # decode：只传 1 个新 token
                                    past_kv=seq.past_kv)
        seq.append_token(argmax(logits[-1]))
# 每步循环结束后，completed 请求立刻释放槽位给下一个 waiting 请求
```

## Static vs Continuous Batching

```
Static Batching（step03b）：
  时刻  0: batch = [A, B, C, D]  固定，不能变
  时刻  5: B 完成，槽位空转
  时刻 10: C 完成，槽位空转
  时刻 15: D 完成，槽位空转
  时刻 20: A 完成，batch 结束 → 才能接受 E, F, G, H

Continuous Batching（step04）：
  时刻  0: running = [A, B, C, D]
  时刻  5: B 完成 → 立刻换入 E
  时刻 10: C 完成 → 立刻换入 F
  时刻 12: E 完成 → 立刻换入 G
  ...     GPU 始终满载，新请求随到随处理
```

## 为什么 Continuous Batching 直到 2022 年才出现？

看完上面的设计，你可能会想：这个思路并不复杂，为什么没有更早引入？

**调度逻辑确实简单，但高效实现它需要两个前提，而这两个前提在 2022 年之前都不成熟。**

### 前提一：动态 KV Cache 内存管理

Static Batching 下，KV Cache 内存很好管理：

```
batch 开始前：为每个请求预分配 max_len 大小的 KV Cache 内存块
batch 结束后：整批一起释放
```

**具体如何预分配？以 Qwen3-0.6B 为例：**

Qwen3-0.6B 结构：28层，每层 8 个 KV 头（GQA），每头维度 64。
每个 token 的 KV Cache = K向量 + V向量 = 2 × 8头 × 64维 = 1024 个 float16 数值 = **2KB**。

假设 batch=4，max_len=500（每个请求最多生成 500 个 token）：

```
GPU 显存中预分配一个大张量：

  kv_cache 形状（概念图，把所有维度展开）:
  [4请求, 500槽位, 28层, 2(K和V), 8头, 64维]
  总大小: 4 × 500 × 28 × 2 × 8 × 64 × 2字节 = 1.8GB
```

> **注意：这个形状是教学的概念展示，不是代码里的真实存储方式。**
> 不同实现有不同的存储策略，下面说明三种：

**实现方式一：HuggingFace transformers 风格（本教程 step03a~step09 使用）**

```python
# 每层返回 (K, V) 元组，存在 Python list 里
# K/V 形状: [已生成的token数, num_kv_heads, head_dim]
past_key_values = [
    (K_layer0,  V_layer0),   # K/V: [seq_len, 8, 64]
    (K_layer1,  V_layer1),
    ...  # 共28层
]

# 每步 decode 时动态拼接（在 CPU/GPU 上 cat）
K_new = torch.cat([past_kv[0], k_current], dim=0)  # 追加新 token 的 K
```

特点：简单直观，但每步都要 `torch.cat` 分配新内存，效率较低，
无法做到真正的 GPU 显存预分配。

**实现方式二：预分配大张量（Static Batching 系统常用）**

```python
# 一次性分配 max_len 大小的张量，in-place 写入
# 每层单独一个张量
kv_cache = [
    {
        "k": torch.zeros(batch, max_len, num_kv_heads, head_dim),  # [4, 500, 8, 64]
        "v": torch.zeros(batch, max_len, num_kv_heads, head_dim),
    }
    for _ in range(num_layers)  # 28层
]

# 每步 decode 时 in-place 写入，不需要重新分配内存
kv_cache[layer]["k"][:, current_pos, :, :] = k_current
```

特点：无内存分配开销，GPU 效率高，但必须提前知道 max_len，
且每个请求始终占用 max_len 的内存（即使实际很短）。

**实现方式三：nano-vllm / vLLM 的 PagedAttention 方式**

```python
# 不按请求分配，而是分成固定大小的 Block（如每块16个token）
# 物理 KV 存储是一个全局大张量
kv_pool = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim)
# shape: [总块数, 16, 8, 64]  每块 16 个 token 槽位

# 每个请求有一个 block_table，记录用了哪些物理块
block_table_A = [7, 3, 15, ...]   # 请求A的token分散存储在物理块7、3、15...
```

特点：显存利用率高（~96%），支持 Continuous Batching 动态分配，
但需要 block_table 翻译逻辑。这就是 step06 的内容。

**这块内存在 batch 开始前就全部分配好**，不管请求实际生成多少 token，
500 个槽位的内存都占着——哪怕请求只生成了 50 个 token，另外 450 个槽位空着但无法被其他请求用。

**实际使用情况（假设请求长度差异很大）：**

```
  请求0 实际生成了 480 个 token:
  ████████████████████████████████████████████████░  96% 利用

  请求1 实际生成了  30 个 token:
  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   6% 利用
       ↑↑↑                  ↑
    实际用到              470个槽位预分配了但空着，而且被锁定无法给其他请求用

  请求2 实际生成了 200 个 token:
  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  40% 利用

  请求3 实际生成了  10 个 token:
  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2% 利用

  整体显存利用率: (480+30+200+10) / (500×4) = 720/2000 = 36%
  → 64% 的显存预分配了但没被用到！
```

Continuous Batching 下请求随进随出，这种「一次性预分配大块」的方法会产生严重碎片：

```
时刻 T：请求1（30 token）完成，释放它的 900MB 内存块
        [请求0的KV][      空洞 900MB      ][请求2的KV][请求3的KV]

时刻 T：新请求4进来，它可能需要 400MB
        空洞有 900MB，够用，但是：
        如果下一步再来请求5（600MB）和请求6（350MB），
        剩余空间 900MB 但无法同时装下 600+350=950MB
        → 内存碎片，明明有空间却放不下

```

→ 解决方案：PagedAttention（step06）

### 前提二：变长序列的高效注意力计算

Continuous Batching 下，同一 batch 内不同请求的序列长度**各不相同**：

```
某一步 decode：
  请求A：已生成 47 个 token，KV Cache 有 47 个 K/V
  请求B：已生成 312 个 token，KV Cache 有 312 个 K/V
  请求C：刚进来做 prefill，prompt 有 128 个 token

  这三个请求的注意力计算形状完全不同，无法简单地拼成一个矩阵
```

解决方案是 **FlashAttention 的变长序列接口**（`flash_attn_varlen_func`）：

```
传统做法（padding 补齐）:
  输入: [A的47个token+265个PAD, B的312个token, C的128个token+184个PAD]
  形状: [3, 312, 1024]  ← 大量 PAD，浪费

FlashAttention varlen:
  输入: [A的47个token, B的312个token, C的128个token]  ← 直接拼接，无 PAD
  形状: [487, 1024]
  cu_seqlens: [0, 47, 359, 487]  ← 告诉 GPU 每个序列的起止位置
```

**这个设计依赖 GPU 硬件吗？**

分两层回答：

**第一层：`varlen` 拼接接口本身是纯软件设计**

把变长序列拼成一维、用 `cu_seqlens` 记录边界，这是软件层面的约定，
和 GPU 硬件无关——只要注意力计算支持这种输入格式就行。

**第二层：FlashAttention 的分块算法依赖 GPU 的片上缓存（SRAM）**

FlashAttention 快的根本原因是把 Q/K/V 切成小块，每块放进 GPU 片上缓存
（Shared Memory）里计算，避免反复读写显存（HBM）：

```
GPU 内存层次结构：

  HBM（显存，大但慢）
  ├── 容量：A100 = 80GB
  └── 带宽：约 2TB/s

  SRAM（片上缓存，小但极快）
  ├── 容量：A100 每个计算单元组约 192KB
  └── 带宽：约 19TB/s  ← 比 HBM 快 10倍！

标准注意力：
  scores = Q·Kᵀ 形状 [seq_len, seq_len]，必须完整写回 HBM
  HBM 读写量: O(seq_len²)  ← seq_len=2048 时约 128MB/层

FlashAttention（分块）：
  把 Q 切成块 Q_i，K/V 切成块 K_j/V_j
  每次把一小块加载进 SRAM，在 SRAM 内完成点积+softmax+加权，只写回最终结果
  HBM 读写量: O(seq_len)  ← 减少了 seq_len 倍！
```

SRAM 的大小是硬件决定的：块大小随 SRAM 容量调整，SRAM 越大效率越高。
不同 GPU 的 SRAM 大小不同，但 FlashAttention 在所有 NVIDIA GPU 上都有收益。

**不同硬件的支持情况：**

```
NVIDIA GPU（CUDA）：  flash-attn 库完整支持，varlen 效果最好
AMD GPU（ROCm）：     有移植版（hipFlashAttention），主流 GPU 都支持
Apple MPS（M系列）：  flash-attn 不支持，用 PyTorch 内置的
                      scaled_dot_product_attention 替代
                      （有类似的 IO 优化但实现不同，本教程 step09 有回退逻辑）
CPU：                 无 SRAM 优化，用标准矩阵乘法实现
```

**结论：** varlen 拼接是纯软件设计；FlashAttention 的 IO 加速依赖 GPU SRAM，
NVIDIA GPU 支持最好，其他平台有替代方案。

### 时间线

```
2017  Transformer 论文发布，推理系统普遍用 Static Batching
2022  Orca 论文（OSDI'22）首次系统提出 Continuous Batching（iteration-level scheduling）
      同年 FlashAttention v1/v2 发布，变长序列支持逐渐完善
2023  vLLM 发布，结合 PagedAttention + Continuous Batching，成为主流推理框架
      吞吐量比 HuggingFace 朴素推理提升约 23×
```

### 总结

Continuous Batching 不需要特殊硬件，普通 GPU 就能跑。
但高效实现它需要两个软件层面的支持：

| 需要解决的问题 | 解决方案 | 在本教程的哪一步 |
|--------------|---------|---------------|
| KV Cache 动态内存管理（碎片问题） | PagedAttention | step06 |
| 变长序列高效注意力计算（无需 padding） | FlashAttention varlen | step09 |

本步（step04）的教学版实现绕开了这两个问题：
每个请求独立维护自己的 `past_key_values`，内存由 Python 管理，不涉及 GPU 显存碎片；
注意力计算沿用 step03a 的逐条处理方式，不做真正的 batch 注意力。
**这样能清晰展示调度逻辑，后续步骤再逐一解决底层问题。**



```bash
python run.py
```

本步 run.py 模拟 8 个并发请求（输出长度各不相同），
对比 Static Batching 和 Continuous Batching 的总完成时间。

## 下一步

step05a：如果来了一个超长 prompt（1000 个 token），它的 prefill
会占用整个 step，让其他请求的 decode 完全停下来等——Chunked Prefill 解决这个问题。
