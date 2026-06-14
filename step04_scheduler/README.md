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

内存布局（简单、规则）:
  [请求A的KV: 500个槽位][请求B的KV: 500个槽位][请求C的KV: 500个槽位]
```

Continuous Batching 下，每个请求随时可能完成、随时可能插入新请求，
KV Cache 的分配和释放变得**动态且不规则**：

```
时刻 0: [请求A的KV: 已用20槽][请求B的KV: 已用30槽][请求C的KV: 已用15槽]
时刻 5: 请求B完成 → 释放它的内存
        [请求A的KV: 已用25槽][           空洞           ][请求C的KV: 已用20槽]
时刻 5: 请求E进入 → 在哪里分配它的内存？
        [请求A的KV: 已用25槽][请求E的KV: 已用5槽 + 空闲][请求C的KV: 已用20槽]
```

**问题：GPU 显存没有像 CPU 那样的动态内存分配器（malloc/free）。**
GPU 上的内存碎片问题非常严重——频繁分配和释放不规则大小的内存块后，
剩余内存可能有很多，但全是碎片，放不下新的连续大块。

解决方案是 vLLM（2023 年）提出的 **PagedAttention**——
把 KV Cache 切成固定大小的小块（Block），像操作系统管理内存页一样管理，
彻底解决碎片问题。**这就是 step06 的内容。**

在 PagedAttention 出现之前，Continuous Batching 只能用简单粗暴的方法：
预估最大需求量，一次性分配，碎片问题悬而未决。

### 前提二：变长序列的高效注意力计算

GPU 矩阵运算最擅长处理形状**规则**的输入。

Static Batching 下，batch 内所有序列等长（padding 补齐），
注意力计算的输入形状固定为 `[batch, seq_len, hidden]`，
可以直接用标准 CUDA kernel 高效计算。

Continuous Batching 下，同一 batch 内不同请求的序列长度**各不相同**：

```
某一步 decode：
  请求A：已生成 47 个 token，KV Cache 有 47 个 K/V
  请求B：已生成 312 个 token，KV Cache 有 312 个 K/V
  请求C：刚进来做 prefill，prompt 有 128 个 token

  这三个请求的注意力计算形状完全不同，无法简单地拼成一个矩阵
```

解决方案是 **FlashAttention 的变长序列支持**（`flash_attn_varlen_func`）——
把不同长度的序列拼接成一个一维向量，用 `cu_seqlens`（每个序列的起止位置）
告诉 kernel 边界在哪里，从而在一次 kernel 调用内处理所有序列：

```
传统做法（padding 补齐）:
  输入: [A的47个token + 265个PAD, B的312个token, C的128个token + 184个PAD]
  形状: [3, 312, 1024]  ← 大量 PAD，浪费

FlashAttention varlen:
  输入: [A的47个token, B的312个token, C的128个token]  ← 直接拼接
  形状: [487, 1024]  ← 无 PAD，无浪费
  cu_seqlens: [0, 47, 359, 487]  ← 告诉 kernel 每个序列的起止
```

**FlashAttention 的这个特性在 2022 年底才趋于成熟。**

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
