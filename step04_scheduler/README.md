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

## 运行

```bash
python run.py
```

本步 run.py 模拟 8 个并发请求（输出长度各不相同），
对比 Static Batching 和 Continuous Batching 的总完成时间。

## 下一步

step05a：如果来了一个超长 prompt（1000 个 token），它的 prefill
会占用整个 step，让其他请求的 decode 完全停下来等——Chunked Prefill 解决这个问题。
