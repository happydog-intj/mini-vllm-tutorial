# step05a — Chunked Prefill

## 教学目标

理解 Prefill 和 Decode 的计算特性差异，以及长 Prefill 如何阻塞 Decode，
Chunked Prefill 如何通过分块混合调度解决这个问题。

## Prefill 和 Decode 的根本差异

先回顾一下 step03a 引入的两个阶段：

```
Prefill（处理 prompt）：
  输入：整个 prompt，比如 512 个 token
  一次前向传播处理 512 个 token
  计算量：O(n²)  ← n=512，注意力矩阵 512×512
  耗时：较长（比如 100ms）

Decode（逐步生成）：
  输入：上一步生成的 1 个 token
  一次前向传播处理 1 个 token
  计算量：O(n)   ← 只需新 token 和历史 KV 做注意力
  耗时：较短（比如 10ms/token）
```

**关键区别：Prefill 是计算密集型，Decode 是内存带宽密集型。**

```
Prefill：大矩阵乘法，GPU 的算力（FLOPS）是瓶颈
  [512, 1024] @ [1024, 1024] → 大量乘法运算

Decode：小矩阵 + 大 KV Cache 读取，GPU 的显存带宽是瓶颈
  [1, 1024] @ [1024, 1024]  → 矩阵小，但要从显存读整个 KV Cache
```

这意味着：Prefill 和 Decode 不能简单地用「谁快谁慢」来比较——
它们消耗的是 GPU 的不同资源。

## 问题：长 Prefill 阻塞 Decode

step04 的 Continuous Batching 让短请求完成后立刻补充新请求，
但有一个新问题：**新请求进来时需要先做 Prefill。**

Prefill 的计算量随 prompt 长度的平方增长：

```
prompt 长度    Prefill 耗时（Qwen3-0.6B，A100 估算）
  128 tokens        约 5ms     ← 还好
  512 tokens        约 20ms    ← 勉强可接受
 1024 tokens        约 80ms    ← 开始明显
 4096 tokens        约 1300ms  ← 1.3 秒！严重阻塞
```

这时候问题来了——如果系统里已经有 8 个请求正在 Decode，
突然来了一个 4096-token 的长 prompt：

```
时间轴：

  时刻 0：8个 decode 请求正在运行，每步约 10ms
                    ↓
  时刻 0：新请求到来，需要先做 4096-token 的 Prefill
                    ↓
  时刻 0~1300ms：整个 GPU 被 Prefill 独占
  ┌─────────────────────────────────────────────────────────┐
  │  Prefill: 处理 4096 个 token [████████████████████████] │
  │  Decode请求A: 等待... ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
  │  Decode请求B: 等待... ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
  │  Decode请求C: 等待... ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
  └─────────────────────────────────────────────────────────┘
  时刻 1300ms：Prefill 结束，8个 decode 请求才能继续

  对于正在生成文字的用户A/B/C 来说：
  他们看到屏幕上的字突然停了 1.3 秒，然后才继续——这体验很差！
```

**这就是「Prefill 阻塞 Decode」的问题。**

用指标来衡量：
- **TTFT**（Time to First Token，首个 token 延迟）：长 prompt 的新请求需要等整个 Prefill 完成
- **TPOT**（Time Per Output Token，已有请求的生成速度）：被 Prefill 阻塞，期间完全停止

## 为什么 step04 的 Continuous Batching 没有解决这个问题？

step04 解决了「短请求完成后槽位空转」的问题，
但 Continuous Batching 本身对每个 step 的调度粒度是：

```
每个 step 要么全是 decode，要么有某个请求做 prefill
当有 prefill 请求时，整个 step 的时间由 prefill 决定
```

step04 的 `scheduler.py` 里，`prefill_seqs` 的 prefill 是一次性完成的：

```python
for seq in prefill_seqs:
    logits, seq.past_kv = model(seq.prompt)  # ← 整个 prompt 一次性 prefill
    seq.append_token(...)
```

如果 `seq.prompt` 有 4096 个 token，这一行就会执行 1300ms，
期间所有 `decode_seqs` 都在等待。

## Chunked Prefill 的解决思路

**把长 Prefill 切成小块（chunk），每步只处理 chunk_size 个 token，
剩余时间留给 decode 请求。**

```
chunk_size = 512 时，4096-token 的 Prefill 分 8 步完成：

时刻 0：   [Prefill chunk1: 512tok][Decode A/B/C/D/E/F/G/H]  ← 约 20ms+80ms
时刻 100ms：[Prefill chunk2: 512tok][Decode A/B/C/D/E/F/G/H]
时刻 200ms：[Prefill chunk3: 512tok][Decode A/B/C/D/E/F/G/H]
...
时刻 700ms：[Prefill chunk8: 512tok][Decode A/B/C/D/E/F/G/H]
时刻 800ms：新请求 Prefill 完成，开始 Decode

对比无 Chunked Prefill：
  Decode 请求被阻塞 1300ms
对比有 Chunked Prefill：
  每步 Decode 只多延迟 20ms（一个 chunk 的 prefill 时间）
```

**Decode 请求的 TPOT 从「阻塞 1300ms」变成「每步多 20ms」。**

## 代价与权衡

Chunked Prefill 不是免费的：

```
无 Chunked Prefill：
  新请求的 TTFT = Prefill 时间 = 1300ms（但期间 decode 被阻塞）

有 Chunked Prefill（chunk_size=512）：
  新请求的 TTFT = 8步 × 每步时间 ≈ 8 × 100ms = 800ms（比 1300ms 快，因为有并发）
  已有请求的 TPOT 影响：每步多约 20ms

chunk_size 越小：
  → decode 请求受影响越小（每步 prefill 时间短）
  → 新请求的 TTFT 越长（需要更多步才能完成 prefill）

chunk_size 越大：
  → 新请求 TTFT 越短
  → decode 请求每步等待越久
```

nano-vllm 的默认 `chunk_size=512`，在 TTFT 和 TPOT 之间取了一个平衡点。

## 实现细节

每个 Sequence 新增 `prefill_offset` 字段，记录已处理了多少 prompt token：

```python
class Sequence:
    prefill_offset: int = 0  # 已 prefill 的 token 数

# 调度时：每步最多处理 chunk_size 个新 token
start = seq.prefill_offset
end = min(start + chunk_size, len(seq.prompt))
chunk = seq.prompt[start:end]
logits, seq.past_kv = model(chunk, past_kv=seq.past_kv)  # 增量 prefill
seq.prefill_offset = end
```

注意：增量 prefill 依赖 KV Cache——第二块 chunk 处理时，
第一块 chunk 的 K/V 已经存在 `past_kv` 里，不需要重算。
这和 Decode 阶段复用 KV Cache 的机制完全相同。

## 关键参数

- `chunk_size`：每步最多处理的 prefill token 数（nano-vllm 默认 512）
- `prefill_offset`：当前序列已完成 prefill 的 token 数

## 运行

```bash
python run.py
```

## 下一步

step05b：Preemption——如果 KV Cache 显存装不下所有 running 请求怎么办？
