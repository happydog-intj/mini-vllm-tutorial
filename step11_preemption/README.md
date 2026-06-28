# Preemption：抢占避免 OOM — Preemption 抢占

## 为什么需要抢占？

在 [Chunked Prefill：切片长 Prompt 的 Continuous Batching](../step10_chunked_prefill/README.md) 中，调度器持续把 waiting 队列的请求接入 running 队列。
这里有一个根本性的问题：**生成长度在请求开始时是未知的**。

系统无法预知一个请求会生成 5 个 token 还是 500 个 token。每接入一个新请求，
调度器都在赌它不会把 KV Cache 撑满。一旦赌错——

```
时刻 T：接入 8 个请求，每个 prompt 长 5 tokens
         running 队列总占用：8 × 5 = 40 个 KV 槽位  ← 看起来没问题

时刻 T+n：每个请求已生成多个 token，KV 长度增长到 8 × (5 + n)
           当 n 足够大：总占用 > max_kv_slots     ← 内存不足！
```

此时系统面临两个选择：

| 选择 | 结果 |
|------|------|
| 不做任何处理 | `RuntimeError: KV Cache 已满` 直接崩溃，所有正在运行的请求全部丢失 |
| 抢占低优先级请求 | 释放部分 KV Cache，系统继续运行，被抢占请求稍后恢复 |

抢占机制的目标是**优雅降级而非崩溃**：当内存不足时，牺牲部分请求的进度，保全系统整体可用性。

## KV Cache 为何会耗尽？

每个 token 在 Transformer 的每一层都需要存储一个 Key 向量和一个 Value 向量（即 KV Cache）。
随着 decode 阶段不断生成新 token，每个序列占用的 KV 槽位数会线性增长：

```
序列状态（kv_len = 已生成的总 token 数，含 prompt）：

  prefill 完成：[p1, p2, p3, p4, p5]          kv_len = 5
  生成 1 token：[p1, p2, p3, p4, p5, t1]       kv_len = 6
  生成 2 token：[p1, p2, p3, p4, p5, t1, t2]   kv_len = 7
  ...

  8 个并发序列，每步 decode 需要的槽位总数：
  sum(seq.kv_len + 1 for seq in running)
       ↑ 当前已占用    ↑ 下一步新增的 1 个
```

`+1` 的原因：decode 每步会生成 1 个新 token，并把它的 K/V 追加进 KV Cache，
所以下一步开始前必须确保还有 1 个空闲槽位。

## 不抢占的后果

`NoPreemptionEngine` 直接在内存不足时抛出异常：

```python
# engine.py: NoPreemptionEngine
if self._used_slots + 1 > self.max_kv_slots:
    raise RuntimeError(f"KV Cache 已满：已用 {self._used_slots}, 上限 {self.max_kv_slots}")
```

这不只是某一个请求失败——整个 batch 的所有请求都会因为未捕获的异常而丢失。
在生产环境中，这等价于服务进程崩溃重启，所有已完成的推理工作全部白费。

## 抢占流程

`PreemptionScheduler.schedule()` 在每个调度步骤开始时检查内存状况，
若不够则主动驱逐，腾出空间后再继续：

```
每个调度步骤开始时：

  ┌─────────────────────────────────────────────────────────┐
  │  计算下一步所需槽位                                       │
  │  needed = sum(seq.kv_len + 1 for seq in running)        │
  └────────────────────┬────────────────────────────────────┘
                       │
          ┌────────────┴──────────────┐
          │ needed <= max_kv_slots?   │
          └────────────┬──────────────┘
                       │
            ┌──────────┴──────────┐
           Yes                    No
            │                     │
            ▼                     ▼
       继续调度             选择 victim（LIFO）
                           victim = running[-1]
                                │
                    ┌───────────┴───────────┐
                    │ 释放 victim 的 KV Cache │
                    │ victim.free_kv_cache()  │
                    │ past_key_values = None  │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │ 重置到 prompt 状态      │
                    │ token_ids = prompt_ids │
                    │ _generated_count = 0   │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │ 插回 waiting 队首       │
                    │ waiting.appendleft()   │
                    │ （下轮优先恢复）        │
                    └───────────┬───────────┘
                                │
                         重新检查 needed
                         （可能需要再驱逐一次）
```

驱逐是一个循环，直到 `needed <= max_kv_slots` 或 `running` 为空才停止。

## 为什么选 LIFO 策略？

LIFO（后进先出）意味着**最晚进入 running 队列的请求最先被抢占**。

从直觉上理解：越早进入 running 队列的请求，已经生成了越多 token，
距离完成越近。抢占一个已经生成了 90% token 的请求，代价远大于
抢占一个刚刚 prefill 完成、只生成了 1 个 token 的请求。

LIFO 不是最优的，但它是一个**低开销的合理近似**：
- 无需维护优先级队列
- 无需估算每个请求距完成的距离
- 实现简单，符合大多数场景的直觉

## 被抢占请求的代价：重新 Prefill

这是抢占机制最重要的代价，需要理解清楚。

被抢占的请求在恢复时，**必须从头重新做一次 prefill**：

```
请求 A 的生命周期（遭遇抢占）：

  第 1 次 prefill：处理 prompt [p1,p2,p3,p4,p5]      ← 计算一次
  decode 若干步：生成 t1, t2, t3 ...

  KV Cache 不足 → 被抢占：
    - past_key_values 被释放（显存回收）
    - token_ids 重置回 [p1,p2,p3,p4,p5]
    - 插回 waiting 队首

  第 2 次 prefill：重新处理 prompt [p1,p2,p3,p4,p5]  ← 再算一次（浪费！）
  继续 decode：重新生成 t1, t2, t3 ...
```

注意：已生成的 token（t1, t2, t3）也被丢弃了，要重新生成。
这意味着抢占会造成**重复计算**，是真实的性能损耗。

抢占发生越频繁，浪费越多。调度器的目标是尽量减少不必要的抢占，
这也是为什么被抢占的请求会插到 `waiting` **队首**而非队尾——
尽快恢复它，减少重新 prefill 的次数。

## 与 Swap to CPU 的对比

本步骤实现的是最简单的抢占策略：**直接丢弃 KV Cache，恢复时重新计算**。

更高级的实现（如 vLLM）支持 **Swap to CPU**：
被抢占时，把 KV Cache 从显存（高带宽显存 HBM）转移到 CPU 内存，
恢复时再搬回来，避免重新 prefill 的计算代价。

```
┌─────────────────────────┐      ┌─────────────────────────┐
│   本步骤：Recompute      │      │   进阶：Swap to CPU      │
├─────────────────────────┤      ├─────────────────────────┤
│ 抢占：释放 KV Cache      │      │ 抢占：KV Cache → CPU 内存│
│ 恢复：重新做 prefill     │      │ 恢复：CPU 内存 → 显存    │
│ 代价：重复计算           │      │ 代价：数据搬运（PCIe带宽）│
│ 实现：极简               │      │ 实现：需要管理 CPU 内存   │
└─────────────────────────┘      └─────────────────────────┘
```

两种方式各有适用场景。当 prefill 速度很快（短 prompt）时，重新计算可能比
搬运数据更快；当 prompt 很长时，Swap to CPU 能节省大量重复计算。

## 代码结构

```
step11_preemption/
├── scheduler.py   # Sequence（含 kv_len、free_kv_cache）+ PreemptionScheduler
├── engine.py      # NoPreemptionEngine（对照组）+ PreemptionEngine
├── model.py       # TinyTransformerWithKVCache（复用自前几步）
└── run.py         # 演示：max_kv_slots=20，8个请求，触发抢占
```

关键新增属性：

```python
# scheduler.py: Sequence
@property
def kv_len(self) -> int:
    """当前占用的 KV 槽位数（= 已生成的总 token 数，含 prompt）"""
    return len(self.token_ids)

def free_kv_cache(self):
    """释放 KV Cache（被抢占时调用）"""
    self.past_key_values = None
```

## 运行

```bash
python run.py
```

预期输出：

```
=======================================================
Preemption：KV Cache 满时优雅降级 vs 崩溃
=======================================================
  无 Preemption: RuntimeError: KV Cache 已满：... 💥
  有 Preemption: 全部 8 个请求成功完成 ✅
  驱逐发生次数: N   ← N > 0，说明确实发生了抢占

✅ step11_preemption 通过
```

`max_kv_slots=20` 故意设得很小（8 个请求 × 5 token prompt = 40，首步就超限），
确保抢占一定会被触发，而不是偶发的边界情况。

## 下一步

现在系统能在内存不足时优雅降级，不会崩溃。
但 KV Cache 的管理方式还很粗糙：每个请求的 KV Cache 是一整块 Tensor，
长度固定为序列当前长度，无法在多个请求之间共享相同的 prompt 前缀。

**PagedAttention：分页内存管理 将引入 PagedAttention**：把 KV Cache 切成固定大小的"页"（Page），
像操作系统管理虚拟内存一样管理显存，彻底解决内存碎片问题，
并为 prefix caching（多个请求共享同一 system prompt 的 KV Cache）奠定基础。
