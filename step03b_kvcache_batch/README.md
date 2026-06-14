# step03b — 多请求 KV Cache + Static Batching

## 教学目标

理解 Static Batching 必须做的 padding 操作，以及由此带来的两类浪费。

## 为什么需要 Batch？

GPU 的矩阵乘法对 batch 维度是并行的：

```
串行处理（4个请求）:
  GPU: [请求A] [请求B] [请求C] [请求D]   ← 依次排队，一次只跑一个
  利用率: ~25%

Batch 处理（4个请求合并为矩阵）:
  GPU: [A B C D] 一次矩阵乘法              ← 并行！
  利用率: ~70%（但有 padding 浪费）
```

## Padding 的必要性

不同长度的序列要放进同一矩阵，必须补齐到最长长度：

```
请求A prompt: [t0 t1 t2 t3 t4 t5 t6 t7 t8 t9]              长=10
请求B prompt: [t0 t1 ... t29]                                长=30（最长）
请求C prompt: [t0 t1 t2 t3 t4]                              长=5
请求D prompt: [t0 t1 ... t19]                                长=20

Pad 到 max_len=30 后的矩阵 [4, 30]:
  请求A: [██████████░░░░░░░░░░░░░░░░░░░░]  10个有效 + 20个PAD
  请求B: [██████████████████████████████]  30个有效
  请求C: [█████░░░░░░░░░░░░░░░░░░░░░░░░░]  5个有效 + 25个PAD
  请求D: [████████████████████░░░░░░░░░░]  20个有效 + 10个PAD
           ↑ 有效计算          ↑ 无效 PAD（浪费！）
```

**本步 run.py 实测：Prefill padding 浪费约 46%**

## Decode 阶段的空转浪费

Static Batching 要求所有请求同步推进，等待最长请求：

```
Decode 推进（max_new_tokens = 20）:
  请求A: [███████████████░░░░░]  实际=15步 空转=5步
  请求B: [██████████░░░░░░░░░░]  实际=10步 空转=10步
  请求C: [████████████████████]  实际=20步 空转=0步（最长）
  请求D: [████████████░░░░░░░░]  实际=12步 空转=8步
           ↑ 有效 decode        ↑ 已完成但占着槽位（浪费！）
```

**本步 run.py 实测：Decode idle 浪费约 29%**

## Static Batching 的两大问题

| 问题 | 原因 | 后续解决方案 |
|------|------|------------|
| Prefill padding 浪费 | 长度不同必须补齐 | step06 PagedAttention（按需分配）|
| Decode idle 空转 | 短请求等长请求 | step04 Continuous Batching |

## 代码结构

```
engine.py
  SerialEngine          ← 逐个请求串行处理（对照组）
  BatchPrefillWrapper   ← 把 padded [batch, max_len] 拆回逐条 prefill
  BatchKVCacheEngine    ← 构造 padding 矩阵，暴露浪费统计数据
```

> **注**：`BatchPrefillWrapper` 教学版拆开逐条 prefill，真实 vLLM 用
> `batch matmul + attention_mask` 一次完成，GPU 上才体现并行加速。
> 本步重点是理解 **padding 的结构和浪费量**，而不是实际加速数字。

## 运行

```bash
python run.py
```

示例输出：
```
Prefill padding（pad 到最长 prompt = 30 tokens）：
  请求0: [██████████░░░░░░░░░░░░░░░░░░░░]  实际=10 pad=20
  ...
  Prefill padding 浪费: 46%  (55/120 slots)

Decode padding（等最长完成 = 20 decode steps）：
  请求0: [███████████████░░░░░]  实际=15 idle=5
  ...
  Decode idle 浪费:     29%  (23/80 steps)

Static Batching 的两大问题：
  1. Prefill padding：46% 的 prefill 计算是无效 PAD  ⚠️
  2. Decode idle：    29% 的 decode 步骤是空转等待  ⚠️
```
