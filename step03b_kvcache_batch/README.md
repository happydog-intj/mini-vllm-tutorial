# step03b — 多请求 KV Cache + Static Batching

## 教学目标

理解 Static Batching 必须做的 padding 操作，以及 **GPU batch 矩阵乘法**如何带来并行加速——以及 padding 带来的浪费代价。

## Batch 加速的本质：GPU 矩阵乘法

LLM 的绝大部分计算是线性层（`Linear`）：

```
单请求：Y = X @ W^T
  X: [seq_len, hidden]   W: [hidden, out]   Y: [seq_len, out]
```

### 串行处理（4个请求）

```python
y1 = x1 @ W.T   # GPU 算一次 [10, 1024] @ [1024, 1024]
y2 = x2 @ W.T   # GPU 算一次 [30, 1024] @ [1024, 1024]
y3 = x3 @ W.T   # GPU 算一次 [ 5, 1024] @ [1024, 1024]
y4 = x4 @ W.T   # GPU 算一次 [20, 1024] @ [1024, 1024]
# 4次独立 GEMM，GPU CUDA core 利用率低
```

### Batch 处理（4个请求合并）

```python
# 先 pad 到同一长度，拼成 batch 矩阵
X = stack([x1_padded, x2_padded, x3_padded, x4_padded])
# X: [4, 30, 1024]  ← batch=4, max_seq_len=30, hidden=1024

Y = X @ W.T   # 一次 batch GEMM！
# Y: [4, 30, 1024]
```

GPU 执行 `[4, 30, 1024] @ [1024, 1024]` 的 batch GEMM：

```
CUDA Core 分配示意（简化）：

串行（4次独立 GEMM）:
  时刻1: [Core00~Core3F 算 x1@W] [Core40~Core7F 空闲]
  时刻2: [Core00~Core3F 算 x2@W] [Core40~Core7F 空闲]
  时刻3: [Core00~Core3F 算 x3@W] [Core40~Core7F 空闲]
  时刻4: [Core00~Core3F 算 x4@W] [Core40~Core7F 空闲]
  利用率: ~25%

Batch GEMM（1次）:
  时刻1: [Core00~Core1F 算 x1@W]
         [Core20~Core3F 算 x2@W]
         [Core40~Core5F 算 x3@W]
         [Core60~Core7F 算 x4@W]
  利用率: ~85%  ← 所有 CUDA Core 同时工作！
```

### 关键数字感受（Qwen3-0.6B，hidden=1024）

| 操作 | 矩阵形状 | 每次 GEMM 的 FLOPs |
|------|---------|------------------|
| Q_proj（串行 batch=1） | [30, 1024] @ [1024, 1024] | 63M |
| Q_proj（batch=4，pad）  | [4, 30, 1024] @ [1024, 1024] | 252M |
| 耗时比（实测，A100）    | 串行 ×4 ≈ 3.2ms | batch ×1 ≈ 0.9ms（**3.5×**） |

> batch GEMM 的加速来自两方面：
> 1. GPU 的 **Tensor Core** 对大矩阵吞吐量更高（小矩阵利用不足）
> 2. 减少了 kernel launch overhead（4次→1次 CUDA kernel 调用）

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
           ↑ GPU 并行算这些  ↑ 也被 GPU 算了，但结果丢弃（浪费！）
```

**本步 run.py 实测：Prefill padding 浪费约 46%**

padding token 的 GEMM 结果不会用到（用 `attention_mask` 屏蔽），
但 GPU 已经算了这部分——这就是 padding 浪费的根源。

## Decode 阶段的空转浪费

Static Batching 要求所有请求同步推进，等待最长请求：

```
Decode 推进（max_new_tokens = 20）:
  请求A: [███████████████░░░░░]  实际=15步 空转=5步
  请求B: [██████████░░░░░░░░░░]  实际=10步 空转=10步
  请求C: [████████████████████]  实际=20步 空转=0步（最长）
  请求D: [████████████░░░░░░░░]  实际=12步 空转=8步
           ↑ 有效 decode        ↑ 已完成但占着 GPU 槽位（浪费！）
```

**本步 run.py 实测：Decode idle 浪费约 29%**

## Static Batching 的两大问题

| 问题 | 原因 | 浪费量（本步实测）| 后续解决方案 |
|------|------|-----------------|------------|
| Prefill padding 浪费 | 长度不同必须补齐 | ~46% | step06 PagedAttention |
| Decode idle 空转 | 短请求等长请求 | ~29% | step04 Continuous Batching |

## 代码结构

```
engine.py
  SerialEngine          ← 逐个请求串行处理（对照组）
  BatchPrefillWrapper   ← 把 padded [batch, max_len] 拆回逐条 prefill
  BatchKVCacheEngine    ← 构造 padding 矩阵，暴露浪费统计数据
```

> **教学注**：`BatchPrefillWrapper` 内部仍逐条 prefill（保持 model 代码不变），
> 真实 vLLM 直接用 `[batch, seq_len, hidden]` 的 batch matmul + `attention_mask`。
> 本步重点展示 **padding 的结构和浪费量**；GPU 并行加速原理见上方图解。

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
