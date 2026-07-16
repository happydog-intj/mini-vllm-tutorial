# adv06 — Pipeline Parallel (PP) 流水线并行

## 1. 教学目标

本章用**纯 Python 串行模拟器**演示两种流水线并行调度策略的显存气泡差异。

学完本章你将能够：

- 理解为何朴素串行（bubble-heavy）流水线让多数 GPU 空闲等待
- 对比 **GPipe** 与 **1F1B** 在显存峰值占用上的本质区别
- 解释"显存气泡"（bubble memory）的含义与 1F1B 如何缩减峰值激活驻留数
- 阅读 Megatron-LM / DeepSpeed PP 代码时，能对应本章的 `gpipe_schedule` / `onef_oneb_schedule`

---

## 2. 问题

**单卡装不下大模型。**

以 LLaMA-70B（fp16）为例，仅权重就需要 ~140 GB。一张 A100（80 GB）远远不够。

**解法一：层间切分（Pipeline Parallelism）**

把模型按层拆到多张 GPU 上，每张 GPU 只持有若干层：

```
GPU0: Layer 0-7
GPU1: Layer 8-15
GPU2: Layer 16-23
GPU3: Layer 24-31
```

数据像流水线一样流过：batch 先在 GPU0 完成层 0-7 的前向，再传给 GPU1，依此类推。

**但朴素流水线有"气泡"（bubble）：**

```
时间轴 →
GPU0: [F][  等待  ][B]
GPU1:     [F][  等待  ][B]
GPU2:         [F][  等待  ][B]
GPU3:             [F][B]
      ←─ bubble ─→
```

当 GPU3 在做前向时，GPU0 处于空闲——它在等 GPU3 的反向梯度传回来。
这段空闲时间称为 **流水线气泡（pipeline bubble）**，占总时间的比例 = `(p-1) / (n+p-1)`，
其中 p = stage 数，n = microbatch 数。

**另一个问题：GPipe 的显存气泡。**

GPipe 要求所有 microbatch 全部完成前向后才开始反向——这意味着所有 microbatch 的中间激活必须**同时驻留**在显存中。当 n 很大时，激活显存成为瓶颈。

---

## 3. 原理

### GPipe 调度：全前向后统一反向

```
时间轴 →（p=4 stage，n=4 microbatch，F=前向，B=反向）

         mb0   mb1   mb2   mb3        mb0   mb1   mb2   mb3
GPU0: [F0][F1][F2][F3]──────────── [B3][B2][B1][B0]
GPU1:     [F0][F1][F2][F3]──────── [B3][B2][B1][B0]
GPU2:         [F0][F1][F2][F3]──── [B3][B2][B1][B0]
GPU3:             [F0][F1][F2][F3] [B3][B2][B1][B0]
      ←─────── 全部 F 完成 ────────→←── 统一 B ──→

峰值驻留激活 (GPU0 视角): 4 个 mb 同时驻留 ← 显存峰值高
```

### 1F1B 调度：一前向一反向交替

```
时间轴 →（p=4，n=4）

         Warmup          Steady      Cooldown
GPU0: [F0][F1][F2][F3|B0][B1][ B2  ][ B3  ]
GPU1:     [F0][F1][F2|B0][F3|B1][B2][ B3  ]
GPU2:         [F0][F1|B0][F2|B1][F3|B2][B3]
GPU3:             [F0|B0][F1|B1][F2|B2][F3|B3]
      ←── p-1 ──→

注: | 为 F 与 B 的切换点（同一个 step 内）

峰值驻留激活 (GPU0 视角): 仅 warmup 深度 = p-1 = 3 个 mb ← 显存峰值低
```

**显存峰值对比：**

| 策略  | 峰值驻留 mb 数 | n=4,p=4 | n=8,p=4 | n=32,p=4 |
|-------|---------------|---------|---------|----------|
| GPipe | n             | 4       | 8       | 32       |
| 1F1B  | p-1           | 3       | 3       | 3        |
| 节省  | —             | 25%     | 62.5%   | 90.6%    |

1F1B 将激活显存峰值从 `O(n)` 压缩到 `O(p)`，当批量大（n >> p）时效果显著。

---

## 4. 实现细节

### `Device`

```python
class Device:
    def __init__(self, name, layers, comm_latency=0.01, fwd_time=0.02, bwd_time=0.04):
        ...
```

模拟单个 GPU stage：包含所持层列表、stage 间通信延迟、前向/反向执行时间。

### `gpipe_schedule(devices, num_microbatches)`

串行模拟 GPipe 调度：

1. 外层循环遍历 `mb in range(n)`，内层遍历所有 stage 做前向
2. 全部前向完成后，外层循环遍历 `mb`，内层逆序遍历 stage 做反向

返回 `(total_time, events)`，`events` 是带时间戳的操作记录。

### `onef_oneb_schedule(devices, num_microbatches)`

串行模拟 1F1B 三阶段调度：

1. **Warmup**：送入 `p-1` 个 mb 做前向（填满 pipeline）
2. **Steady-state**：每引入一个新 mb 做 F，立即对最旧的 pending mb 做 B（维持显存平衡）
3. **Cooldown**：对所有剩余 mb 做 B（排空 pipeline）

### `compute_theoretical_peak(schedule_type, num_microbatches, num_stages)`

理论分析函数（不依赖事件列表）：

- GPipe → 返回 `n`
- 1F1B  → 返回 `p-1`

用于断言与报告，绕过串行模拟无法体现并行激活并发驻留的局限。

---

## 5. 教学版 vs 真实框架

### 本教学版的局限性

| 方面 | 教学版 | 真实框架 |
|------|--------|----------|
| 执行方式 | 单机串行模拟 | 多进程 / 多 GPU 真分布式 |
| 通信 | 固定延迟常数 | NCCL send/recv，受带宽和拓扑影响 |
| 总时间 | 串行累加，1F1B 不必然更短 | 真实并行下 1F1B 吞吐 ≈ GPipe |
| 显存峰值 | 理论推导（不可实测） | 可用 `torch.cuda.max_memory_allocated()` 直接测量 |
| 1F1B 正确性 | 教学简化版，正确演示三阶段结构 | Megatron 有更复杂的调度优化（interleaved 1F1B） |

**因此**：本模拟器的串行总时间**不能**用于比较两种调度的速度优劣。显存气泡优势须看 `compute_theoretical_peak()` 的理论值。

### 真实框架实现

**Megatron-LM**（NVIDIA）：
- 实现了标准 1F1B 和 **Interleaved 1F1B**（虚拟 stage，将 bubble 从 `(p-1)/n` 进一步压缩到 `(p-1)/(mn)`）
- 代码: `megatron/schedules.py` → `forward_backward_pipelining_without_interleaving`

**DeepSpeed**：
- `PipelineEngine` 支持 `train_batch()` 内置 1F1B 调度
- 自动处理 activation checkpointing 与 pipeline bubble 的 trade-off

**vLLM PP**：
- vLLM 的 PP 主要用于推理（无反向），stage 间通过 NCCL send/recv 传递 hidden states
- 调度为纯前向流水线，无 warmup/cooldown 阶段的复杂度
- 代码: `vllm/worker/pipeline_parallel_worker.py`

### 真正的 1F1B 调度（非教学版）

```
Interleaved 1F1B（Megatron-LM 中）:
  把每个 stage 进一步切成 m 个虚拟 stage，使 bubble ratio 从
  (p-1)/(n+p-1) 降至 (p-1)/(mn+p-1)

  代价：每个 microbatch 需要额外 p-1 次 pipeline 通信（延迟增加）
```

---

## 6. 运行

```bash
cd advanced/adv06_pipeline_parallel
python run.py
```

预期输出：

```
==============================================================
  Pipeline Parallel 调度对比 (stages=4, microbatches=4)
==============================================================

【GPipe 调度】
  串行模拟总时间          : ...
  理论峰值驻留 mb 数       : 4  （= n，全部 microbatch 同驻）

【1F1B 调度】
  串行模拟总时间          : ...
  理论峰值驻留 mb 数       : 3  （= p-1，仅 warmup depth 个 mb）

  显存峰值对比  GPipe=4 mb  vs  1F1B=3 mb
  1F1B 节省峰值激活显存    : 25.0%

...
✅ adv06_pipeline_parallel 通过
```

---

## 7. 下一步

本章演示了**层间切分**（Pipeline Parallelism）的显存气泡优化思路。

实践中 PP 与 TP（Tensor Parallelism，step17）常组合使用：
- **TP**：同一层的权重矩阵拆到多卡（层内并行，每层 1 次 all_reduce）
- **PP**：不同层拆到多卡（层间并行，显存线性缩减，有 pipeline bubble）
- **3D 并行** = DP + TP + PP（如 Megatron-Turing NLG 530B 的训练方案）

**下一步（adv07 — Sequence Parallel）**：

在极长序列（如 context=128k tokens）场景下，即使 TP 分担了权重显存，
LayerNorm / Dropout 等操作仍会产生 `[batch, seq, hidden]` 量级的激活显存瓶颈。
Sequence Parallelism（SP）把 seq 维度也切分到多卡，与 TP 的列并行/行并行配合，
进一步压缩激活显存峰值。
