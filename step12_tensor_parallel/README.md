# Step 12: Tensor Parallelism — 列并行与行并行

## 为什么需要 Tensor Parallelism

**问题一：模型太大，装不进单卡显存。**

以 LLaMA-70B 为例，仅模型权重（fp16）就需要约 140 GB 显存，而主流单卡（如 A100 80GB）远远不够。Pipeline Parallelism 是一种解法：把不同层放在不同 GPU 上。但它引入了"气泡"（某些 GPU 空闲等待），且延迟随流水线深度线性增加。

**问题二：即使装得进，单卡吞吐也可能是瓶颈。**

矩阵乘法（GEMM）是 Transformer 中最密集的计算。把一层的权重矩阵切分到多张 GPU 上并行计算，可以成倍提升单层的计算吞吐，而不需要把不同层拆到不同卡。

**Tensor Parallelism 的思路**：把单层的权重矩阵沿某个维度切分，每张 GPU 只持有一个分片，独立计算本地结果，在必要时用一次 `all_reduce` 汇总。这样多卡就像一张更大的"虚拟 GPU"。

---

## 核心概念

### 列并行（ColumnParallelLinear）

**切分方向**：沿权重矩阵的输出维度（列方向）切分。

```
原始权重 W: [in_features × out_features]

切分为 tp_size 份：
  GPU 0: W0  [in_features × (out_features/tp_size)]
  GPU 1: W1  [in_features × (out_features/tp_size)]

每张 GPU 拿到完整输入 X [batch, seq, in_features]，独立计算：
  GPU 0: Y0 = X @ W0.T   shape: [batch, seq, out_features/tp_size]
  GPU 1: Y1 = X @ W1.T   shape: [batch, seq, out_features/tp_size]

结果在逻辑上拼接（concat）：
  Y = [Y0 | Y1]          shape: [batch, seq, out_features]
```

**关键性质**：每张 GPU 的计算完全独立，**无需任何通信**。Y0 和 Y1 之间没有数据依赖。

**为什么 Q/K/V 投影适合列并行？**

注意力机制中，`Q = X @ W_Q`，最终会被切分成多个 head：`Q = [Q_head0 | Q_head1 | ...]`。每个 head 的计算彼此独立。列并行天然对应这种切分——每张 GPU 负责一部分 head，各 GPU 上的注意力计算互不干扰。

### 行并行（RowParallelLinear）

**切分方向**：沿权重矩阵的输入维度（行方向）切分。

```
原始权重 W: [in_features × out_features]

切分为 tp_size 份：
  GPU 0: W0  [(in_features/tp_size) × out_features]
  GPU 1: W1  [(in_features/tp_size) × out_features]

输入 X 也对应切分（来自上一层列并行的输出）：
  GPU 0: X0  [batch, seq, in_features/tp_size]
  GPU 1: X1  [batch, seq, in_features/tp_size]

每张 GPU 计算部分结果：
  GPU 0: Y0 = X0 @ W0.T   shape: [batch, seq, out_features]
  GPU 1: Y1 = X1 @ W1.T   shape: [batch, seq, out_features]

Y0 + Y1 只是最终结果的一半，必须汇总：
  all_reduce(SUM): Y = Y0 + Y1   shape: [batch, seq, out_features]
```

**为什么需要 all_reduce？**

行并行的数学基础是矩阵乘法的分块性质：

```
X @ W.T = [X0 | X1] @ [W0; W1].T = X0 @ W0.T + X1 @ W1.T
```

两个 GPU 各自算出了完整输出的一个"加法分量"，必须相加才能得到正确结果。`all_reduce(SUM)` 做的正是这件事——每张 GPU 最终都持有正确的完整输出。

**为什么 O_proj（输出投影）适合行并行？**

注意力输出 `AttnOut = [head0_out | head1_out | ...]` 正好是按 head 拼接的，天然与行并行的输入切分对应：每张 GPU 的激活恰好是对应 head 的输出，直接喂入行并行层即可，无需额外通信来重组激活。

### 标准搭配：列并行 + 行并行 = 一次 all_reduce

```
输入 X（每张 GPU 持有相同副本）
        │
        ▼
[ColumnParallelLinear]   ← 无通信，各 GPU 独立计算
        │
        │  各 GPU: 中间激活（out_features/tp_size）
        ▼
[RowParallelLinear]      ← 前向结束时一次 all_reduce
        │
        ▼
输出 Y（每张 GPU 持有相同的完整结果）
```

整个 Attention 子层（Q/K/V 投影 + O 投影）或 MLP 子层（gate/up 投影 + down 投影）只需 **1 次 all_reduce**，通信与计算的比例极低。

---

## 通信量分析

一次 all_reduce 的通信量：

```
通信量 = 2 × (tp_size - 1) / tp_size × tensor_size
```

对于 `[batch, seq, hidden_size]` 的张量，假设 batch=8、seq=2048、hidden_size=8192（fp16）：

```
单次通信量 ≈ 8 × 2048 × 8192 × 2 bytes ≈ 256 MB
```

每个 Transformer 层有 2 次 all_reduce（Attention 子层 1 次，MLP 子层 1 次）。

**NVLink vs PCIe 的影响**：
- **NVLink**（如 A100/H100 机内互联）：带宽可达数百 GB/s，256 MB 的 all_reduce 耗时仅约 1 ms 量级，相对于 GEMM 计算时间可以忽略不计。
- **PCIe**（跨机或无 NVLink 的卡间）：带宽约 16–32 GB/s，同等通信量耗时可达数十 ms，通信开销不可忽视，需要配合 overlap 技术（通信与计算重叠）缓解。

因此，Tensor Parallelism 最适合 **同一台机器内有高速互联（NVLink）的多 GPU**，跨机使用时收益会被通信抵消。

---

## 代码说明

### `linear.py`

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size=1, ...):
        self.out_features_per_rank = out_features // tp_size
        # 每张 GPU 只分配 out_features/tp_size 列的权重
        self.weight = nn.Parameter(torch.randn(self.out_features_per_rank, in_features))

    def forward(self, x):
        return x @ self.weight.T  # 无通信，直接计算
```

`ColumnParallelLinear` 的每个实例只持有完整权重的 `1/tp_size`。在真实多 GPU 环境中，每张 GPU 初始化时从完整权重中取对应的分片（通常由模型加载逻辑完成）。

```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size=1, ...):
        self.in_features_per_rank = in_features // tp_size
        # 每张 GPU 只分配 in_features/tp_size 行的权重
        self.weight = nn.Parameter(torch.randn(out_features, self.in_features_per_rank))

    def forward(self, x):
        out = x @ self.weight.T
        if self.tp_size > 1 and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)  # 关键：汇总各 GPU 的部分结果
        return out
```

`all_reduce` 只在 `tp_size > 1` 且分布式环境已初始化时触发，单 GPU 运行时自动跳过，代码兼容两种场景。

### `run.py`

`run.py` 在 `tp_size=1` 下验证两个并行层与标准 `nn.Linear` 的数值等价性：

```
Tensor Parallelism 线性层验证
=============================================
ColumnParallelLinear (tp_size=1): torch.Size([4, 16, 256]) → torch.Size([4, 16, 512])  ✅
RowParallelLinear (tp_size=1):    torch.Size([4, 16, 512]) → torch.Size([4, 16, 256])  ✅

切分策略说明 (tp_size=2 时):
  ColumnParallel: weight [256×512] → [256×256] × 2 GPU
  RowParallel:    weight [512×256] → [256×256] × 2 GPU
    + all_reduce 跨 GPU 求和
```

---

## 文件说明

| 文件 | 功能 |
|------|------|
| `linear.py` | `ColumnParallelLinear` + `RowParallelLinear` 实现 |
| `run.py` | `tp_size=1` 单机数值验证（与标准 `nn.Linear` 对比） |

## 运行

```bash
python run.py
```

多 GPU 测试需要用 `torchrun` 启动并初始化 `torch.distributed`：

```bash
# 2 GPU 示例（需要有实际的多 GPU 环境）
torchrun --nproc_per_node=2 run_dist.py
```

在多 GPU 环境中，每张 GPU 上的进程独立运行 forward，`all_reduce` 调用会自动在进程组内同步。

---

## 权衡与局限

| 方面 | 说明 |
|------|------|
| **通信开销** | 每层 2 次 all_reduce；NVLink 下几乎可忽略，PCIe 下不可忽略 |
| **显存节省** | 权重显存线性缩减（2 GPU → 各持一半权重），但激活显存不变 |
| **tp_size 限制** | `out_features` 必须能整除 `tp_size`；通常 tp_size ≤ 8（单机 NVLink 域内） |
| **与其他并行的组合** | 实践中常与 Pipeline Parallelism 结合（TP 做层内，PP 做层间） |
| **代码侵入性** | 需要修改每个线性层的定义，比 Data Parallelism 侵入性更高 |

---

## 下一步

本步骤实现了单层的张量并行组件。下一步（step13）将引入 **Benchmark**：用量化的指标（吞吐量、延迟、首 token 时间）评测各项优化的实际效果，建立性能基线。
