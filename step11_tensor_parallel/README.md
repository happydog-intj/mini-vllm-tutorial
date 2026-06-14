# Step 11: Tensor Parallelism — 列并行与行并行

## 本节目标

实现 Tensor Parallelism（张量并行）的核心组件：ColumnParallelLinear 和 RowParallelLinear。

## 核心概念

### 为什么需要 Tensor Parallelism

大模型（如 70B）的单层权重矩阵可能达到数 GB，超过单卡显存。
Tensor Parallelism 将权重矩阵按行或列切分到多张 GPU 上。

### 列并行（Column Parallel）

将输出维度（列）切分：

```
W [in × out]  →  [W1 | W2]  （每 GPU 持有 out/tp_size 列）

GPU0: Y1 = X @ W1.T   shape: [seq, out/tp]
GPU1: Y2 = X @ W2.T   shape: [seq, out/tp]

输出 concat: Y = [Y1 | Y2]   shape: [seq, out]
```

- 每 GPU 独立计算，**无需通信**
- 适用于：Q/K/V 投影、MLP gate/up 投影

### 行并行（Row Parallel）

将输入维度（行）切分：

```
W [in × out]  →  [W1; W2]  （每 GPU 持有 in/tp_size 行）

GPU0: Y1 = X1 @ W1.T   （X1 是 X 的前半部分）
GPU1: Y2 = X2 @ W2.T

最终: Y = Y1 + Y2   ← all_reduce 求和
```

- 需要一次 **all_reduce** 通信
- 适用于：O 投影、MLP down 投影

### 标准搭配

```
X → [ColumnParallel] → 中间激活（各 GPU 独立）→ [RowParallel] → all_reduce → 输出
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `linear.py` | ColumnParallelLinear + RowParallelLinear |
| `run.py` | tp_size=1 单机验证（与标准 Linear 对比） |

## 运行

```bash
python run.py
```

多 GPU 测试需要 `torchrun` 或 `torch.distributed` 初始化。
