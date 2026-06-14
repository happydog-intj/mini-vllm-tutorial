# Step 10: CUDA Graph — Decode 阶段延迟优化

## 本节目标

理解 CUDA Graph 原理，学习如何用它消除 decode 阶段的 Python/CUDA 调度 overhead。

## 核心概念

### Decode 阶段的性能瓶颈

在 decode 阶段，每步只处理 **1 个 token**（batch_size=1, seq_len=1），计算量极小。
但每次调用都会触发：

1. Python 解释器逐行执行模型前向
2. PyTorch dispatcher 为每个算子做类型检查
3. CUDA driver 向 GPU 提交 kernel launch 命令

这些 CPU-side overhead 可能占总延迟的 30-50%。

### CUDA Graph 解决方案

```python
# 1. 预热（让 CUDA 分配好所有内存）
for _ in range(3):
    output = model(input_ids, positions, past_kv)

# 2. 录制
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input_ids, positions, past_kv)

# 3. 推理时 replay（直接执行录制好的 kernel 序列）
input_ids.copy_(new_input_ids)  # 更新输入（原地修改）
g.replay()                       # 极低延迟
result = output.clone()
```

**关键约束**：
- 输入/输出 tensor 的 shape 和 dtype 必须固定
- 不能有动态控制流（if/while 取决于 tensor 值）
- 需要为每种 batch_size 单独录制

### 与 continuous batching 的关系

vLLM 对每个 decode batch size（1, 2, 4, 8, ...）分别录制 CUDA Graph，
推理时根据实际 batch size 选择对应的 graph 执行。

## 运行

```bash
python run.py
```

无 GPU 时自动跳过，打印说明信息。
