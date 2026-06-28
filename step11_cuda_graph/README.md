# Step 11: CUDA Graph — 消除 Decode 阶段的调度开销

## 本节目标

理解 CUDA Graph 的工作原理，学习如何用它消除 decode 阶段每步推理的 Python 调度开销，并理解为什么这个优化只对 decode 有效、对 prefill 无效。

---

## 为什么 Decode 阶段有调度开销？

在 decode 阶段，模型每步只处理 **1 个新 token**（seq_len=1）。计算量极小，每步的绝大部分时间都花在"准备执行"上，而不是"实际计算"上。

具体来说，每次调用模型前向时，CPU 要走完这条链路：

```
Python 解释器（逐行执行 forward()）
    ↓
PyTorch dispatcher（为每个算子做类型推断、设备检查）
    ↓
CUDA driver（将 kernel launch 命令写入 command buffer）
    ↓
GPU 实际执行 kernel
```

这条链路中，**Python 解释器 → PyTorch dispatcher → CUDA driver** 这三段全部在 CPU 上串行执行。在 decode 阶段，GPU 上真正的矩阵乘法只需要几微秒，但 CPU 准备这次调用却需要几毫秒——GPU 大部分时间都在等 CPU 把下一批 kernel 提交过来。

这种现象叫做 **CPU-bound launch overhead**，是 decode 阶段延迟的主要来源之一。

---

## CUDA Graph 录制了什么？

CUDA Graph 是 CUDA 提供的一种机制，允许把一组 CUDA 操作（kernel launch、内存拷贝、同步等）预先录制成一张"图"，之后每次执行只需要提交这张图——**不再重新走 Python→PyTorch→driver 的链路**。

```
录制阶段（只做一次）：

  Python forward() 正常执行一遍
  CUDA driver 不立即执行 kernel，而是把所有 kernel launch 记录下来
  ↓
  形成一张 CUDAGraph（包含 kernel 序列、依赖关系）

                ┌─────────────────────────────────┐
                │  CUDAGraph                      │
                │  [kernel_1] → [kernel_2] → ...  │
                │  (形状、指针地址已固定)          │
                └─────────────────────────────────┘

Replay 阶段（每次推理）：

  g.replay()
  ↓
  CUDA driver 直接重放这张图，跳过 Python 和 PyTorch 层
```

**关键限制**：CUDAGraph 录制的是具体的 kernel 调用，包括 tensor 的内存地址和形状。replay 时这些都不能变。这意味着：
- 输入/输出 tensor 的 **shape 必须固定**
- 必须使用**同一块内存**（不能换新 tensor，只能原地修改内容）
- 不能有依赖 tensor 值的动态控制流

---

## 为什么 Prefill 不能用 CUDA Graph？

Prefill 阶段处理的是用户输入的 prompt，每个请求的长度不同（seq_len 可能是 10、100、1000……）。

CUDA Graph 要求录制和 replay 时 **形状完全一致**。如果 prefill 的 seq_len 每次都不一样，就需要为每种可能的长度都录制一张图——这在实际中不可行（seq_len 的取值空间太大）。

因此：
- **Prefill**：形状不固定 → 不能用 CUDA Graph → 走普通 PyTorch 路径
- **Decode**：每步 seq_len=1，形状固定 → 可以用 CUDA Graph → 每种 batch size 录制一张图

```
                Prefill               Decode
seq_len:      10 / 100 / 1000 ...       1（固定）
batch_size:   动态                    1, 2, 4, 8 ...（有限）
能用图吗？      ✗                         ✓
```

---

## 静态缓冲区的必要性

由于 CUDAGraph replay 时内存地址不能变，必须提前分配好**静态缓冲区**，让模型的输入和输出始终写到同一块内存上。

```python
# 录制前分配静态缓冲区（内存地址固定）
static_input_ids  = torch.zeros(batch_size, 1, dtype=torch.long, device="cuda")
static_positions  = torch.zeros(batch_size, 1, dtype=torch.long, device="cuda")
# KV cache 也必须是静态分配的同一块内存

# 录制
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input_ids, static_positions, static_kv_cache)

# 推理时：修改静态缓冲区的内容，不能换新 tensor
static_input_ids.copy_(new_token_ids)   # 原地修改，地址不变
static_positions.copy_(new_positions)
g.replay()                               # 重放，使用新内容
result = static_output.clone()           # 把结果拷贝出来
```

如果不使用静态缓冲区，每次推理都会生成新的 tensor（新地址），CUDAGraph replay 时仍然使用录制时的旧地址，读到的是旧数据——结果错误。

---

## Warm-up 的必要性

在录制 CUDA Graph 之前，必须先进行几次普通的前向（warm-up）：

```python
# warm-up：让 CUDA 分配好所有内存、完成 JIT 编译
for _ in range(3):
    _ = model(static_input_ids, static_positions, static_kv_cache)

# 然后再录制
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input_ids, static_positions, static_kv_cache)
```

原因有两个：

1. **内存分配**：PyTorch 的内存分配器在第一次运行时会申请 CUDA 内存。如果在图录制期间发生内存分配，这个分配操作会被录制进图里，但 replay 时分配器的状态已经不同，会导致错误。warm-up 让所有内存在录制前就已经分配好。

2. **CUDA JIT 编译**：部分 kernel（如 flash attention 的 triton 实现）在第一次运行时会做 JIT 编译。这个编译过程不应该被录制进图里，warm-up 确保编译在录制前完成。

---

## 多 Batch Size 的处理方式

实际系统中，decode 阶段的 batch size 不是固定的——同时在处理的请求数量会随时变化（1、2、4……）。

解决方式是**为每种 batch size 分别录制一张图**：

```
录制阶段（启动时执行一次）：

  batch_size=1  → 录制 graph_1
  batch_size=2  → 录制 graph_2
  batch_size=4  → 录制 graph_4
  batch_size=8  → 录制 graph_8
  ...

推理阶段：

  实际请求数=3 → 找到 ≥3 的最小录制尺寸（4）→ replay graph_4
               → 多余的 slot 用 padding 填充，结果取前 3 个
```

vLLM 使用 2 的幂次作为录制的 batch size 集合，以覆盖常见情况并控制录制的总内存占用。

```
batch size: 1  2  4  8  16  32  ...
录制图数量: 各一张（内存中同时保留）
总开销:     显存中保留多份静态缓冲区 + 图元数据
```

---

## 权衡与代价

CUDA Graph 不是免费的：

| 收益 | 代价 |
|------|------|
| 消除每步的 Python/PyTorch/driver 调度开销 | 启动时需要录制，增加初始化时间 |
| decode 延迟显著降低（特别是小 batch） | 每种 batch size 需要一份静态缓冲区，增加显存占用 |
| GPU 利用率提高（减少等待 CPU 的时间） | 不能有动态控制流，限制了模型的灵活性 |
| 减少 CPU-GPU 同步次数 | 调试更难（graph 内部的错误难以定位） |

**什么情况下收益最大**：计算量很小（decode 小 batch）、kernel 数量多（深层模型）、GPU 很快而 CPU 跟不上。

**什么情况下收益有限**：prefill 阶段（计算量大，调度开销占比小）、batch size 很大（GPU 本身已经满载）。

---

## 完整流程示意

```
启动阶段
  ├─ 分配静态 KV cache
  ├─ 分配各 batch size 的静态输入缓冲区
  ├─ warm-up（每种 batch size 跑 3 次前向）
  └─ 录制 CUDA Graph（每种 batch size 一张）

推理阶段（每步 decode）
  ├─ 确定当前 batch size → 选择对应的 graph
  ├─ 把新 token ids 原地写入静态缓冲区
  ├─ g.replay()  ← 这一步替代了整个 model.forward()
  └─ 从静态输出缓冲区读取 logits
```

---

## 运行

```bash
python run.py
```

无 GPU 时自动跳过，打印说明信息。

有 GPU 且有模型权重时，会演示 CUDA Graph 的录制与 replay 流程。

---

## 下一步

到这一步，我们已经优化了 decode 阶段的调度延迟。下一步（step12）将引入 **Tensor Parallelism**：把模型权重切分到多张 GPU 上，通过列并行与行并行让单卡装不下的大模型也能高效推理。
