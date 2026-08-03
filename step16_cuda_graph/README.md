# Step 16: CUDA Graph — 消除 Decode 阶段的调度开销

## OS 类比：批处理 vs 直接执行

CUDA Graph 解决的问题，在操作系统里有一个经典的对应：**系统调用开销**。

每次用户程序调用 `read()`、`write()` 这类系统调用时，都要经历：

```
用户态代码
    ↓ 陷入内核（syscall）
内核处理请求
    ↓ 返回用户态
用户态继续
```

这个用户态 → 内核 → 用户态的来回本身有固定开销（约 1~10μs）。如果程序每次只读 1 个字节就 syscall 一次，大部分时间都花在切换上，而不是真正的 I/O。

操作系统的解决方案是 **io_uring**（Linux 5.1+）：

```
传统方式（每次 I/O 一次 syscall）：
  read() → syscall → read() → syscall → read() → syscall ...
  大量时间花在用户/内核态切换

io_uring（批量提交）：
  把 N 个 I/O 请求写入共享内存的 submission queue
  一次 syscall 批量提交 → 内核批量执行 → 结果写回 completion queue
  切换次数从 N 次降为 1 次
```

**CUDA Graph 做的是完全一样的事，只是换了一层**：

| | io_uring | CUDA Graph |
|---|---|---|
| 问题 | 每次 I/O 都有 syscall 开销 | 每次 kernel 都有 CPU 调度开销 |
| 解决方案 | 把 I/O 操作写入 submission queue，批量提交 | 把 kernel launch 序列录制成图，一次提交 |
| 节省的开销 | 用户/内核态切换 | Python → PyTorch → CUDA driver 链路 |
| 适用场景 | I/O 密集型小请求 | Decode（每步只有 1 个 token，计算极小）|
| 不适用场景 | 大块 I/O（本身开销相对小）| Prefill（计算量大，launch 开销占比小）|

另一个更直接的类比是 **DMA（直接内存访问）**：CPU 配置一次 DMA 传输参数后，硬件独立搬运数据，CPU 不再逐字节介入——GPU replay CUDA Graph 时，CPU 提交一次图之后，GPU 按录制的顺序独立执行所有 kernel，CPU 不再逐 kernel 介入。

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

### 两个核心 API:`torch.zeros` 与 `tensor.copy_`

上面这段代码里有两个看似普通、实则承担关键作用的 API。理解它们，才能理解"静态缓冲区"为什么这样写。

#### `torch.zeros`:预分配一块"地址固定、形状固定"的显存

```python
static_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device="cuda")
```

`torch.zeros` 在这里的用途**不是**"我要一个全零的 tensor"，而是**"我要在 CUDA 上申请一块固定形状、固定 dtype、固定地址的显存"**。全零只是它顺带填的初始值,真正被利用的是以下三点:

1. **地址固定**:`torch.zeros` 一次性向 PyTorch 的 CUDA 缓存分配器（caching allocator）申请一块显存。只要这个 tensor 对象不被回收、不被重新分配,它在 GPU 上的物理地址在整个推理过程中保持不变。这正是 CUDAGraph replay 所要求的——图里录制的指针必须始终指向有效数据。

2. **形状固定**:tensor 的 shape 在创建时就钉死为 `(batch_size, 1)`。CUDAGraph 录制的是"对这块形状的显存做这些 kernel",replay 时形状不能变,所以缓冲区形状必须从一开始就和 decode 阶段的形状一致(decode 每步 seq_len=1)。

3. **dtype 与 device 固定**:图里的 kernel 是针对特定数据类型(int64 的 token ids)在特定设备(cuda)上编译/录制的。`torch.zeros` 用 `dtype=torch.long, device="cuda"` 把这两点钉死。

换句话说,`torch.zeros` 充当的是一块**"地址永不变、形状永不变"的模板槽位**——图录好后,这个槽位就是模型输入的固定"插口"。

> 为什么不用 `torch.empty`?`torch.empty` 同样能申请固定显存且更省一次清零,但 decode 的 token id 输入语义上是"待填的真实数据",用 `zeros` 让"初始为无效 token 0"的含义更清晰,且规避了未初始化显存可能带来的 NaN/调试困难。两者在"固定地址"这个核心诉求上等价。

#### `tensor.copy_`:原地改写内容,地址不变

```python
static_input_ids.copy_(new_token_ids)   # 原地修改,地址不变
```

`.copy_(src)` 是 PyTensor 的**原地拷贝**:`src` 的数据被写入 `self` 这块显存,而 `self` 的**地址、形状、dtype 都不变**。这正是它相对 `static_input_ids = new_token_ids` 这种"赋值替换"的关键区别:

| 写法 | 数据地址 | CUDAGraph replay 能否读到新数据 |
|------|---------|------------------------------|
| `static_input_ids = new_token_ids` | **变了**(指向新 tensor 的新地址) | ✗ 图里录的是旧地址,读到旧数据 |
| `static_input_ids.copy_(new_token_ids)` | **不变**(仍是原 buffer) | ✓ 图里的指针指向同一块内存,读到刚写入的新内容 |

`copy_` 在底层会发射一个 CUDA memcpy kernel(或 memcpy_d2d)。这个 kernel **不在图里**——它发生在 `g.replay()` 之前的正常 PyTorch 路径上。replay 时,模型 kernel 读到的就是这块 buffer 里**最新被 copy_ 写入**的内容,从而实现了"换数据不换地址"。

#### 二者如何配合

```
torch.zeros          →  钉死"插口"的地址/形状/dtype(图录的就是这个插口)
       │
       ▼
[ 录制 CUDAGraph:模型 kernel 指向这个插口 ]
       │
       ▼
copy_(new_data)      →  把新 token 数据写进同一个插口(地址没变)
       │
       ▼
g.replay()           →  模型按录制的指针读这块内存 → 读到的是新数据 ✓
```

一句话总结:`torch.zeros` 负责把"插口"的位置钉死,`copy_` 负责"插口"里换内容而不挪位置——两者共同满足 CUDAGraph "形状、地址都不能变,只有内容可变"的硬性约束。

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

实际系统中，decode 阶段的 batch size 不是固定的——同时在处理的请求数量会随时变化（1、2、4……）。要理解为什么这件事在 decode 阶段尤其棘手，得先看 decode 的一个根本计算特性。

### Decode 为什么对 batch size 如此敏感

Prefill 和 decode 的**计算瓶颈完全不同**，这决定了 batch size 对两者的意义天差地别：

| | Prefill | Decode |
|---|---------|--------|
| 每步算什么 | 一次性处理整段 prompt(seq_len 可能上千) | 每步只算 1 个新 token |
| 瓶颈类型 | **Compute-bound** | **Memory-bound** |
| 瓶颈在哪 | 矩阵乘法的算术吞吐 | 反复从显存把权重搬进 SRAM 的带宽 |

Decode 每步只算 1 个 token,矩阵乘法退化为"权重矩阵 × 一个向量"——算术量极小,但**整份模型权重都要从显存(HBM)读进 SRAM 一次**。GPU 的算术单元远远跑不满,真正卡住的是显存带宽。

这带来一个关键事实:decode 是 **memory-bound** 的。把 1 个请求和把 8 个请求塞进同一次 forward,**权重只需从显存搬一次**(8 个请求共用同一份权重),算术量只增加 8 倍,但权重搬运量**不变**。

```
batch=1 decode:  搬权重(1次) + 算1个token    → 带宽几乎全浪费在搬权重上
batch=8 decode:  搬权重(1次) + 算8个token    → 同样搬一次权重,算术量×8,带宽利用率×8

  ↓ 结论:decode 的吞吐几乎随 batch size 线性增长,直到带宽被算术填满
```

这就是为什么 decode 阶段"把更多请求塞进同一次 forward"价值巨大——它几乎是在**白嫖本要被浪费的显存带宽**。也正因为如此,decode 阶段对"当前 batch size 是几"非常敏感:它直接决定这步 forward 的吞吐。

### 核心矛盾:动态 batch size ↔ 固定形状

现在两个前面章节建立的事实撞到了一起:

```
事实 A (来自 step09 Continuous Batching):
  每完成一个 decode 步就检查哪些请求结束、补入新请求
  → 实际 batch size 随时在变:这一步 3 个,下一步 5 个,再下一步 2 个...

事实 B (来自本节 CUDA Graph):
  图录制时 tensor 的 shape 必须钉死,replay 时形状不能变
  → 一张图只能服务一个固定的 batch size
```

**矛盾**:Continuous Batching 让 batch size 成为每步都在跳变的量,而 CUDA Graph 恰恰要求形状固定。不可能"为当前 batch size 临时录一张图"——录制本身就要 warm-up + 跑一遍 forward,开销远大于它要消除的调度开销。

### 解决方案:为每种 batch size 预录一张图(Bucketing)

既然不能临录,就**启动时一次性为所有可能用到的 batch size 各录一张图**,推理时按当前请求数挑对应的那张 replay:

```
录制阶段(启动时执行一次):

  batch_size=1  → 录制 graph_1
  batch_size=2  → 录制 graph_2
  batch_size=4  → 录制 graph_4
  batch_size=8  → 录制 graph_8
  ...

推理阶段:

  实际请求数=3 → 找到 ≥3 的最小录制尺寸(4)→ replay graph_4
               → 多余的 slot 用 padding 填充,结果取前 3 个
```

这里有两个设计抉择值得展开:**为什么按 2 的幂取录制尺寸集合**,以及 **padding 的浪费到底有多大**。

#### 为什么是 2 的幂(power-of-2 bucketing)

如果对 1..N 每个 batch size 都录一张图,图数量是 N 张,显存里要常驻 N 份静态缓冲区——不可接受。如果只录一张(录最大尺寸),小 batch 时要 padding 到最大,浪费极大。

2 的幂是两者的平衡点:覆盖 `[1,2,4,8,16,32,...,N_max]`,图数量从 N 降到 `log₂(N_max)`。对最大 batch size 256 的系统,只需 9 张图。

```
batch size: 1  2  4  8  16  32  ...  256
录制图数量: 各一张(内存中同时保留)        → 共 9 张,而非 256 张
总开销:     显存中保留 9 份静态缓冲区 + 图元数据
```

#### Padding 浪费率:最坏 2 倍,平均 1.33 倍

任意 batch size `b` 都会被向上取整到最近的 2 的幂 `B = 2^⌈log₂ b⌉`,多余的 `B - b` 个 slot 填 padding。浪费率的边界很干净:

```
最坏情况:b = B/2 + 1(如 5 个请求 → 取到 8 的图)
  padding = B - b = 3,浪费 3/8 = 37.5%,实际计算/有效计算 ≈ 1.6 倍

理论最坏:b = B/2 + 1 → 浪费接近 50%,即最坏 ~2 倍计算量

平均情况:假设 b 在每个区间 [B/2, B] 内均匀分布
  平均 padding = B/4,平均浪费率 25%,平均 ~1.33 倍计算量
```

考虑到 CUDA Graph 带来的调度开销消除通常是数倍量级,padding 这点平均 1.33 倍的算术浪费(且 decode 本来就 memory-bound、算术单元闲着)是完全可以接受的——被 padding 的 slot 走的是同样那份权重的搬运,带宽本就要花。

#### vLLM 的实现:CUDAGraph tree

vLLM 把上述机制封装成 `CUDAGraphRunner`(俗称 cudagraph tree):启动时遍历预设的 batch size 集合,每个尺寸各做一次 warm-up + 录制 + 保存对应的静态输入/输出缓冲区;推理时根据当前 `num_seqs` 选图、`copy_` 写入真实数据、`replay()`、读出结果。对不在预设集合里的尺寸,默认按 2 的幂向上取整(也可配置为禁用 cudagraph、回退普通 forward)。

> 一个工程细节:静态缓冲区按"最大录制尺寸"分配一份,各尺寸的图共用其前缀——`graph_4` 用的就是 `graph_8` 缓冲区前 4 行的地址。这样显存只占一份最大缓冲区,而不是每个尺寸各一份。

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

到这一步，我们已经优化了 decode 阶段的调度延迟。下一步（Tensor Parallelism：多卡分布式推理）将引入 **Tensor Parallelism**：把模型权重切分到多张 GPU 上，通过列并行与行并行让单卡装不下的大模型也能高效推理。
