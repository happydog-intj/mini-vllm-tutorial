# adv07: Sequence Parallel（序列并行）

## 1. 教学目标

- 理解 **Sequence Parallel（SP）** 的动机：当序列很长时，TP 切隐藏维已不够，激活在序列维仍然很大
- 掌握 SP 的核心思路：把序列维切到多卡，配合 **AllGather / ReduceScatter** 通信原语完成正确计算
- 通过可运行的单进程模拟器，直观感受 Q 切分、全量 KV 计算、分片拼回的流程
- 对比教学版与 Megatron-LM SP / Ring Attention / DeepSpeed Ulysses 的差异

---

## 2. 问题：为什么 TP 不够用？

### TP（Tensor Parallelism）的残留瓶颈

TP 把权重的**隐藏维**切分到多卡，但激活张量仍是完整的序列维：

```
激活形状：[batch, seq, hidden]

TP 切分后：
  - 权重：hidden → hidden/tp_size  ← 节省了权重显存
  - 激活：seq 维不变               ← 每卡仍持有完整序列激活
```

### 长序列时的激活瓶颈

以 LLaMA-70B 为例，hidden=8192，seq=32768，batch=1，fp16：

```
LayerNorm 输入激活：32768 × 8192 × 2 bytes ≈ 512 MB（每卡都要存）
```

即使 TP=8，LayerNorm 的激活**不能**用 TP 切，因为 LayerNorm 需要看到完整的 hidden 维。
激活显存没有随 tp_size 缩减，长序列时成为瓶颈。

### SP 的解决思路

把序列维也切分：每卡只持有 `seq / sp_size` 行激活。

```
SP 切分后每卡激活：
  [seq/sp_size, hidden] ← 激活显存线性缩减 sp_size 倍
```

代价：LayerNorm（需要完整 hidden）和 Attention（需要完整序列或额外通信）
需要在计算前后插入 AllGather / ReduceScatter 通信。

---

## 3. 原理：序列维切分 + AllGather / ReduceScatter

### 整体数据流（Megatron-LM SP 风格）

```
各卡：本卡激活 [seq/N, hidden]
        │
        │  AllGather（序列维拼回）
        ▼
各卡：完整激活 [seq, hidden]
        │
        │  LayerNorm（需要完整 hidden，现在序列也完整了）
        ▼
各卡：完整归一化激活 [seq, hidden]
        │
        │  ColumnParallel QKV 投影（TP，hidden 维切分）
        ▼
各卡：QKV 分片 [seq, hidden/N]
        │
        │  Attention（每卡只算本头）
        ▼
各卡：Attn 输出分片 [seq, hidden/N]
        │
        │  RowParallel O_proj → ReduceScatter（hidden 规约 + seq 切分）
        ▼
各卡：本卡激活 [seq/N, hidden]  ← 回到 SP 状态，进入下一层
```

### AllGather 示意

```
卡 0: [A]        AllGather       卡 0: [A, B]
卡 1: [B]     ─────────────►     卡 1: [A, B]
```

### ReduceScatter 示意

```
卡 0: [X]        Reduce          卡 0: [X+X']
卡 1: [X']    ─────────────►     卡 1: [X+X']

                Scatter
             ─────────────►   卡 0: [(X+X') 的前半段]
                              卡 1: [(X+X') 的后半段]
```

### 通信量分析

| 通信原语      | 方向       | 通信量（per layer）           |
|-------------|------------|-------------------------------|
| AllGather   | seq 维展开  | `(sp-1)/sp × seq × hidden × 2` |
| ReduceScatter | seq 维收缩 | `(sp-1)/sp × seq × hidden × 2` |

与 TP（每层 2 次 AllReduce）总量相当，但**激活显存**从 `[seq, hidden]` 降至 `[seq/sp, hidden]`。

### ❓ Q1：Q 切分后 K/V 保持完整就等价了，那 SP 的意义在哪？

**问题**：如果 K/V 每卡都要完整的一份，那不是每卡都存了 `[seq, hidden]` 的激活？SP 省显存的效果哪来的？

**答案**：好问题！这里需要区分**计算时**和**存储时**：

```
计算时（LayerNorm 后）：
  每卡通过 AllGather 拿到完整 [seq, hidden] → 做 LayerNorm → 做 QKV 投影

存储时（进入下一层前）：
  ReduceScatter 后，每卡只持有 [seq/sp, hidden] → 激活显存确实缩减了！

关键点：K/V 的"完整"是指**注意力计算需要看到全部序列**，
但 K/V 本身在各卡上也是分片的——计算某卡的 Q 分片时，
只需要通过 AllGather 临时凑齐 K/V（计算完就释放）。
所以**峰值激活存储**仍然是 [seq/sp, hidden]。
```

### ❓ Q2：`all_gather` 用 `[shard] * N` 模拟合理吗？

**问题**：真实 AllGather 是 N 张卡各贡献不同的 shard，教学版用同一个 shard 复制 N 次，语义对吗？

**答案**：形状上对，内容上不对。但这不影响**演示目的**：

```python
# 教学版：all_gather([A,B]) → [A,B,A,B]  （复制同一份）
# 真实版：all_gather(rank0 的 [A,B]) → [A,B,C,D]  （各卡贡献不同部分）

# 但对于验证形状和通信量来说：
#  - 输出形状 = local_size × N ✓
#  - 通信量 = (N-1)/N × total_size ✓（每卡发送自己的部分）

教学版省略了跨进程通信，只演示"形状变换"和"为什么需要 AllGather"。
```

### ❓ Q3：SP 和 TP 能同时用吗？

**问题**：SP 切序列维，TP 切隐藏维，会不会冲突？

**答案**：**可以且经常组合使用**（Megatron-LM 的 3D 并行 = DP + TP + SP）：

```
SP=4, TP=2 时：
  激活被切成 [seq/4, hidden/2] 每卡
  Attention 计算：SP 负责序列通信，TP 负责 hidden 维通信
  通信复用：Megatron 把 TP 的 AllReduce 拆成 AllGather+ReduceScatter，
           与 SP 的通信完全复用，不增加额外通信量
```

关键是 SP 和 TP 使用**同一进程组**——SP 的通信就嵌入在 TP 的通信原语中，不额外增加带宽消耗。

### ❓ Q4：为什么每个卡的 Q 分片仍需看到全部 K/V 才能正确计算？

**问题**：SP 把 Q 切到各卡了，为什么 K/V 不能也只看本段的，非要 AllGather 拿全部 K/V？

**答案**：因为 attention 的计算公式决定了这一点：

```
out[i] = softmax(q[i] @ K.T / sqrt(d)) @ V
```

每一行 Q 的输出，需要和**所有位置的 K** 做点积算 score，再对**所有位置的 V** 做加权求和。因果 mask 只挡住"未来位置"（j > i），但 j ≤ i 的所有位置都必须参与计算：

```
Sequence Parallel（多卡并行处理同一个 prompt，seq=1024, sp=2）：

  卡 0 持有: Q[0:512]
  卡 1 持有: Q[512:1024]

  卡 0 算 out[256] 时：
    scores = q[256] @ K[0:256].T   ← 必须看到位置 0~256 的所有 K
    out[256] = softmax(scores) @ V[0:256]

  卡 1 算 out[768] 时：
    scores = q[768] @ K[0:768].T   ← 必须看到位置 0~768 的所有 K！
                          ↑
              这些 K 有一半在卡 0 上，必须通信拿过来
```

如果只看本段 KV（Q[512:1024] 只看 K[512:1024]），那 out[768] 就丢失了对位置 0~511 的注意力，结果是**数学错误**的。

### ❓ Q5：SP 和 Chunked Prefill 是什么关系？看起来都是"把序列切块"

**问题**：Chunked Prefill（基础章节）也是把长 prompt 切成 chunk 分别处理，SP 是不是它的分布式版本？

**答案**：不是。虽然都涉及"把序列切块"，但解决的问题和切分维度不同：

| | Chunked Prefill | Sequence Parallel |
|---|---|---|
| **目的** | 控制单次 prefill 计算量，避免长 prompt 阻塞 decode 请求 | 把激活显存分摊到多张卡，突破单卡显存瓶颈 |
| **切分位置** | 时间维度 — 同一张卡，把长 prompt 分多次处理 | 空间维度 — 同一时刻，把序列分到多张卡并行处理 |
| **是否分布式** | 单卡，不需要通信 | 多卡，需要 AllGather/ReduceScatter 通信 |
| **K/V 依赖** | 每个 chunk 只需前缀 KV（因果 mask 天然截断 + KV Cache 累积） | 每个卡的 Q 分片需看到**全部 K/V**（必须通信） |

核心区别在于 attention 计算时 KV 的来源：

```
Chunked Prefill（单卡，时间切分）：
  chunk 0 先算 Q[0:512]，产生 KV Cache[0:512]
  chunk 1 再算 Q[512:1024]，此时 KV Cache[0:512] 已经在本卡显存里
  → 不需要跨卡通信，KV 是在同一张卡上逐步积累的

Sequence Parallel（多卡，空间切分）：
  卡 0 和卡 1 **同时**算，卡 1 的 Q[512:1024] 需要卡 0 持有的 K[0:512]
  → 这些 K 物理上在另一张卡的显存里，必须 AllGather 通信拿过来
```

本质区别：Chunked Prefill 是**串行**处理各 chunk，前一个 chunk 的 KV 自然在本卡积累；SP 是**并行**处理各分片，但 attention 的全局依赖关系不会因为并行就消失，所以必须通信。

如果硬要类比，**Ring Attention** 更像是 Chunked Prefill 的分布式版本——它把 KV 也分块了，通过卡间环形传递 KV 分片，每次只处理一块 KV，累积出完整的 attention 结果。而 Megatron 的 SP 思路不同：它是把 TP 的 AllReduce 拆成 AllGather + ReduceScatter，顺便完成序列维的切分/拼接，本质是 TP 的延伸。

---

## 4. 实现细节

### `sp_attention`（改进版，数值等价于标准 attention）

```python
def sp_attention(q, k, v, seq_splits=2):
    chunk = (seq + seq_splits - 1) // seq_splits
    outs = []
    for s in range(seq_splits):
        lo, hi = s*chunk, min((s+1)*chunk, seq)
        qc = q[lo:hi]          # 本卡 Q 分片（序列维切分）
        # K、V 保持完整（模拟每卡已 AllGather 到完整 KV）
        scores = qc @ k.T / sqrt(d)
        outs.append(softmax(scores) @ v)
    return cat(outs, dim=0)
```

**为什么数值与标准 attention 完全一致？**

标准 attention 第 i 行输出：`out[i] = softmax(q[i] @ K.T / sqrt(d)) @ V`

每行只依赖 `q[i]`，与其他 Q 行无关。把 Q 切片后独立计算，cat 拼回，与整体计算**数学完全等价**。

**与 plan 原始代码的区别（改进说明）：**

原始 plan 中 `sp_attention` 让每段只看自己段内的 KV（`kc = k[lo:hi]`），
这样每段 Q 只能关注到自己段内的 KV，**数学上不等于标准 attention**，
无法通过 `torch.allclose` 断言。

本实现改为：Q 按段切分，但 K/V 保持完整（模拟每卡已完成 AllGather(KV)），
这样 `sp_attention` 与标准 attention **数值完全等价**，断言 `allclose` 有意义。

### `all_gather`

```python
def all_gather(local_shard, world_size=2, dim=0):
    return torch.cat([local_shard] * world_size, dim=dim)
```

教学版用 `cat` 模拟"world_size 张卡各有一个 shard，AllGather 后每卡都有完整张量"。
真实场景中每张卡的 shard 内容不同（是完整张量的不同切片），此处教学简化为所有卡持有相同分片。

### `reduce_scatter`

```python
def reduce_scatter(full, world_size=2, dim=0):
    reduced = full * world_size   # 模拟：world_size 卡各一份 full，对应位置求和
    shard_size = full.size(dim) // world_size
    return reduced.narrow(dim, 0, shard_size)   # rank 0 取前 1/world_size 段
```

教学版：
- **Reduce**：`world_size` 张卡各持有相同的 `full`，求和 = `full * world_size`
- **Scatter**：取 rank 0 的分片（前 `seq/world_size` 行）

真实场景中各卡持有的 `full` 内容各不相同（是行并行的部分结果），求和后的完整结果才是正确输出，再 Scatter 分配给各卡。

---

## 5. 教学版 vs 真实框架

| 对比维度           | 本教学版（adv07）                    | Megatron-LM SP               | DeepSpeed Ulysses / Ring Attention |
|------------------|--------------------------------------|------------------------------|------------------------------------|
| **运行环境**       | 单进程 CPU，串行循环模拟              | 多进程，NCCL 通信             | 多进程，NCCL / P2P 通信             |
| **SP 切分维度**    | 序列维（Q 切分演示）                 | 序列维（激活完整生命周期管理） | 序列维（head 切分 or ring-style）   |
| **AllGather**     | `cat([shard] * N)`（同一 shard）      | 真实跨卡 AllGather            | 真实跨卡 AllGather / P2P 环形通信  |
| **ReduceScatter** | `full * N` 取前段（简化语义）         | 真实跨卡 ReduceScatter        | 真实跨卡 ReduceScatter             |
| **Attention 正确性** | 数值等价标准 attention（改进版）   | 等价（通过通信补全 KV）        | 等价（Ring 传递 KV 分片）           |
| **通信重叠**       | 无（串行演示）                        | 计算-通信重叠                 | 计算-通信重叠                       |
| **适用规模**       | 教学理解                              | 万亿参数模型，超长序列         | 超长序列（>100K tokens）            |

### Megatron-LM SP 的关键设计

- SP 与 TP 联合使用（同一进程组），不增加通信量
- TP 的 `AllReduce` 拆成 `AllGather + ReduceScatter`，与 SP 通信复用
- 激活显存从 `[seq, hidden]` 降至 `[seq/N, hidden]`，使超长序列成为可能

### DeepSpeed Ulysses

- 把 Attention 的 Q/K/V 投影沿 **head 维** 切分到各卡
- 每卡计算自己负责的 head，所有序列位置的 Q/K/V 都参与
- AlltoAll 通信代替 AllGather/ReduceScatter，通信量更低（特别是 head 数多时）

### Ring Attention

- 每卡只持有 K/V 的 1/N 分片，Q 的对应段
- 卡间以环形 P2P 传递 K/V，同时计算当前 block 的 attention
- 通信完全隐藏在计算中，理论通信开销为 0（计算时间 >> 通信时间时）

---

## 6. 运行

```bash
cd advanced/adv07_sequence_parallel
python run.py
```

预期输出：

```
========================================================
验证 1：sp_attention 与标准 attention 数值一致性
========================================================
  输入 q/k/v shape : [16, 32]
  sp_attention  out: [16, 32]
  standard_attn out: [16, 32]
  最大绝对误差      : < 1e-05
  ✅ sp_attention 与标准 attention 数值 allclose

========================================================
验证 2：all_gather 形状（沿 dim=0 拼接）
========================================================
  local_shard shape : [8, 32]
  all_gather   out  : [16, 32]
  ✅ all_gather 形状正确，前后分片内容一致

========================================================
验证 3：reduce_scatter 形状（沿 dim=0 切分）
========================================================
  full_tensor shape : [16, 32]
  reduce_scatter out: [8, 32]
  ✅ reduce_scatter 形状正确

========================================================
验证 4：reduce_scatter 数值语义
========================================================
  rank-0 shard == full * 2 的前 8 行
  ✅ reduce_scatter 数值语义正确

✅ adv07_sequence_parallel 通过
```

---

## 7. 下一步

**adv08: Data Parallel + Dynamic Load Balancing（DPLB）**

本步骤（SP）解决了**单机长序列激活显存**的问题。
下一步 adv08 在多副本 Data Parallel 的基础上引入**动态负载均衡**：

- 不同请求序列长度差异大时，简单 DP 导致卡间负载不均
- DPLB 动态重新分配 batch 到各卡，消除尾效应（tail latency）
- 与 SP 正交，可以联合使用：SP 处理单卡长序列，DPLB 均衡跨卡吞吐
