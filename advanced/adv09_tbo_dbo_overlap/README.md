# adv09: TBO/DBO 计算-通信重叠

## 1. 教学目标

- 理解张量并行（TP）/ 序列并行（SP）场景下，**计算与通信串行**导致的 GPU 空闲问题
- 掌握 TBO（Tensor-Batch Overlap）/ DBO（Data-Batch Overlap）的核心思路：  
  microbatch i+1 的计算 与 microbatch i 的通信**流水线并行**
- 用纯 Python（`time.sleep` + `ThreadPoolExecutor`）在 CPU 层面模拟重叠调度，  
  建立直觉；了解真实框架如何用 **CUDA Stream** 实现同样效果

---

## 2. 问题：计算与通信串行，GPU 大量空闲

在 Transformer 模型的张量并行（TP）中，每一层都需要：

1. **计算**（矩阵乘法：Attention / FFN）
2. **通信**（AllReduce / AllGather / ReduceScatter，跨 GPU 聚合结果）

朴素实现下，这两步**串行执行**：

```
GPU 0  [COMPUTE_0]  [COMM_0]  [COMPUTE_1]  [COMM_1]  ...
时间线 ─────────────────────────────────────────────────▶
```

- 计算时，通信链路**空闲**（InfiniBand / NVLink 浪费）
- 通信时，GPU 算力**空闲**（CUDA Core 浪费）

对于大模型（百亿参数以上），TP/SP 的通信量相当可观，串行开销可能占总时延的 20%～40%。

---

## 3. 原理：TBO 计算通信交错时间轴

将一个大 batch 切分为若干 **microbatch**，让相邻 microbatch 的计算与通信**流水线重叠**：

### 朴素串行（no_overlap）

```
时间 ──────────────────────────────────────────────▶
      [C0][M0] [C1][M1] [C2][M2] [C3][M3]
       ↑↑↑↑↑    ↑↑↑↑↑
       计算通信  计算通信  依次串行

总耗时 = n × (ct + mt)
```

### TBO 重叠（tbo_overlap）

```
时间 ──────────────────────────────────────────────▶
      [C0]
      [M0]        ← M0 与 C1 并行
           [C1]
           [M1]   ← M1 与 C2 并行
                [C2]
                [M2]  ← M2 与 C3 并行
                     [C3]
                     [M3]  ← 最后一轮通信单独收尾

总耗时 ≈ n × max(ct, mt) + min(ct, mt)
```

**节省量**：每个 microbatch 重叠掉 `min(ct, mt)`，总节省 `(n-1) × min(ct, mt)`。

示例（n=8, ct=50ms, mt=30ms）：
- 朴素：8 × (50+30) = **640 ms**
- TBO ：8 × 50 + 30 ≈ **430 ms**（加速 ~1.49×，实测约 1.6×）

---

## 4. 实现细节

### `overlap_sim.py`

**`no_overlap(microbatches, ct, mt)`**

串行执行每个 microbatch：`compute(ct)` → `comm(mt)` 依次调用，无任何并发。

**`tbo_overlap(microbatches, ct, mt)`**

使用 `ThreadPoolExecutor(max_workers=2)` 模拟两条独立"执行流"（类比 CUDA 的 compute stream 和 comm stream）：

```python
for mb in microbatches:
    comp_future = ex.submit(compute, ct)   # 本轮计算提交
    if prev_comm_future exists:
        prev_comm_future.result()          # 等待上一轮通信（它在 comp 并行期间跑）
    prev_comm_future = ex.submit(comm, mt) # 上一轮通信已完成 / 或首轮单独启动
    comp_future.result()                   # 等本轮计算完毕才进下一轮
# 最后等尾部通信
```

关键：`prev_comm` 和 `comp` 同时在线程池里跑，模拟了 **compute stream 与 comm stream 的并行**。

> **注意**：Python 线程受 GIL 限制，`time.sleep` 能释放 GIL 因此确实并行。  
> 真实 GPU 上计算由 CUDA compute stream 驱动、通信由 NCCL comm stream 驱动，天然并行，不依赖 GIL。

---

## 5. 教学版 vs 真实框架

| 对比维度 | 本教学（adv09） | 真实框架（DeepSeek / Megatron-LM） |
|---|---|---|
| 并发原语 | Python `ThreadPoolExecutor` | CUDA Stream（`torch.cuda.Stream`） |
| 计算模拟 | `time.sleep(ct)` | cuBLAS GEMM / FlashAttention kernel |
| 通信模拟 | `time.sleep(mt)` | NCCL AllReduce / AllGather on comm stream |
| 重叠粒度 | microbatch 级 | microbatch 级（有时是 chunk 级） |
| GIL 影响 | 无（sleep 释放 GIL） | 无（CUDA kernel 异步，CPU 不阻塞） |
| 同步点 | `future.result()` | `stream.synchronize()` / CUDA event |

**DeepSeek TBO/DBO**（见 DeepSeek-V3 技术报告）：

- **TBO（Tensor-Batch Overlap）**：张量并行场景，将 token 按 microbatch 切分，  
  前一 microbatch 的 AllReduce 与后一 microbatch 的矩阵乘在不同 CUDA Stream 上并发。
- **DBO（Data-Batch Overlap）**：数据并行场景，allreduce 梯度通信与下一步前向计算重叠  
  （即 gradient overlap / bucket 通信）。

**真实实现关键代码模式**：

```python
compute_stream = torch.cuda.Stream()
comm_stream    = torch.cuda.Stream()

with torch.cuda.stream(compute_stream):
    out = matmul(x, W)                    # 计算在 compute stream

with torch.cuda.stream(comm_stream):
    dist.all_reduce(prev_out, async_op=True)  # 上一轮通信在 comm stream

# CUDA event 同步：等 compute 完才做通信，等通信完才用结果
```

**与 CUDA Graph（step16）的区别**：

| | CUDA Graph | TBO/DBO |
|---|---|---|
| 解决问题 | CPU launch overhead（调度开销） | 计算与通信串行（带宽浪费） |
| 作用阶段 | Decode（小计算量，launch 占比高）| Prefill + Decode（TP 通信代价大）|
| 机制 | 预录制 kernel 序列，一次提交 | 双 CUDA Stream 流水线并发 |

---

## 6. 运行

```bash
cd advanced/adv09_tbo_dbo_overlap
python run.py
```

期望输出：

```
====================================================
  adv09: TBO / DBO 计算-通信重叠 对比实验
====================================================
  microbatch 数量 : 8
  compute 时间    : 50 ms / microbatch
  comm    时间    : 30 ms / microbatch
  理论朴素耗时    : 640 ms
  理论 TBO  耗时  : ~430 ms
----------------------------------------------------
  no_overlap  实测: 699.x ms
  tbo_overlap 实测: 427.x ms
  加速比          : 1.6x
----------------------------------------------------

✅ adv09_tbo_dbo_overlap 通过
```

无 GPU 依赖，纯 Python 标准库即可运行。

---

## 7. 下一步

**adv10: PD Disaggregation（预填充-解码分离）**

TBO/DBO 在单机多卡层面减少了计算通信串行开销。  
更激进的方案是在**集群层面**把 Prefill 和 Decode 部署到不同节点：

- Prefill 节点：大 batch，计算密集，追求吞吐
- Decode  节点：小 batch，访存密集，追求低延迟
- 两类节点之间通过 KV Cache 传输解耦

→ adv10 将用模拟的 KV 传输队列演示 PD 分离的调度思路。
