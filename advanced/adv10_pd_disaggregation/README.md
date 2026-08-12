# adv10: PD Disaggregation — Prefill/Decode 分离部署

## 1. 教学目标

- 理解 Prefill（算力密集）与 Decode（存储带宽密集）的根本资源差异
- 掌握合并部署（Colocated）的核心瓶颈：两类工作负载抢同一 GPU，互相阻塞，配比难调
- 掌握 PD 分离部署（Disaggregated）的核心收益：
  1. **P/D 引擎独立配比**：按负载特性单独选型/扩容
  2. **流水线重叠**：P 处理下一请求的 Prefill 时，D 已在解码上一请求
- 用纯 Python 模拟（数学时间线追踪）理解调度直觉；了解真实框架的实现

> **注意：本教程为纯 Python 教学模拟。**  
> 所有"耗时"均为数学计算值（秒），不依赖 GPU、不实际执行 LLM 推理。  
> `transfer_kv` 中的 `time.sleep` 仅演示概念；P/D 引擎的"忙碌时间"为  
> 公式模拟结果，不代表真实部署的精确性能数据。

---

## 2. 问题：合并部署时，Prefill 与 Decode 互相阻塞

### Prefill 与 Decode 的资源特性

```
Prefill（处理 prompt）：
  计算量 ∝ prompt_len²    ← 注意力矩阵 n×n 大矩阵乘法
  瓶颈：GPU 算力（FLOPS）
  特点：越长的 prompt 越重（2048 tokens 约是 512 tokens 的 16×）

Decode（逐步生成）：
  计算量 ∝ kv_size × steps  ← 每步读整个 KV Cache
  瓶颈：显存带宽（Memory Bandwidth）
  特点：每步计算量小，但需反复读写大量 KV 数据
```

### 合并部署的问题

当 Prefill 与 Decode 部署在同一 GPU 上时：

```
时间轴（合并部署，串行执行）：

  请求1  [Prefill1]  [Decode1]
  请求2             [Prefill2]  [Decode2]
  请求3                        [Prefill3]  [Decode3]

  ───────────────────────────────────────────▶ 时间

  问题1：Prefill 独占 GPU 时，Decode 完全阻塞（算力竞争）
  问题2：Decode 运行时，Prefill 被迫等待（带宽与算力混用）
  问题3：配比无法单独优化——同一块 GPU 既要跑大矩阵又要跑带宽密集任务
  问题4：长 prompt（2048 tokens）的 Prefill 耗时是短 prompt 的 16×，
         严重拉高所有请求的 TTFT 和总延迟
```

实测（本教程数据）：

```
8 个长 prompt 请求（512~2048 tokens），合并部署：
  wall_time ≈ 21215 ms
  P 利用率: 96.8%（Prefill 几乎占满）
  D 利用率:  3.2%（Decode 被严重挤压）
```

---

## 3. 原理：PD 分离 + KV 迁移 + 独立配比

### 合并部署 vs PD 分离 — ASCII 时间线

```
合并部署（Colocated）：P 与 D 共享同一引擎，严格串行

  引擎  [─Prefill1─][─Decode1─][─Prefill2─][─Decode2─][─Prefill3─][─Decode3─]
        ↑           ↑          ↑
        Prefill 独占 GPU       Decode 等待

  总耗时 = Σ (t_prefill_i + t_decode_i)   ← 全部串行，无重叠

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PD 分离（Disaggregated）：P/D 独立节点，流水线重叠

  P 引擎  [──Prefill1──][──Prefill2──][──Prefill3──]
                │              │              │
               KV             KV             KV   ← KV 迁移（网络传输）
                │              │              │
  D 引擎        [──Decode1──][──Decode2──][──Decode3──]

  ───────────────────────────────────────────▶ 时间

  收益1：D 解码请求1 时，P 已在 Prefill 请求2（流水线重叠）
  收益2：P 节点可单独选高算力 GPU（专为大矩阵优化）
  收益3：D 节点可单独选高带宽 GPU/内存（专为 KV Cache 读写优化）
  收益4：P/D 节点数量可独立扩容（按实际瓶颈配比）
```

### 流水线时间线数学追踪

```
p_time ← P 引擎时钟（下一个 Prefill 的开始时刻，初始 = 0）
d_time ← D 引擎时钟（D 的空闲时刻，初始 = 0）

对每个请求 i：
  t_p       = prompt_len² / (p_speed × 1e6)    # Prefill 耗时
  p_done    = p_time + t_p                      # P 完成时刻
  kv_arrive = p_done + kv_latency               # KV 到达 D 节点
  d_start   = max(d_time, kv_arrive)            # D 最早可开始时刻
  t_d       = kv_size × steps / (d_speed × 1e6)
  d_time    = d_start + t_d                     # D 完成时刻
  p_time    = p_done                            # P 无需等 D，立即继续

wall_time = max(p_time, d_time)                 # 所有任务完成时刻
```

### ❓ Q1：Prefill 耗时为什么是 `prompt_len²`？

**问题**：注意力矩阵是 `[seq, seq]`，所以是平方。但实际推理中 KV Cache 是增量的，每次 prefill 的复杂度真的是 O(n²) 吗？

**答案**：对于**纯 prefill**（全新 prompt，没有 cached KV），是的——注意力矩阵大小是 `[prompt_len, prompt_len]`，矩阵乘法复杂度 O(n²×d)。

```
但如果 prompt_len = 2048：
  注意力矩阵：2048 × 2048 = 4,194,304 个元素
  计算量（单头）：4M × d_head 次乘加
  
对比 prompt_len = 512：
  注意力矩阵：512 × 512 = 262,144 个元素
  2048² / 512² = 16× → 正如 README 所说
```

注意：这是 prefill 阶段的复杂度。decode 阶段（单 token 增量）的注意力是 O(seq × d)，不是平方。

### ❓ Q2：KV 迁移延迟在实际中是多少？

**问题**：教学版用常数 `kv_latency=0.001`（1ms），真实值取决于什么？

**答案**：真实延迟 = **KV 大小 / 网络带宽**：

```
KV 大小 = prompt_len × num_layers × 2 × num_heads × d_head × 2 bytes
LLaMA-7B, prompt_len=2048, fp16:
  = 2048 × 32 × 2 × 32 × 128 × 2 bytes ≈ 1 GB

网络带宽：
  NVLink（同机）：~200 GB/s → 传输 ~5ms
  InfiniBand（跨机）：~25 GB/s → 传输 ~40ms
  以太网（跨机）：~10 GB/s → 传输 ~100ms

所以 KV 迁移在生产中不可忽略，特别是跨机部署时。
```

教学版用 1ms 是为了让流水线效果可见，不代表真实值。

### ❓ Q3：什么情况下合并部署反而更好？

**问题**：PD 分离总是更好吗？有没有合部署更优的场景？

**答案**：**短 prompt + 短生成**场景下，合并部署更好：

```
短 prompt（64 tokens），短生成（10 steps）：
  Prefill 耗时 ≈ 64² / 1e6 ≈ 0.004s（几乎为零）
  Decode 耗时 ≈ 64 × 10 / 1e6 ≈ 0.0006s（也很小）

合并部署：总耗时 ≈ 0.0046s
分离部署：总耗时 ≈ max(0.004, 0.0006) + kv_latency ≈ 0.004 + 1ms ≈ 0.005s
  → KV 迁移开销超过了流水线重叠收益！
```

PD 分离的收益在**长 prompt（>512 tokens）+ 长生成（>50 steps）**时才显著。短请求场景，合并部署更简单、延迟更低。

---

## 4. 实现细节

### PrefillEngine

```python
class PrefillEngine:
    def prefill(self, prompt_len):
        t = (prompt_len ** 2) / (self.speed * 1e6)
        self.busy += t
        return {'kv_size': prompt_len}, t
```

- 耗时 ∝ `prompt_len²`，模拟注意力矩阵的算力密集特性
- `speed` 为独立配比参数，可不同于 D 引擎
- 返回 `kv_state`（携带 `kv_size`）和本次耗时

### DecodeEngine

```python
class DecodeEngine:
    def decode(self, kv_state, steps):
        t = kv_state['kv_size'] * steps / (self.speed * 1e6)
        self.busy += t
        return t
```

- 耗时 ∝ `kv_size × steps`，模拟每步读全量 KV Cache 的带宽密集特性
- 独立 `speed` 参数，可针对存储密集型工作负载单独优化

### transfer_kv

```python
def transfer_kv(kv_state, latency=0.001):
    time.sleep(latency)
    return latency
```

- 模拟 KV Cache 在 P/D 节点间的网络传输延迟
- 真实场景通过 NVLink / RDMA / PCIe 传输；延迟通常为 0.1~10 ms
- 教学版用常数延迟近似（实际取决于 KV 大小和网络带宽）

### colocated — 合并部署

```python
def colocated(reqs, p_speed=1.0, d_speed=1.0):
    pe = PrefillEngine(p_speed)
    de = DecodeEngine(d_speed)
    total = 0.0
    for prompt_len, steps in reqs:
        kv, t_p = pe.prefill(prompt_len)
        t_d = de.decode(kv, steps)
        total += t_p + t_d
    return total, pe.busy, de.busy
```

- 严格串行：每个请求先 Prefill 再 Decode，无重叠
- `p_speed` 与 `d_speed` 物理上共享同一 GPU，应取相同值（分离无意义）

### disaggregated — 分离部署（流水线追踪）

```python
def disaggregated(reqs, p_speed=1.0, d_speed=1.0, kv_latency=0.001):
    p_time, d_time = 0.0, 0.0
    for prompt_len, steps in reqs:
        kv, t_p = pe.prefill(prompt_len)
        p_done = p_time + t_p
        kv_arrive = p_done + kv_latency
        d_start = max(d_time, kv_arrive)
        t_d = de.decode(kv, steps)
        d_time = d_start + t_d
        p_time = p_done
    return max(p_time, d_time), pe.busy, de.busy
```

- 用双时钟（`p_time` / `d_time`）模拟两个独立引擎的流水线调度
- P 引擎处理完即推进时钟，无需等待 D 完成（关键：P/D 解耦）
- `max(d_time, kv_arrive)` 保证 D 收到 KV 后才开始 Decode

---

## 5. 教学版 vs 真实框架

本教程为**纯 Python 模拟**，与真实部署有以下差异：

| 维度 | 本教程（教学版） | 真实框架 |
|------|----------------|---------|
| 执行方式 | 数学公式模拟耗时 | 实际 GPU 矩阵运算 |
| KV 迁移 | `time.sleep(latency)` 常数近似 | 按实际 KV 大小通过网络/NVLink 传输 |
| 流水线 | 单线程时间线追踪 | 多进程 / CUDA Stream 真实并发 |
| 调度器 | 简单顺序循环 | 复杂的在线调度（抢占、优先级、背压） |

### 真实框架实现参考

**DeepSeek / vLLM PD Disaggregation**
- vLLM 在 v0.4+ 引入 Disaggregated Prefill 接口
- DeepSeek 在生产环境中使用 PD 分离，P 集群与 D 集群独立部署
- KV 通过 RDMA（InfiniBand）在节点间传输，延迟约 1~5 ms

**Mooncake KVStore**
- 月之暗面提出的分布式 KV Cache 存储层
- 将 KV Cache 从 GPU 卸载到 CPU/DRAM/SSD，实现跨节点高效共享
- 论文：*Mooncake: A KVCache-centric Disaggregated Architecture* (2024)

**DistServe**
- 学术系统，最早系统性研究 PD 分离的论文
- 核心贡献：量化 Prefill/Decode 的资源异构性，提出独立配比和流水线调度
- 论文：*DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving* (OSDI 2024)

**Splitwise**
- Microsoft 的 PD 分离实现，侧重成本优化
- 论文：*Splitwise: Efficient Generative LLM Inference Using Phase Splitting* (ISCA 2024)

---

## 6. 运行

```bash
cd advanced/adv10_pd_disaggregation
python run.py
```

预期输出（数值仅供参考）：

```
============================================================
  adv10: PD Disaggregation — Prefill/Decode 分离对比实验
============================================================
  请求数量  : 8
  Prompt 长度分布 : [2048, 1024, 1536, 2048, 512, 1024, 1800, 2048]
  KV 迁移延迟     : 1.0 ms/请求
============================================================
[A] 合并部署 (colocated, p_speed=1.0, d_speed=1.0)
    wall_time  : 21215.18 ms
[B] 分离部署 (disaggregated, p_speed=1.0, d_speed=1.0)
    wall_time  : 20603.94 ms  ← 流水线重叠收益
[C] 分离部署 (disaggregated, p_speed=2.0, d_speed=1.0)  ← 独立配比优化
    wall_time  : 10333.19 ms

  [B] vs [A] 加速比（流水线）   : 1.03x
  [C] vs [A] 加速比（配比优化）  : 2.05x

✅ adv10_pd_disaggregation 通过
```

---

## 7. 下一步

**adv11: AFD — Attention/FFN 分离**

PD 分离是在请求粒度把 Prefill 与 Decode 分到不同节点。
adv11 更进一步，在**算子粒度**把 Attention（访存密集）与 FFN（算力密集）
分配到不同硬件，解决 MoE / 超大 FFN 场景下单卡资源不均衡的问题。

```
PD 分离（adv10）：               AFD 分离（adv11）：
  请求粒度                          算子粒度
  ┌─────────┐  KV  ┌─────────┐    ┌──────────┐  激活  ┌──────────┐
  │ Prefill │ ──→  │ Decode  │    │ Attention│  ──→   │   FFN    │
  │  节点   │      │  节点   │    │  节点    │        │  节点    │
  └─────────┘      └─────────┘    └──────────┘        └──────────┘
  算力密集  带宽密集              访存密集              算力密集
```

→ 下一步：[adv11_afd_attention_ffn](../adv11_afd_attention_ffn/README.md)
