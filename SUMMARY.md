# 总结：大模型推理服务的配置、指标与优化

本文是 mini-vllm-tutorial 全部 15 步的总结。经过从零实现，我们已经理解了 LLM 推理引擎的每一个核心组件。现在把这些知识汇聚成一张实际运维的地图：**上线一个推理服务，需要关注什么、监控什么、从哪里找优化空间。**

---

## 一、系统配置：上线前的关键决策

### 1. KV Cache 显存预算

KV Cache 是推理服务中最核心的资源，直接决定系统能同时服务多少请求。

```
KV Cache 显存 = num_layers × 2(K+V) × num_kv_heads × head_dim × max_total_tokens × dtype_bytes

Qwen3-0.6B 示例（BF16）:
  = 28 × 2 × 8 × 64 × max_tokens × 2字节
  = 57,344 字节 × max_tokens
  ≈ 56KB / token

A100 80GB，模型权重占 1.2GB，剩余约 78GB：
  最多缓存约 78GB / 56KB ≈ 1,400,000 个 token 的 KV
```

实际配置时 `gpu_memory_utilization`（vLLM 默认 0.9）控制留给 KV Cache 的显存比例。

### 2. 并发数与 max_num_seqs

`max_num_seqs`：调度器同时运行的最大请求数。

```
设太小 → GPU 利用率低，吞吐量上不去
设太大 → 每个请求等待的其他请求变多，平均延迟上升
         且 KV Cache 可能不够，触发频繁 Preemption（见 step11）

经验值：先用 max_num_seqs = 总KV槽位 / 平均序列长度 作为上限
```

### 3. chunk_size（Chunked Prefill）

控制每步最多处理的 prefill token 数（step10）。

```
chunk_size 小（如 256）：
  → decode 请求每步延迟增加少（每步 prefill 时间短）
  → 长 prompt 的 TTFT 更长（需要更多步）
  
chunk_size 大（如 2048）：
  → 长 prompt TTFT 更短
  → decode 请求每步可能卡顿更明显

推荐：根据实际负载的 prompt 长度分布调整。
      nano-vllm 默认 512，vLLM 默认 512。
```

### 4. block_size（PagedAttention）

KV Cache 的分配粒度（step12）。

```
block_size 小（如 8）：
  → 内存碎片少，利用率高
  → Block 数量多，block_table 查找开销稍大

block_size 大（如 32）：
  → 管理开销小
  → 最后一个 Block 可能有较多未用槽位

推荐：16 或 32，通常不需要调整。
```

### 5. 精度选择

| 精度 | 显存 | 速度 | 精度损失 | 适用场景 |
|------|------|------|---------|---------|
| FP32 | 4字节/参数 | 慢 | 无 | 训练、调试 |
| BF16 | 2字节/参数 | 快 | 极小 | 推理标准选择 |
| INT8 | 1字节/参数 | 更快 | 小 | 显存紧张时 |
| INT4 | 0.5字节/参数 | 最快 | 中等 | 极限压缩 |

本教程 step15 起使用 BF16，是生产推理的标准选择。

---

## 二、核心指标：监控什么

### 延迟指标

```
TTFT（Time to First Token）= 从请求发出到返回第一个 token 的时间
  主要受 Prefill 影响，随 prompt 长度增加而增加（O(n²)）
  典型目标：< 500ms（对话场景）

TPOT（Time Per Output Token）= 每个生成 token 的平均时间
  主要受 Decode 影响，GPU 显存带宽是瓶颈
  典型目标：< 50ms/token（即 > 20 tok/s）

E2EL（End-to-End Latency）= 完整请求的总时间
  = TTFT + TPOT × (output_len - 1)
```

### 吞吐量指标

```
Throughput = 单位时间生成的 token 总数（tok/s）
  衡量系统整体处理能力，与 batch size 强相关
  
RPS（Requests Per Second）= 每秒处理的完整请求数
  = Throughput / 平均输出长度

典型优先级：
  在线服务（聊天）：优先低延迟（TTFT < 500ms）
  离线批处理（摘要、翻译）：优先高吞吐（tok/s 最大化）
```

### 资源指标

```
GPU 利用率（Compute Utilization）
  Prefill 阶段：计算密集，利用率高（应接近 100%）
  Decode 阶段：内存带宽密集，利用率可能较低（正常）
  持续低（< 30%）：请求量不足 或 调度效率低

GPU 显存利用率
  KV Cache 占用：应尽量高（步骤 step12 的目标是 ~96%）
  持续低（< 50%）：max_num_seqs 设置过小，或请求量不足

KV Cache 命中率（Prefix Cache Hit Rate）
  = 命中缓存的 Block 数 / 总请求的 Block 数
  系统提示词场景下应 > 80%
  接近 0：无重复前缀，不适合前缀缓存

Preemption 次数
  频繁 Preemption（step11）说明 KV Cache 不够用
  需要减少 max_num_seqs 或增加显存
```

### 队列指标

```
Waiting Queue Length = 等待调度的请求数
  持续增长：系统过载，处理速度跟不上请求到达速度
  
Scheduling Delay = 请求等待进入 running 队列的时间
  正常应 < 100ms
  过高：调度器频繁 Preemption 或 chunk_size 设置不合理
```

---

## 三、寻找优化空间：瓶颈分析

优化推理服务的第一步是**找到瓶颈在哪里**，不同瓶颈的优化方向完全不同。

### 瓶颈诊断树

```
TTFT 高？
  ├─ prompt 很长（> 1024 token）？
  │    → 减小 chunk_size，让 decode 不被阻塞（step10）
  │    → 开启 Prefix Caching，相同前缀复用（step13）
  └─ prefill 本身慢？
       → 换更快的 GPU，或开启 FlashAttention（step16）

TPOT 高（生成慢）？
  ├─ GPU 利用率低（< 60%）？
  │    → 增大 max_num_seqs，提高并发（step09）
  │    → 检查 Preemption 是否频繁（step11）
  └─ GPU 利用率高（> 90%）但仍慢？
       → Decode 已到显存带宽上限，需要更好的 GPU 或 Tensor Parallelism（step18）

Throughput 低？
  ├─ KV Cache 利用率低（< 60%）？
  │    → 增大 max_num_seqs
  │    → 检查 PagedAttention 配置（step12）
  └─ 队列持续增长？
       → 系统过载，需要扩容（更多 GPU 或 Tensor/Pipeline Parallelism）

Preemption 频繁？
  → KV Cache 不够，选一个：
     a. 减少 max_num_seqs（降低并发）
     b. 减小 max_model_len（限制最大序列长度）
     c. 换更大显存的 GPU
     d. 降低精度（BF16 → INT8，节省显存）
```

### 各步骤的优化效果速查

| 优化手段 | 对应步骤 | 主要收益 | 代价 |
|---------|---------|---------|------|
| KV Cache | step07 | Decode 速度提升 10-100× | 显存占用增加 |
| Continuous Batching | step09 | 吞吐量提升 2-5× | 实现复杂度 |
| Chunked Prefill | step10 | TPOT 稳定，TTFT 改善 | 长 prompt TTFT 微增 |
| Preemption | step11 | 避免 OOM，稳定性提升 | 被抢占请求需重新 prefill |
| PagedAttention | step12 | 显存利用率 ~18% → ~96% | block_table 管理开销 |
| Prefix Caching | step13 | 相同前缀节省 50-90% prefill | 显存常驻，需要 LRU 管理 |
| FlashAttention | step16 | 注意力计算速度 2-4×，显存大幅减少 | 需要 NVIDIA GPU |
| CUDA Graph | step17 | Decode 延迟降低 30-60% | 只对 Decode 有效 |
| Tensor Parallelism | step18 | 线性扩展吞吐，支持更大模型 | 需要多 GPU + NVLink |

### 实际优化流程

```
第一步：建立基线
  python step19_benchmark/run.py
  记录：TTFT P50/P95/P99，TPOT，Throughput，GPU利用率，显存使用

第二步：确定主要瓶颈
  用上面的「瓶颈诊断树」判断当前限制在哪里

第三步：单变量实验
  每次只改一个配置，重新跑 benchmark，对比数据
  常见旋钮：
    max_num_seqs：10 → 20 → 40 → 80
    chunk_size：128 → 256 → 512 → 1024
    gpu_memory_utilization：0.7 → 0.8 → 0.9

第四步：检查是否引入新问题
  优化吞吐量时：TTFT 是否变高了？
  优化 TTFT 时：Throughput 是否下降了？
  这些权衡通常无法避免，找到业务可接受的平衡点
```

---

## 四、本教程覆盖的技术栈

```
用户请求
    │
    ▼
HTTP 服务（step20）
FastAPI + SSE 流式输出 + asyncio 异步队列
    │
    ▼
调度器（step09 + step10 + step11）
Continuous Batching → Chunked Prefill → Preemption
    │
    ▼
内存管理（step12 + step13）
PagedAttention → Prefix Caching
    │
    ▼
模型推理（step15 + step16 + step17 + step18）
Qwen3-0.6B → FlashAttention → CUDA Graph → Tensor Parallelism
    │
    ▼
基础组件（step01~step07）
Tokenizer → Embedding → Attention → Transformer → KV Cache
    │
    ▼
指标采集（step19）
TTFT / TPOT / Throughput / GPU 利用率 / KV 命中率
```

---

## 五、与 nano-vllm 的差距

本教程是教学实现，与生产级推理引擎（nano-vllm、vLLM）的主要差距：

| 特性 | mini-vllm-tutorial | nano-vllm |
|------|-------------------|-----------|
| KV Cache 存储 | Python list（HuggingFace 风格） | 物理显存块，Triton kernel 写入 |
| 注意力计算 | PyTorch SDPA | FlashAttention varlen（prefill）+ with_kvcache（decode）|
| Batch 注意力 | 逐请求 for 循环 | 真正的变长 batch，一次 kernel |
| 采样 | Python argmax/multinomial | Gumbel-Max + `@torch.compile` |
| Tensor Parallel | 单进程模拟 | multiprocessing + NCCL all_reduce |
| 性能（A100，Qwen3-0.6B）| ~50 tok/s（教学版）| ~1400 tok/s |

差距约 28×，来自每一层的工程优化叠加。理解了本教程的每个步骤，再去读 nano-vllm 的源码，每一行都能对应到这里学到的概念。

---

*mini-vllm-tutorial 完结。从零到完整推理引擎，15 步，8 个阶段。*
