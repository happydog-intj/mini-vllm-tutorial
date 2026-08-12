# adv08: Data Parallel + DPLB — 数据并行与负载均衡

---

## 1. 教学目标

- 理解**数据并行（Data Parallelism）**：多个完整模型副本并行处理不同请求
- 掌握**朴素轮询（RoundRobin）**在异构/突发场景下的局限性
- 学习 **DPLB（Data-Parallel Load Balancing）** 核心思想：基于实时负载路由
- 通过纯 Python 模拟，直观感受不同路由策略对副本负载均衡的影响
- 为下一步 TBO/DBO（Token-level / Decode-phase Batching Optimization）打基础

---

## 2. 问题：朴素轮询在异构/突发负载下失效

### 场景

生产环境中，推理服务通常跑多个副本（Data Parallel）：

```
客户端请求
    |
   [LB]  ← 负载均衡器
  / | \
 R0 R1 R2  ← 3 个推理副本（完整模型各一份）
```

**异构**：副本所在节点可能性能不同（热限速、显存带宽差异、NUMA 不对齐……）

### RoundRobin 的问题

朴素轮询按顺序依次分配请求，完全不感知各副本的当前负载：

```
请求 1 → R0    请求 2 → R1    请求 3 → R2
请求 4 → R0    请求 5 → R1    请求 6 → R2
...
```

当 **R1 处理速率只有 R0/R2 的 30%** 时，突发流量会在 R1 上形成无限积压：

```
时刻 t=30（突发期）:
  R0: 负载 ~  50  ░░░░░░░░░░░░░░░░░░░░
  R1: 负载 ~ 800  ██████████████████████████████████████████████████████
  R2: 负载 ~  48  ░░░░░░░░░░░░░░░░░░░░

→ R1 严重积压，R0/R2 空转，整体吞吐浪费，SLA 恶化
```

---

## 3. 原理：DPLB 最小负载路由

### 多副本 + LB 路由全景

```
                   ┌───────────────────────────────┐
  Client ──────▶  │    Load Balancer (LB)          │
  requests         │                               │
                   │  RoundRobin:  req → R[i%N]    │
                   │  LeastLoad:   req → min(load) │
                   └──────────┬────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐    ┌─────────┐    ┌─────────┐
         │Replica-A│    │Replica-B│    │Replica-C│
         │speed=1.0│    │speed=0.3│    │speed=1.2│
         │ queue[] │    │ queue[] │    │ queue[] │
         │in_flight│    │in_flight│    │in_flight│
         └─────────┘    └─────────┘    └─────────┘
```

### 负载分布对比（模拟结果）

```
RoundRobin（朴素轮询）——B 副本严重积压:
  replica-A ████                     ( 2,727)
  replica-B ███████████████████████████████████  (70,559)
  replica-C ████                     ( 2,566)
  平均方差: 233,005

LeastLoad / DPLB——三副本接近均衡:
  replica-A ████████                 ( 4,266)
  replica-B ████████████             ( 6,325)
  replica-C ██████                   ( 3,540)
  平均方差: 863  (改善 99.6%)
```

**关键洞察**：

- DPLB 将 B 的积压及时"暴露"给路由器，后续请求绕开 B 直到其消化完队列
- 不需要任何模型权重或 GPU 拓扑感知，仅凭 `load = in_flight + sum(queue)` 即可决策
- 收益在**异构副本 + 突发流量**场景下最显著

### ❓ Q1：LeastLoad 会导致"蜂拥效应"吗？

**问题**：所有请求都路由到当前最轻的副本，不会导致它瞬间变重吗？

**答案**：会的！这是 LeastLoad 的经典问题。看模拟过程：

```
时刻 t，三副本负载：A=5, B=800, C=3
  请求 1 → 选 C（最轻=3）→ C 变成 8
  请求 2 → 选 A（最轻=5）→ A 变成 10
  请求 3 → 选 C（最轻=8）→ C 变成 13
  请求 4 → 选 A（最轻=10）→ A 变成 15
```

请求会在 A 和 C 之间"震荡"，而不是都涌向一个。这是因为**每次路由后负载立刻更新**，下一轮路由会感知到新负载。但如果请求是批量到达（同一时刻多个），确实可能出现蜂拥。生产系统的解法：**加权随机**（按负载的倒数作为概率权重），而非严格选最小。

### ❓ Q2：`in_flight + sum(queue)` 这个负载度量合理吗？

**问题**：队列里的大小和正在处理的大小直接相加，单位一致吗？

**答案**：单位都是 token 数，所以**量纲一致**，但**权重不同**：

```python
load = in_flight + sum(queue)
# in_flight 的 token 正在被 GPU 处理，即将完成
# queue 的 token 还没开始处理，需要排队

更精细的负载度量：
load = in_flight * 0.5 + sum(queue) * 1.0
# in_flight 打折（因为它快完了）
# queue 全额计入（它还要等）
```

教学版用简单加法是因为模拟的 `step()` 函数中，in_flight 和 queue 的"剩余工作量"可以近似等价。真实系统中需要考虑 KV Cache 占用、GPU 利用率等多维指标。

### ❓ Q3：会话亲和（sticky routing）和 DPLB 冲突怎么办？

**问题**：如果请求 A 的第一轮在 R0 上，KV Cache 也在 R0。第二轮来了，DPLB 说 R1 最轻——但 R1 没有 A 的 KV Cache！

**答案**：这就是 **KV Cache 命中率和负载均衡的 trade-off**：

```
方案 1：强制亲和（sticky）
  请求 A 的所有轮次都去 R0
  → KV Cache 命中率 100%，但 R0 可能过载

方案 2：KV 迁移（migrate）
  请求 A 第二轮去 R1，R0 把 A 的 KV 传给 R1
  → 负载均衡，但迁移有网络开销

方案 3：软亲和（soft sticky）
  优先去 R0，但如果 R0 负载 > 阈值则去 R1
  → 平衡两者（vLLM prefix caching 用这种）
```

教学版没实现多轮对话，所以没有这个问题。生产系统中必须处理。

---

## 4. 实现细节

### `Replica` — 副本抽象

```python
class Replica:
    def load(self):
        return self.in_flight + sum(self.queue)  # 总未完成 token

    def step(self, dt):
        done = min(self.in_flight, int(self.speed * dt * 100))  # 按速率消耗
        self.in_flight -= done
        while self.queue and self.in_flight < 100:
            self.in_flight += self.queue.pop(0)  # 从队列补充
```

- `speed`：处理速率倍率，0.3 = 每步消耗约 30 token（模拟降速副本）
- `in_flight`：当前批次 token，上限 100（模拟最大批大小）
- `queue`：等待进入批次的请求列表

### `RoundRobinLB` — 朴素轮询

```python
class RoundRobinLB:
    def __init__(self): self.i = 0
    def route(self, replicas, req_size):
        r = replicas[self.i % len(replicas)]
        self.i += 1
        return r
```

无状态感知，O(1) 路由，不感知各副本负载。

### `LeastLoadLB` — DPLB 核心

```python
class LeastLoadLB:
    def route(self, replicas, req_size):
        return min(replicas, key=lambda r: r.load())
```

每次路由前查询所有副本的实时负载，选最轻者。O(N) 查询，N 为副本数。

### `simulate` — 时间步驱动模拟

```python
def simulate(replicas, lb, arrivals, total_time=50):
    for t in range(total_time):
        # 注入当前时刻到达的请求
        while arrivals and arrivals[0][0] == t:
            _, size = arrivals.pop(0)
            r = lb.route(replicas, size)
            r.queue.append(size)
        # 推进所有副本一步
        for r in replicas:
            r.step(1)
        # 记录快照
        log.append({r.name: r.load() for r in replicas})
    return log
```

离散事件模拟：每时间步先注入请求再推进副本状态。

---

## 5. 教学版 vs 真实框架

| 维度 | 本教学模拟 | 真实系统（vLLM/生产） |
|------|------------|----------------------|
| **数据并行** | Python `Replica` 对象模拟 token 队列 | 独立进程/容器，各持完整模型权重 |
| **负载度量** | `in_flight + sum(queue)`（token 数） | 待处理 token 数、KV-cache 占用率、GPU 利用率 |
| **DPLB 路由** | `min(replicas, key=load)` 贪心 | 带心跳更新的分布式一致性哈希 + 最小连接数 |
| **异构感知** | `speed` 系数模拟 | 实测 throughput / 动态权重调整 |
| **健康检查** | 无 | 心跳检测、熔断、副本下线摘除 |
| **会话亲和** | 无 | 可选粘性路由（KV-cache 命中率优化） |

### vLLM 多实例路由

vLLM 官方推荐在多实例部署时使用外部负载均衡器（如 Envoy、Nginx）配合最小连接数策略，或使用 `vllm.serve` 的 `--router least-load` 参数（实验性）。核心原则与本模拟一致：**感知副本实时负载，而非盲目轮询**。

---

## 6. 运行

```bash
cd advanced/adv08_data_parallel_dplb
python run.py
```

预期输出（关键部分）：

```
策略 1: RoundRobin (朴素轮询)
  replica-B: 累计负载= 70559  █████████████████████████████████████
  平均跨副本负载方差: 233,005

策略 2: LeastLoad / DPLB (最小负载路由)
  replica-B: 累计负载=  6325  ███
  平均跨副本负载方差: 863  (改善 99.6%)

✅ adv08_data_parallel_dplb 通过
```

---

## 7. 下一步

**adv09 — TBO/DBO（Token-level Batching Optimization / Decode-phase Batching Optimization）**

在单副本内部，不同请求处于不同生成阶段（prefill / decode）。TBO/DBO 探索如何在一个批次内混合调度 prefill 与 decode token，进一步提升 GPU 利用率，减少"气泡"浪费。

这是数据并行（多副本）→ 批内优化（单副本内部）的自然延伸。
