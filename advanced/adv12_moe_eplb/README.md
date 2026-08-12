# adv12 — Mixture of Experts (MoE) + EPLB 专家负载均衡

---

## 1. 教学目标

- 理解 **Mixture of Experts (MoE)** 架构:用多个专家 FFN 替换单一 FFN,每个 token 只激活 top-k 个。
- 掌握 **Top-k 路由** 原理:gate 网络打分 → softmax → topk 选择 → 加权求和。
- 认识 **专家负载不均衡** 问题:某些专家被大量 token 选中,其他专家几乎空闲。
- 学习 **EPLB (Expert Parallel Load Balancing)** 核心思想:通过贪心装箱将专家重新分配到设备,使各设备负载更均等。

---

## 2. 问题

### 稠密模型的算力浪费

在标准 Transformer 中,每个 token 必须流过 **完整的 FFN (MLP)**:

```
token → Attention → FFN (全部参数) → 下一层
                    ^^^^^^^^^^^^^^^^
                    每个 token 都要走,算力 O(seq × d_ff)
```

随着模型越来越大(d_ff 从 4096 到 16384+),FFN 的计算量线性增长,但每个 token 实际上只需要一小部分"知识"。

### MoE 的思路

把单一 FFN 替换为 N 个**专家 FFN**,每个 token 只激活其中 top-k 个:

```
token → gate → 选 top-k 专家 → 只走 k 个 FFN → 加权求和
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               计算量从 O(d_ff) 降为 O(k × d_ff),k << N
```

但 MoE 带来新问题:**路由天然不均衡**。

```
专家负载 (token 数):
  专家 0 ████████████████████████ 38  ← 过载
  专家 1 █████████████████████    34
  专家 2 █████████████            29
  专家 3 ████████████             27  ← 相对空闲

最大/最小比 = 38/27 ≈ 1.4 → 专家 0 成为瓶颈
```

在真实多卡部署中,专家分布在不同 GPU 上,最慢的 GPU 决定整个系统吞吐。

---

## 3. 原理

### 3.1 Top-k 路由示意

```
输入 token x  [seq=64, d_model=8]
        |
        v
  gate = Linear(8 → 4)
        |
        v
  softmax → scores [64, 4]
        |
        v
  topk(k=2) → 每个 token 选 2 个专家
        |
     ┌──┴──────────────────┐
     | topk_val [64, 2]    |  归一化权重
     | topk_idx [64, 2]    |  专家编号
     └──┬──────────────────┘
        |
  for each expert e:
    mask = (topk_idx == e)   ← 哪些 token 选了专家 e
    out[mask] += weight * expert_e(x[mask])
        |
        v
  out [64, 8]   load [4]
```

### 3.2 专家负载柱状图

```
路由结束后统计 bincount:

token 路由次数 (top_k=2, seq=64, 总路由=128 次):

专家 0 | ███████████████████████████████████ | 38
专家 1 | ██████████████████████████████████  | 34
专家 2 | █████████████████████████████       | 29
专家 3 | ███████████████████████████         | 27
        0          10          20          30  40
```

### 3.3 EPLB 重分布 (贪心装箱 LPT)

```
目标: 把 4 个专家分配到 2 台设备,使设备负载最均衡

步骤:
  1. 按负载排序: [38, 34, 29, 27] (专家 0,1,2,3)
  2. 贪心: 每次选负载最重的专家,分配给当前最轻的设备

                Device 0          Device 1
                --------          --------
  分配专家 0(38) → Device 0         [38]   [  ]
  分配专家 1(34) → Device 1         [38]   [34]
  分配专家 2(29) → Device 1         [38]   [63]    ← 此时 D1 轻
  分配专家 3(27) → Device 0         [65]   [63]
                                    ↑         ↑
                              最终相差 2,接近完美均衡

朴素均分 (专家0+1 → D0, 专家2+3 → D1):
  Device 0: 38+34 = 72  ← 瓶颈
  Device 1: 29+27 = 56

EPLB 装箱:
  Device 0: 38+27 = 65
  Device 1: 34+29 = 63  ← 仅差 2,方差从 128 降至 2
```

---

## 4. 实现细节

### 4.1 MoELayer

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=2):
        self.gate    = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model)
                                      for _ in range(num_experts)])
```

**前向传播关键点:**

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | `softmax(gate(x))` | 得到 [seq, num_experts] 概率分布 |
| 2 | `topk(scores, k)` | 选最高 k 个专家 |
| 3 | `topk_val / sum` | 归一化,使权重和为 1 |
| 4 | mask + 加权累加 | 每个专家只处理属于自己的 token |
| 5 | `bincount` | 统计每专家负载 |

**topk_val 归一化的必要性:** 若不归一化,top-k 权重之和 < 1(因为 softmax 还分配了权重给其他专家),输出幅值会系统性缩小。

### 4.2 expert_imbalance

```python
def expert_imbalance(load):
    return load.max().float() / (load.min().float() + 1e-6)
```

- 值 = 1.0:完美均衡
- 值 > 1.0:存在不均衡,值越大越严重
- `+ 1e-6`:防止某专家 load=0 时除零

### 4.3 eplb_rebalance

```python
def eplb_rebalance(load, num_devices):
    experts_sorted = torch.argsort(load, descending=True)  # LPT 排序
    device_load = [0] * num_devices
    assignment = {}
    for e in experts_sorted.tolist():
        d = device_load.index(min(device_load))  # 最轻设备
        assignment[e] = d
        device_load[d] += load[e].item()
    return assignment, device_load
```

这是经典的 **LPT (Longest Processing Time First)** 算法,时间复杂度 O(E log E + E·D),其中 E=专家数,D=设备数。可证明该算法给出的最大完工时间不超过最优解的 4/3 倍。

### ❓ Q1：topk_val 为什么要归一化？

**问题**：softmax 输出已经是概率了，top-k 选出来再归一化，不是多此一举？

**答案**：top-k 的概率之和**不等于 1**：

```
softmax 输出 [0.4, 0.3, 0.2, 0.1]（总和=1）
topk(k=2) → [0.4, 0.3]，总和 = 0.7 ≠ 1

不归一化：output = 0.4*e0(x) + 0.3*e1(x) → 输出只有原来的 70%！
归一化后：[0.4/0.7, 0.3/0.7] = [0.571, 0.429] → 权重和为 1 ✓
```

丢弃的概率质量要重新分配给保留的专家，否则多层叠加后信号越来越弱。

### ❓ Q2：专家容量（capacity factor）是什么？溢出怎么办？

**问题**：教学版"无限制"，但生产有 capacity factor，是什么？

**答案**：Capacity factor 限制**每个专家最多处理多少 token**：

```
capacity = 20/专家:
  专家 0: 38 token → 只处理 20 个，溢出 18 个

溢出处理方案:
  1. 丢弃（drop）→ Mixtral 用此法，简单但丢信息
  2. 路由到 shared expert → DeepSeek-V3 用此法，精度损失小
  3. 增大 capacity → 更安全但更费显存
```

### ❓ Q3：LPT 装箱的 4/3 近似比是什么意思？

**问题**："不超过最优解的 4/3 倍"——实际中通常差多少？

**答案**：设最优分配下最重设备的负载为 OPT，LPT 保证最重设备 ≤ 4/3 × OPT：

```
本例：LPT=65, 最优=65 → 65/65 = 1.0（完美！）

4/3 是**最坏情况上界**，实际通常远好于这个值。
当专家数多、负载分布均匀时，LPT 几乎总是达到最优。
```

---

## 5. 教学版 vs 真实框架

| 维度 | 本教程 (教学版) | 真实框架 |
|------|----------------|----------|
| **专家数** | 4 个 | DeepSeek-V3: 256 专家; Mixtral: 8 专家 |
| **top-k** | 2 | DeepSeek-V3: top-8; Mixtral: top-2 |
| **路由粒度** | token 级 | token 级 (主流) |
| **专家位置** | 单机单卡 | 跨多卡,专家并行 (EP) |
| **EPLB** | 贪心装箱,静态映射 | 动态:在线监控 → 触发热专家迁移 |
| **负载均衡辅助损失** | 无 | auxiliary balance loss (防 gate 坍塌) |
| **专家容量** | 无限制 | capacity factor 限制,溢出 token 丢弃或路由到备用 |

### DeepSeek-MoE / DeepSeek-V3

- 256 路由专家 + 1 共享专家,top-8 路由
- 引入 **device-limited routing**:每个 token 的专家必须分布在 ≤ M 台设备,减少 all-to-all 通信

### Mixtral 8x7B

- 8 个专家,每 token 激活 2 个
- 每层参数量 = 8 × FFN,但 FLOPs 仅 2 × FFN
- 使用 auxiliary loss: `loss_balance = α × Σ(f_i × P_i)` 引导均匀路由

### vLLM MoE 支持

- 对 Mixtral/DeepSeek 等模型实现了 fused MoE CUDA kernel
- 支持专家并行 (EP),专家分片到不同 GPU
- 运行时根据请求批量动态调整专家分配

---

## 6. 运行

```bash
cd advanced/adv12_moe_eplb
python run.py
```

预期输出(种子 42):

```
==================================================
adv12: MoE + EPLB 专家负载均衡
==================================================

[路由分布] 每个专家被路由到的 token 数 (共 128 次路由):
  专家 0:  38  ██████████████████████████████████████
  专家 1:  34  ██████████████████████████████████
  专家 2:  29  █████████████████████████████
  专家 3:  27  ███████████████████████████

[不均衡度] max/min 负载比 = 1.407
  ✓ 天然路由确实不均衡 (imbalance > 1.0)

[EPLB 重均衡] 专家 -> 设备映射:
  专家 0(负载  38) -> 设备 0
  专家 1(负载  34) -> 设备 1
  专家 2(负载  29) -> 设备 1
  专家 3(负载  27) -> 设备 0

[各设备负载对比]
  设备          朴素均分      EPLB
  设备 0        72.0      65.0
  设备 1        56.0      63.0

  朴素均分 -> 最大设备负载: 72.0, 方差: 128.00
  EPLB     -> 最大设备负载: 65.0, 方差: 2.00
  ✓ EPLB 后各设备负载更均衡

✅ adv12_moe_eplb 通过
```

**依赖:** 仅 `torch`(纯 CPU,无需 GPU)。

---

## 7. 下一步

**adv13 — Linear Attention**

标准 Attention 的计算复杂度是 O(seq²),当序列很长时成为瓶颈。Linear Attention 通过核函数近似把复杂度降为 O(seq),是超长上下文推理的关键技术。

相关概念:
- 核函数近似:`Q(K^T V)` vs `(QK^T)V`
- 递推式状态更新(类 RNN)
- 代表模型:RetNet、Mamba、GLA
