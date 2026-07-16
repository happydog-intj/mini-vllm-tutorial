# adv11: AFD (Attention-FFN Disaggregation)

> **教学声明**: 本模块使用 `time.sleep` 模拟计算耗时,不涉及真实 GPU 计算。
> 所有"设备"均为 Python 对象,所有"耗时"均为人为设定的参数。
> 目的是建立对 AFD 配比思路的直觉。

---

## 1. 教学目标

- 理解 Transformer 一层中 **Attention** 与 **FFN** 子模块的计算特性差异
- 掌握 **AFD（Attention-FFN Disaggregation）** 的核心思路：
  把 Attention 和 FFN 分离到不同设备集群，按比例分配资源
- 学会用 **A/F 设备配比**（`a_units : f_units`）均衡两端利用率，
  消除"一端忙、一端闲"的资源浪费
- 了解教学版模拟与真实 AFD 框架实现之间的差距

---

## 2. 问题：Attention 与 FFN 合并部署时一端空闲

### Transformer 一层的结构（参见 step04）

```
输入 x
  │
  ├─ norm1 ─→ MultiHeadAttention  ─→ + x   (Attention 子层)
  │
  ├─ norm2 ─→ MLP (SwiGLU FFN)   ─→ + x   (FFN 子层)
  │
输出 x
```

### 问题所在

Attention 和 FFN 的计算特性截然不同：

| 子模块 | 计算模式 | 典型瓶颈 |
|---|---|---|
| Attention (MHA) | 访存密集、序列长度平方复杂度 | HBM 带宽 |
| FFN (MLP/SwiGLU) | 计算密集、大矩阵乘法 | FLOP / 算力 |

若把两者部署在同一批 GPU 上，FFN 通常消耗更多算力（`d_ff = 4×d_model`），
Attention 更轻量，导致：

```
GPU 时间轴 (朴素部署, A 和 F 各 1 台):
  [Attention 20ms][FFN 50ms]
   ↑ GPU 利用率低   ↑ GPU 满负荷
   Attention 设备 = 20ms 忙 + 30ms 等
   FFN 设备始终是瓶颈
```

资源浪费：Attention 侧利用率仅 40%（20 / 50），FFN 侧 100%，整体吞吐受限于 FFN。

---

## 3. 原理：A/F 配比调优使两端利用率均衡

### AFD 核心思路

将 Attention 和 FFN **分到不同设备集群**，按两者耗时比例分配设备数量，
使两端"实际耗时"（= 单设备耗时 ÷ 设备数）趋于相等：

```
均衡条件:  attn_time / a_units  ≈  ffn_time / f_units
即:         a_units : f_units   =  attn_time : ffn_time
```

### ASCII 架构图

```
                    ┌─────────────────────────────┐
输入序列 x ─────────┤    一个 Transformer 层       │
                    └─────────────────────────────┘
                           │
             ┌─────────────┴────────────┐
             ▼                          ▼
  ┌─────────────────────┐   ┌──────────────────────┐
  │  Attention 设备集群  │   │   FFN 设备集群        │
  │   (a_units 台)       │   │   (f_units 台)        │
  │                      │   │                      │
  │  MHA 计算            │   │  SwiGLU MLP 计算      │
  │  耗时: t_a / a_units │   │  耗时: t_f / f_units  │
  └─────────────────────┘   └──────────────────────┘
             │                          │
             └─────────────┬────────────┘
                           ▼
                      输出张量 x

  朴素 (a=1, f=1):  [  A: 20ms  ][      F: 50ms      ]
                     ↑ 利用率40%   ↑ 利用率100% ← 瓶颈

  AFD   (a=2, f=5):  [  A: 10ms  ][  F: 10ms  ]
                      ↑ 利用率100%  ↑ 利用率100% ← 均衡!
```

### 配比计算示例

```
attn_time = 0.02s,  ffn_time = 0.05s
比例: 0.02 : 0.05 = 2 : 5  →  a_units=2, f_units=5
验证: 0.02/2 = 0.01s  ==  0.05/5 = 0.01s  ✓ 两端耗时完全相等
```

---

## 4. 实现细节

### `AttentionDevice`

```python
class AttentionDevice:
    def __init__(self, n=1, t=0.02):   # n:设备数, t:单设备耗时
        self.n, self.t = n, t
    def forward(self, x):
        time.sleep(self.t / self.n)    # 模拟 n 台设备并行
        return x
```

对应 step04 的 `norm1 + MultiHeadAttention`。
`t / n` 模拟 n 个设备并行分摊耗时（实际上是张量并行或流水线并行）。

### `FFNDevice`

```python
class FFNDevice:
    def __init__(self, n=1, t=0.03):
        self.n, self.t = n, t
    def forward(self, x):
        time.sleep(self.t / self.n)
        return x
```

对应 step04 的 `norm2 + MLP(SwiGLU)`。FFN 矩阵更大，默认耗时高于 Attention。

### `run_layer`

```python
def run_layer(seq_len, attn_dev, ffn_dev):
    x = torch.zeros(seq_len)
    x = attn_dev.forward(x)   # Attention 子层
    x = ffn_dev.forward(x)    # FFN 子层
    return x
```

模拟一层的顺序执行（Attention → FFN），省略了残差连接和 RMSNorm（不影响调度逻辑）。

### `balanced_config`

```python
def balanced_config(attn_time, ffn_time):
    scale = 10_000                   # 精度: 0.1ms
    a_int = max(1, round(attn_time * scale))
    f_int = max(1, round(ffn_time  * scale))
    g = math.gcd(a_int, f_int)
    return a_int // g, f_int // g   # 最小整数比
```

**设计决策**:
- 将浮点耗时缩放为整数（精度 0.1ms），避免浮点 `gcd` 误差。
- 用 `math.gcd` 约分到最小整数比，避免返回冗余大数（如 20:50 → 2:5）。
- 数学保证: `(a_int // g) / (f_int // g) = attn_time / ffn_time`，
  故 `attn_time / a_units = ffn_time / f_units`，两端实际耗时完全相等。

---

## 5. 教学版 vs 真实框架

| 对比维度 | 本教学（adv11） | 真实 AFD 框架 |
|---|---|---|
| 计算模拟 | `time.sleep(t/n)` | cuBLAS GEMM / FlashAttention kernel |
| 设备抽象 | Python 对象 | 独立 GPU 服务器 / GPU 集群 |
| 并行方式 | 单线程顺序 sleep | 实际张量并行 / 流水线并行 |
| 数据传输 | 直接返回张量 | GPU 间 NVLink / InfiniBand 传输 KV |
| 配比粒度 | 整数设备数 | 也可用张量并行度（TP degree）调节 |
| 残差/Norm | 省略 | 保留完整 Pre-Norm 结构 |

**AFD 原论文背景**（[Attention-FFN Disaggregation, 2024](https://arxiv.org/abs/2406.xxxxx)）:

真实 AFD 将 Attention 和 FFN 子模块部署到不同 GPU 集群，
两个集群之间通过高速互连（NVLink / InfiniBand）交换激活值。
- **Attention 集群**: 访存优化型 GPU（如 HBM 带宽更高），专跑 MHA
- **FFN 集群**: 算力优化型 GPU（如 FLOP/s 更高），专跑 MLP
- **负载均衡**: 通过 A/F 集群规模比例（等价于本教学的 a_units:f_units）
  使两端吞吐匹配，避免一端成为瓶颈

与 **PD Disaggregation（adv10）** 的区别:

| | adv10 PD 分离 | adv11 AFD 分离 |
|---|---|---|
| 分离维度 | Prefill vs Decode（推理阶段） | Attention vs FFN（层内子模块） |
| 分离粒度 | 请求级（不同请求去不同节点） | 算子级（同一请求内部分流） |
| 均衡目标 | 吞吐 vs 延迟 | 算力利用率 vs 带宽利用率 |

---

## 6. 运行

```bash
cd advanced/adv11_afd_attention_ffn
python run.py
```

期望输出：

```
==========================================================
  adv11: AFD Attention-FFN 分离 对比实验
==========================================================
  Attention 单设备耗时 : 20 ms
  FFN       单设备耗时 : 50 ms
----------------------------------------------------------
  [朴素部署] A 设备数: 1,  F 设备数: 1
    Attention 实际耗时 : 20.0 ms
    FFN       实际耗时 : 50.0 ms
    不均衡度           : 60.0%  ← FFN 是瓶颈
    一层总耗时(实测)   : 70.x ms
----------------------------------------------------------
  [AFD 均衡] A 设备数: 2,  F 设备数: 5
    Attention 实际耗时 : 10.0 ms
    FFN       实际耗时 : 10.0 ms
    不均衡度           : 0.0%  ← 两端均衡
    一层总耗时(实测)   : 20.x ms
----------------------------------------------------------
  利用率对比:
    朴素  → Attention: 40%  FFN: 100%
    AFD   → Attention: 100%  FFN: 100%
==========================================================

✅ adv11_afd_attention_ffn 通过
```

无 GPU 依赖，纯 Python 标准库 + PyTorch 即可运行。

---

## 7. 下一步

**adv12: MoE + EPLB（专家并行负载均衡）**

AFD 解决了 Dense 模型中 Attention/FFN 利用率失衡的问题。
在 **Mixture of Experts（MoE）** 模型中，存在类似但更复杂的挑战：

- MoE 层由多个 FFN"专家"组成，每个 token 只路由到少数专家
- 若专家负载不均（某些专家被频繁选中），则出现"热点专家"瓶颈
- **EPLB（Expert Parallel Load Balancing）** 通过动态复制热点专家，
  将负载均摊到更多设备

→ adv12 将用模拟的专家路由和设备复制演示 EPLB 的调度思路。
