# step06 — 采样算法：logits → next_token

## 为什么需要采样？Greedy 有什么问题？

最直觉的做法是每步都选概率最高的 token（贪心搜索，greedy）。这在很多任务上有效，但有一个根本性缺陷：**一旦某个 token 被选中，它就永远无法被撤回**。

贪心搜索容易陷入**重复退化**：

```
输入：  "The weather is"
Greedy: "The weather is nice. The weather is nice. The weather is nice. ..."
```

为什么会重复？因为"nice"之后，"The"的概率最高；"The"之后，"weather"最高……模型掉入了局部最优的循环。这不是模型质量问题，而是 argmax 这个操作本身的特性——它把所有概率压缩成了一个确定性选择，丧失了概率分布中其他 token 携带的信息。

**解决思路**：保留概率分布，按分布随机采样，而不是每次都取最大值。

---

## Logits 是什么？

语言模型最后一层输出一个形状为 `[vocab_size]` 的向量，称为 **logits**（原始分数，未归一化）：

```
         vocab_size = 256 (本例)
         ┌──────────────────────────────────────────┐
logits = │ 0.1  -2.3   8.7   0.4  -0.9  ...  1.2  │
         └──────────────────────────────────────────┘
           tok0  tok1  tok2  tok3  tok4       tok255
                        ↑
                    分数最高，代表 'C'

经过 softmax 归一化后变成概率：
prob[i] = exp(logits[i]) / Σ exp(logits[j])

prob ≈ [0.001, 0.000, 0.991, 0.002, 0.001, ...]
                        ↑
                    概率 99.1%
```

`softmax` 是单调变换：logits 中最大的 token 在概率分布中依然最大。但 softmax 是非线性的——logits 的差距越大，概率越向头部集中。

---

## 贯穿全节的示例 logits

下面所有策略都用同一组 6 个 token 的 logits 演示（假设词表只有 6 个词）：

```
token:   A     B     C     D     E     F
logits:  4.0   2.0   1.0   0.5  -1.0  -2.0
```

先用 softmax 算出原始概率（T=1）：

```
exp(logits):  54.6  7.39  2.72  1.65  0.37  0.14   → 合计 = 66.87

probs:        0.816  0.111  0.041  0.025  0.005  0.002
token:          A      B      C      D      E      F
```

A 的概率高达 81.6%，但 B/C/D 也有一定份额。后续各策略都从这里出发。

---

## Temperature：改变分布的尖锐程度

Temperature 不只是"控制随机性"，它有精确的数学含义：**在 softmax 之前将 logits 除以 T**，改变概率分布的形状。

```python
scaled_logits = logits / temperature
probs = softmax(scaled_logits)
```

数学上，这等价于对概率分布做指数变换：

```
原始 probs = [p1, p2, ..., pn]
温度 T 后   = [p1^(1/T), p2^(1/T), ..., pn^(1/T)]  （归一化后）
```

**T < 1（低温）**：logits 被放大，差距拉大，分布更尖锐，高概率 token 获得更多概率质量。

**T = 1**：不改变原始分布。

**T > 1（高温）**：logits 被压缩，差距缩小，分布趋向均匀，"意外" token 更容易被选到。

**T→0 的极限就是 greedy**（最大值独占全部概率）。

**数值示例**（用上面的 6-token logits）：

```
原始 logits:  A=4.0  B=2.0  C=1.0  D=0.5  E=-1.0  F=-2.0

T=0.5  → logits/T:  A=8.0   B=4.0   C=2.0   D=1.0   E=-2.0  F=-4.0
        probs:       A=0.969  B=0.026  C=0.003  D=0.001  E≈0    F≈0
        （A 的概率从 81.6% 升至 96.9%，分布更尖锐）

T=1.0  → probs:      A=0.816  B=0.111  C=0.041  D=0.025  E=0.005  F=0.002
        （原始分布不变）

T=2.0  → logits/T:  A=2.0   B=1.0   C=0.5   D=0.25  E=-0.5  F=-1.0
        probs:       A=0.464  B=0.171  C=0.104  D=0.081  E=0.038  F=0.023
        （A 从 81.6% 降至 46.4%，B/C/D 获得更多概率质量）
```

三种温度下，B 被采样到的概率分别是 2.6%、11.1%、17.1%——温度越高，"意外" token 越容易出现。

```
            T=0.5        T=1.0        T=2.0
概率分布：  █            ▆            ▄▄▄
            ░            ░▄           ▄▄▄▄▄
token:    A B C D      A B C D      A B C D
        （极度尖锐）   （原始）     （趋向均匀）
```

---

## Top-k：固定候选集大小

Top-k 采样先找出概率最高的 k 个 token，将其余 token 的 logits 设为 `-inf`（softmax 后概率为 0），再在这 k 个 token 上做 temperature 采样：

```python
top_k_values, _ = torch.topk(logits, k=min(k, logits.size(-1)))
threshold = top_k_values[-1]           # 第 k 大的值
filtered = logits.masked_fill(logits < threshold, float("-inf"))
return temperature_sample(filtered, temperature)
```

**Top-k 的问题**：候选集大小固定为 k，但不同位置的概率分布差异很大：

```
场景 A（分布尖锐）：         场景 B（分布平坦）：
top-1 概率 = 0.95            top-1 概率 = 0.08
top-2 概率 = 0.03            top-2 概率 = 0.07
top-3 概率 = 0.01            top-3 概率 = 0.07
...                          ...
k=10 → 剩余 k-3 个           k=10 → 但 top-100 概率都差不多
       几乎零概率 token               10 个不够，丢失了大量合理选项
```

Top-k=10 在场景 A 里浪费了 7 个槽位在零概率 token 上；在场景 B 里又把大量合理 token 截掉。**k 是一个对分布形状不敏感的超参数**。

**数值示例**（k=3，T=1.0，同一组 logits）：

```
原始 logits:  A=4.0  B=2.0  C=1.0  D=0.5  E=-1.0  F=-2.0

step1  找 top-3：A(4.0)  B(2.0)  C(1.0)，阈值 = 1.0
step2  D/E/F logits 设为 -inf：
       过滤后:  A=4.0  B=2.0  C=1.0  D=-inf  E=-inf  F=-inf
step3  对剩余 3 个做 softmax：
       exp:     54.6   7.39   2.72
       probs:   A=0.843  B=0.114  C=0.042
```

原来 D/E/F 合计只有 3.2% 的概率，被截掉后，A/B/C 按比例重新归一化，A 从 81.6% 升至 84.3%。

---

## Top-p（Nucleus Sampling）：自适应候选集

Top-p 换了一个视角：不固定候选集数量，而是固定**候选集覆盖的概率质量**。

算法：按概率从大到小排序，累加概率，一旦累积概率超过 p，截掉后面所有 token：

```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(softmax(sorted_logits), dim=-1)

# 移除累积概率已超过 p 的部分（注意：减去当前值，保留边界token）
sorted_remove = cumulative_probs - softmax(sorted_logits) > p
sorted_logits[sorted_remove] = float("-inf")
```

对同样两个场景，top-p=0.9 的行为：

```
场景 A（分布尖锐）：              场景 B（分布平坦）：
tok0=0.95 → 累积 0.95 ≥ 0.9     tok0=0.08 → 累积 0.08
候选集 = {tok0}，只有 1 个        tok1=0.07 → 累积 0.15
                                  ...
                                  tok11=0.06 → 累积 0.91 ≥ 0.9
                                  候选集 = 12 个 token
```

**候选集大小随分布自动调整**，这是 top-p 比 top-k 更常用的原因。

**数值示例**（p=0.9，T=1.0，同一组 logits）：

```
按概率从大到小排序：
  token   prob   累积概率
  A       0.816   0.816   ← 还未超过 0.9，保留
  B       0.111   0.927   ← 累积超过 0.9，但 B 是让累积刚超过的那个，保留
  C       0.041   0.968   ← 累积已超过 0.9（上一步累积 0.927 - 0.041 = 0.886 < 0.9），保留
  D       0.025   0.993   ← 上一步累积 0.968 > 0.9，截掉
  E       0.005    ...    ← 截掉
  F       0.002    ...    ← 截掉

候选集 = {A, B, C}，对这 3 个重新 softmax：
  probs:  A=0.843  B=0.114  C=0.042
```

对比 top-k=3 的结果完全一致——因为这组 logits 的前 3 个 token 恰好覆盖了 90%+ 的概率质量。但如果分布更平坦，top-p 会自动纳入更多候选；top-k 则固定只留 3 个。

Top-p 和 top-k 可以叠加使用：先 top-k 限制绝对上限，再 top-p 动态收紧。

---

## Gumbel-Max Trick：为什么数学等价？

标准 temperature 采样的步骤：

```
logits → 除以 T → softmax → multinomial 采样
```

Gumbel-Max Trick 绕过了 softmax 和 multinomial，只需：

```
next_token = argmax(logits / T + gumbel_noise)
```

其中 `gumbel_noise[i] = -log(-log(U[i]))`，`U[i] ~ Uniform(0,1)`。

**为什么等价？** Gumbel 分布有一个关键性质：

> 若 `x_i` 是独立的 Gumbel 分布随机变量，均值为 `log(p_i)`，  
> 则 `argmax(x_i)` 服从以 `p_i` 为权重的离散分布。

把 `logits[i]/T` 看作第 i 个类别的对数概率（差一个常数），加上 Gumbel 噪声后取 argmax，恰好等价于从 `softmax(logits/T)` 中采样。

代码实现：

```python
def gumbel_max_sample(logits: Tensor, temperature: float = 1.0) -> Tensor:
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-20)))
    return torch.argmax(logits / temperature + gumbel_noise)
```

**实际价值**：在 GPU 上，`argmax` 可以完全并行化，而 `multinomial`（需要归一化 + 累积分布 + 二分查找）在 batch 推理中会形成瓶颈。Gumbel-Max 把采样变成了一个向量加法 + argmax，与 greedy 的计算图几乎相同，对已有推理优化友好。

**代价**：需要生成与 vocab_size 等大的随机数向量，内存访问量更大。在极大 vocab（如 200k token 的模型）上需权衡。

---

## 策略对比总结

| 策略 | 核心操作 | 候选集大小 | 适用场景 |
|------|----------|------------|----------|
| Greedy | `argmax(logits)` | 1（固定） | 代码生成、需要确定性输出 |
| Temperature | `softmax(logits/T)` + 采样 | 全词表 | 通用，配合 top-k/p 使用 |
| Top-k | 保留 top-k，再 temperature 采样 | 固定 k | 分布相对均匀的场景 |
| Top-p | 保留累积概率 ≥ p，再 temperature 采样 | 自适应 | 通用，生产系统常用默认值 |
| Gumbel-Max | `argmax(logits/T + gumbel)` | 全词表（等价） | batch 推理，替代 temperature 采样 |

---

## NaiveEngine 的采样分派逻辑

`engine.py` 中的 `generate()` 按参数选择采样策略：

```python
if temperature == 0:          # T=0 → greedy
    next_id = greedy_sample(logits)
elif use_gumbel:              # 显式指定 gumbel
    next_id = gumbel_max_sample(logits, temperature)
elif top_k > 0:               # top-k 优先
    next_id = top_k_sample(logits, top_k, temperature)
elif top_p < 1.0:             # 其次 top-p
    next_id = top_p_sample(logits, top_p, temperature)
else:                         # 纯 temperature
    next_id = temperature_sample(logits, temperature)
```

注意：这里 top-k 和 top-p 是互斥的（先检查 top_k > 0），实际生产中两者通常叠加。

---

## 运行

```bash
python run.py
```

预期输出（`logits[65]=10.0`，其余随机）：

```
logits 峰值在 token 65 ('A'), 值=10.0

Greedy:          token  65  (确定性)
Temperature=0.1: token  65  (低温→集中在高概率Token)
Temperature=2.0: token  ???  (高温→更随机)
Top-k (k=10):    token  ???  (从概率最高的10个里选)
Top-p (p=0.9):   token  ???  (从累积概率90%的Token里选)
Gumbel-Max:      token  ???  (等价于temperature采样)

✅ step06_sampler 通过
```

Greedy 和低温（0.1）必然选 token 65；高温和其他策略结果取决于随机种子。

---

## 下一步

step08 要解决的新问题：**KV Cache**。

当前的 `NaiveEngine` 每生成一个新 token，都要把整个已生成序列重新喂给 Transformer 做全量 attention 计算。生成第 n 个 token 时，前 n-1 个 token 的 Key/Value 矩阵其实没有变化，却被重复计算了 n 次。

step08 引入 KV Cache，把历史 token 的 K/V 矩阵缓存起来，让每步推理只计算最新 token 的 attention，把计算量从 O(n²) 降到 O(n)——这是 LLM 推理加速的核心机制。
