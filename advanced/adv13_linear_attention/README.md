# adv13 — Linear Attention / SSM 线性注意力

## 1. 教学目标

- 理解标准 softmax attention 的计算瓶颈(O(seq²) 显存)
- 掌握线性 attention 的核心思想:用特征映射 φ 替代 softmax
- 实现并验证**递推形式**与**矩阵形式**的数学等价性
- 对比线性 attention 与 Mamba/SSM 的异同

## 2. 问题:标准 Attention 的 O(seq²) 瓶颈

### 什么是 Linear Attention

**Linear Attention（线性注意力）** 是对标准 Transformer 中 Softmax Attention 的一种替代方案。名字中的"Linear"指的是其**计算复杂度相对于序列长度是线性的 O(seq)**，而非标准 attention 的 O(seq²)。

标准 Attention 的计算流程：`Q·K^T → softmax → ×V`，其中 `Q·K^T` 产生一个 [seq, seq] 的注意力矩阵，这就是 O(seq²) 的来源。

Linear Attention 的核心想法：**能否绕过这个 [seq, seq] 矩阵？** 通过将 softmax 替换为一个可分解的特征映射 φ，利用矩阵乘法的结合律，把计算顺序从 `(Q·K^T)·V` 变为 `Q·(K^T·V)`。后者的中间结果是 [d, d] 的状态矩阵（d 是 head_dim），当 seq >> d 时，计算量和显存占用都大幅降低。

```
标准 Attention:    Q·K^T → [seq, seq] → softmax → ×V → [seq, d]
                           ↑ O(seq²) 瓶颈

Linear Attention:  φ(Q) · (φ(K)^T · V) → [seq, d]
                            ↑ [d, d] 状态矩阵，与 seq 无关
```

这种替换使得 Linear Attention 在推理时可以像 RNN 一样逐 token 递推（每步 O(d²)），而非像标准 attention 那样每步都要回看整个历史（每步 O(seq·d)）。

### 标准 Attention 的瓶颈细节

标准 softmax attention 的计算流程:

```
scores = Q·Kᵀ / √d     # [seq, seq] — 这里是瓶颈
weights = softmax(scores)
output = weights · V
```

显存占用随序列长度**平方增长**:

```
seq=1K  → 注意力矩阵 ~4MB   (fp32)
seq=4K  → ~64MB
seq=16K → ~1GB
seq=32K → ~4GB    ← 单卡 A100(80GB)已捉襟见肘
```

长文本推理(128K tokens)时,O(seq²) 已无法承受,催生了线性 attention 研究。

## 3. 原理:线性 Attention 与状态递推

### 3.0 数学直觉：为什么"换个括号"就能从 O(seq²) 变成 O(seq)

> 以下用 3Blue1Brown 式的"先建立几何直觉，再看公式"的方式来理解。

**从一个具体的数字例子开始。**

假设 seq=4, d=2。标准 attention 要算的核心是：

```
out[i] = Σ_j  weight[i,j] × v[j]

其中 weight[i,j] ∝ exp(q_i · k_j)
```

把它铺开看——你在算一个 4×4 的"关注度矩阵"，然后用它对 V 做加权求和：

```
         k_0   k_1   k_2   k_3
   q_0 [ 0.9   0.1   0.0   0.0 ]     每一行是一个概率分布
   q_1 [ 0.3   0.5   0.2   0.0 ]     行数 = seq → 矩阵大小 seq²
   q_2 [ 0.1   0.2   0.6   0.1 ]
   q_3 [ 0.0   0.1   0.3   0.6 ]

output = 这个矩阵 × V     # [4,4] × [4,2] → [4,2]
```

**问题的本质：** 你在做 `(Q·K^T) · V`。括号里的 `Q·K^T` 是 [seq, seq]，当 seq=128K 时就是 128K×128K = 160 亿个数字。

**关键洞察：矩阵乘法有结合律。**

如果你先算 `K^T · V`（[d, seq] × [seq, d] = [d, d]），再算 `Q · (K^T·V)`（[seq, d] × [d, d] = [seq, d]），你**完全跳过了 [seq, seq] 这个巨型矩阵**：

```
标准顺序:  (Q · K^T) · V
            [seq,seq]         ← 存不下！

换括号后:  Q · (K^T · V)
               [d, d]        ← 只有 128×128 = 16K 个数字！
```

**但是！softmax 毁了结合律。**

标准 attention 不是简单的 `Q·K^T·V`，而是 `softmax(Q·K^T)·V`。softmax 是逐行非线性操作，你没法把它"移到括号外面"——这就像你不能把 `sin(a+b)` 拆成 `sin(a) + sin(b)`。

```
softmax(Q·K^T) · V  ≠  Q · softmax(K^T) · V    ← 非线性破坏了结合律！
```

**Linear Attention 的核心 trick：用可分解的 φ 替代 softmax。**

如果我们不要 softmax，而是用一个逐元素的映射 φ：

```
weight[i,j] = φ(q_i)^T · φ(k_j)    （注意：这是两个向量的内积！）
```

那么：

```
out[i] = Σ_j φ(q_i)^T · φ(k_j) · v[j]
       = φ(q_i)^T · Σ_j ( φ(k_j) · v[j]^T )    ← 结合律！把 j 的求和提出来
       = φ(q_i)^T · S                             ← S 是 [d, d] 的"状态矩阵"
```

**一句话总结**：softmax 是非线性的，堵死了结合律的路；φ 是逐元素映射，保持了向量内积结构，结合律就能用了。这就是"Linear"的本质——不是说模型变简单了，而是计算顺序变成了线性复杂度。

**从"大矩阵"到"滚动记忆"——因果版的直觉。**

自回归生成时，还有因果约束：第 t 步只能看到 1..t 的历史。这时状态矩阵 S 变成一个**逐步累积的记忆**：

```
想象你在读一本书，每读一页就在脑中更新一个"摘要"（S 矩阵）：

  读第 1 页: S₁ = 第1页的要点
  读第 2 页: S₂ = S₁ + 第2页的要点
  读第 3 页: S₃ = S₂ + 第3页的要点
  ...

  当有人提问（q_t）时，你查阅当前的摘要 S_t 来回答：
  answer_t = q_t · S_t

  你不需要翻回去重读所有历史页面！
  → 这就是为什么推理时每步只需 O(d²)，而非 O(seq×d)
```

对比标准 attention：每次有人提问，你都要翻回去逐页重读（O(seq)），然后汇总回答。书越长，每次回答越慢。而 Linear Attention 维护着一份不断更新的摘要，回答只需查阅摘要（O(d²)），与已读页数无关。

### 3.1 用特征映射替代 Softmax

```
标准 attention:
  A[i,j] = exp(q_i · k_j / √d) / Z_i        Z_i = Σ_j exp(q_i · k_j / √d)
  需要完整 [seq, seq] 矩阵

线性 attention:
  A[i,j] ≈ φ(q_i)ᵀ · φ(k_j)              φ 为非负特征映射(如 elu+1)
  利用结合律: Σ_j φ(q_i)ᵀφ(k_j) v_j = φ(q_i)ᵀ (Σ_j φ(k_j) v_jᵀ)
                                                   ^^^^^^^^^^^^^^^^^^^^^^
                                                   状态矩阵 S  [d, d]
```

### 3.2 状态矩阵递推 (Causal / 因果版)

```
ASCII 示意图:

时间步:   t=1        t=2        t=3        ...   t=T
           │          │          │                │
           ▼          ▼          ▼                ▼
k_1,v_1→ S_1  → S_2 ← k_2,v_2  → S_3 ← k_3,v_3  → S_T
           │          │          │                │
          q_1→o_1    q_2→o_2    q_3→o_3          q_T→o_T

S_t = S_{t-1} + outer(φ(k_t), v_t)    # 状态更新: O(d²)
o_t = φ(q_t) @ S_t                     # 输出查询: O(d²)

全序列总复杂度: O(seq × d²)  vs  标准 O(seq² × d)
```

### 3.3 复杂度对比

```
┌──────────────────────┬──────────────────┬────────────────────────┐
│ 方法                 │ 时间复杂度        │ 关键显存               │
├──────────────────────┼──────────────────┼────────────────────────┤
│ standard_attention   │ O(seq² × d)      │ O(seq²) 注意力矩阵     │
│ linear_attention     │ O(seq × d²)      │ O(d²)  状态矩阵 S      │
└──────────────────────┴──────────────────┴────────────────────────┘

当 seq >> d 时(长文本场景),线性 attention 节省大量显存和计算。
```

## 4. 实现细节

### 4.1 `standard_attention(q, k, v)`

标准 softmax attention,教学参考基线:

```python
scores = q @ k.T / sqrt(d)    # [seq, seq]
return softmax(scores) @ v
```

### 4.2 `linear_attention(q, k, v)` — 因果递推形式

```python
qf, kf = φ(q), φ(k)          # 特征映射
S = zeros(d, d)               # 状态矩阵
for t in range(seq):
    S += outer(kf[t], v[t])   # 状态更新
    out[t] = qf[t] @ S        # 查询状态
```

这是**因果**版本:第 t 步只看 1..t 的 k/v,适合自回归生成。

### 4.3 `linear_attention_matrix(q, k, v)` — 非因果矩阵形式

```python
qf, kf = φ(q), φ(k)
S = kf.T @ v        # φ(K)ᵀ V,  [d, d]
return qf @ S       # φ(Q) S,   [seq, d]
```

使用全序列信息,与非因果递推(`linear_attention_noncausal`)数学严格等价
(验证见 run.py 断言 ②)。

**注意**: linear_attention 与 standard_attention 数值不同是**预期行为**:
- `φ(x) = elu(x)+1` 是 `exp(x)` 的近似,但没有 softmax 的归一化(Z 分母)
- 两者的输出形状相同,语义近似,但数值存在差异
- 实际系统(如 RetNet、GLA)会加 normalizer 缓解此差异

### ❓ Q1：为什么 φ(x) = elu(x)+1 能近似 softmax？

**问题**：elu+1 和 exp 看起来完全不一样，怎么能代替？

**答案**：关键不是"近似 exp"，而是**满足非负性 + 内积结构**：

```
线性 attention 核心假设: A[i,j] ≈ φ(q_i)ᵀ · φ(k_j)

需要 φ 满足:
  1. φ(x) ≥ 0（非负，类比 exp(x) > 0）
  2. φ(q)ᵀφ(k) 能捕捉 q 和 k 的相似度

elu(x)+1:
  x > 0 → elu(x)+1 = x+1     （正值线性增长）
  x < 0 → elu(x)+1 ≈ α·exp(x)+1  （趋近正值）

所以 elu+1 保证所有特征值 ≥ 0。但它确实不等于 softmax——没有归一化分母 Z。
```

### ❓ Q2：线性 attention 没有归一化，输出幅值不会爆炸吗？

**问题**：状态矩阵 S 每步累积，seq 越长 S 越大？

**答案**：**会的！** 真实系统的解法：

```
RetNet: 加衰减因子 γ^t → S_t = γ·S_{t-1} + outer(kf_t, v_t)
GLA: 门控机制，动态控制遗忘率
Layer Norm: 输出前归一化
```

教学版没有这些机制，只在短序列上演示形状正确性。

### ❓ Q3：递推形式和矩阵形式等价的条件是什么？

**答案**：**非因果（non-causal）模式**下等价。因果递推（自回归生成用）只看到 1..t 的 k/v，矩阵版（全序列训练用）看到全部，所以数值不同——这是预期的。

### 4.4 `LinearAttentionLayer`

```python
class LinearAttentionLayer(nn.Module):
    def __init__(self, d_model):
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return linear_attention(wq(x), wk(x), wv(x))
```

可直接替换 Transformer 中的 MultiHeadAttention,保持接口兼容。

## 5. 教学版 vs 真实框架

### 5.1 线性 Attention 路线

| 方法         | 特征映射 φ         | 是否因果 | 备注                          |
|------------|-------------------|---------|-------------------------------|
| 本教学       | elu+1             | 因果    | 最简实现                       |
| Performer  | 随机特征近似 exp   | 均可    | FAVOR+ 算法                   |
| RetNet     | 衰减因子 γ        | 因果    | 带遗忘机制                     |
| GLA        | 门控线性 attention | 因果    | 与 SSM 统一                   |
| RWKV       | 指数衰减           | 因果    | 纯 RNN 形式部署                |

### 5.2 SSM / Mamba 路线

Mamba 属于**选择性状态空间模型(Selective SSM)**,与线性 attention 同属"O(seq) 线性递推"家族,但有本质区别:

```
线性 Attention:
  S_t = S_{t-1} + outer(k_t, v_t)         # 固定累积,无遗忘
  输出: q_t @ S_t

Mamba (S4/SSM):
  h_t = A_t · h_{t-1} + B_t · x_t        # A_t 是输入相关的遗忘门(选择性!)
  y_t = C_t · h_t                         # 输出投影

关键差异:
  - Mamba 的 A、B、C 矩阵由输入动态生成(selective),能选择性记忆/遗忘
  - 线性 attention 的 S 是纯累积,没有遗忘机制
  - Mamba 使用 parallel scan 实现高效并行训练,推理时退化为 O(1)/步的 RNN
```

### 5.3 统一视角

```
             ┌─────────────────────────────────────┐
             │      线性递推序列模型家族             │
             │                                     │
             │  Linear Attn  ──  GLA  ──  Mamba    │
             │       ↑              ↑        ↑     │
             │    无遗忘         门控遗忘   选择性   │
             │  φ(Q)(φ(K)ᵀV)    gating    S4 scan  │
             └─────────────────────────────────────┘
                         共同点: O(seq) 推理,O(d²) 状态
```

## 6. 运行

```bash
cd advanced/adv13_linear_attention
python run.py
```

预期输出:

```
[断言①] 形状检查通过: standard=torch.Size([16, 32]), linear=torch.Size([16, 32])
        注意:数值不同——linear_attention 用 φ(x)=elu(x)+1 近似 softmax,...
[断言②] 矩阵形式 vs 非因果递推形式 最大绝对误差: ...e-07
        allclose 通过 (atol=1e-5) ✓
[断言③] LinearAttentionLayer 前向通过: 输出形状=torch.Size([16, 32])
...
✅ adv13_linear_attention 通过
```

## 7. 下一步

**adv14: Multi-LoRA** — 如何在同一个基础模型上同时服务多个 LoRA adapter,
实现低显存开销的多租户 fine-tuned 模型推理。

相关阅读:
- [Transformers are RNNs (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236) — 线性 attention 原论文
- [Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [RetNet (Sun et al., 2023)](https://arxiv.org/abs/2307.08621)
