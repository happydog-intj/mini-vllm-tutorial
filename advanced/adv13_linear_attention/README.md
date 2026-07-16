# adv13 — Linear Attention / SSM 线性注意力

## 1. 教学目标

- 理解标准 softmax attention 的计算瓶颈(O(seq²) 显存)
- 掌握线性 attention 的核心思想:用特征映射 φ 替代 softmax
- 实现并验证**递推形式**与**矩阵形式**的数学等价性
- 对比线性 attention 与 Mamba/SSM 的异同

## 2. 问题:标准 Attention 的 O(seq²) 瓶颈

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
