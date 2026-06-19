# step03a — 单请求 KV Cache

## 为什么需要 KV Cache？

在 step01 的朴素推理中，每生成一个新 token，引擎都要把整个已有序列重新跑一遍前向：

```
生成第 1 个新 token：输入 [t0,t1,t2,t3,t4]           → 5 步注意力计算
生成第 2 个新 token：输入 [t0,t1,t2,t3,t4,t5]         → 6 步注意力计算
生成第 3 个新 token：输入 [t0,t1,t2,t3,t4,t5,t6]       → 7 步注意力计算
...
生成第 n 个新 token：输入长度 = 5+n                    → (5+n) 步注意力计算
```

总计算量随序列长度平方增长：O(n²)。生成越长，越慢。

这是 step01 的根本性能瓶颈。KV Cache 通过**缓存已算过的中间结果**来消除重复计算。

---

## 核心洞察：K 和 V 只依赖自身 token

Transformer 注意力的计算公式：

```
Q_i = x_i · W_Q
K_i = x_i · W_K    ← 只和 token i 自身的向量 x_i 有关
V_i = x_i · W_V    ← 只和 token i 自身的向量 x_i 有关

注意力输出：Attn(Q, K, V) = softmax(Q · K^T / √d) · V
```

关键：**K_i 和 V_i 只取决于 x_i**，而 x_i 只包含 token i 的信息（位置编码 + 词嵌入）。

不管后续来多少新 token，**历史 token 的 K/V 值永远不会改变**。

反观 Q：当新 token 进来，它要对所有历史 K 做点积，Q 本身只在当前步存在，不需要缓存。

所以：
```
K_i / V_i → 计算一次，永久缓存 ✅
Q_i       → 只用一次，不需要缓存 ✅
```

---

## Prefill vs Decode：两阶段生成

KV Cache 把生成过程分成截然不同的两个阶段：

```
Prefill 阶段（处理 prompt，一次前向）
─────────────────────────────────────
  输入: [t0, t1, t2, t3, t4]   ← 整个 prompt，一次性喂入

  每层注意力计算：
    K0,V0 | K1,V1 | K2,V2 | K3,V3 | K4,V4

  输出:
    - 最后一个位置的 logits → 采样得到第一个新 token t5
    - past_key_values       → 所有层的 K/V 缓存下来

Decode 阶段（逐步生成，每步只传 1 个 token）
─────────────────────────────────────────────
  Step 1: 输入 [t5]（仅 1 个 token）
    新算: K5 = t5·W_K，  V5 = t5·W_V
    拼接: K_full = [K0,K1,K2,K3,K4, K5]  ← cat(past, 新K)
    注意力: Q5 对 K_full 做点积          ← 历史 K/V 从缓存读，不重算
    输出: logits → 采样 t6，更新 past_key_values

  Step 2: 输入 [t6]
    新算: K6，V6
    K_full = [K0,...,K5, K6]
    ...以此类推
```

每个 Decode 步骤，矩阵乘法的计算量只与当前新 token（1个）有关，**不随序列长度增长**。

---

## 计算量分析：从 O(n²) 到 O(n)

**朴素推理**（step01）：
```
生成 n 个新 token，第 k 步序列长度 = prompt_len + k

注意力矩阵乘法 Q·K^T：
  第 1 步: (1 × d) · (d × (L+1)) → O(L)
  第 2 步: (1 × d) · (d × (L+2)) → O(L+1)
  ...
  第 n 步: (1 × d) · (d × (L+n)) → O(L+n-1)

但朴素引擎连 Q/K/V 都重算，第 k 步要算 (L+k) 个 token 的 Q/K/V
总计算量 ∝ L + (L+1) + ... + (L+n) = O(n² + n·L)
```

**KV Cache**：
```
Prefill 一次性算好所有历史 K/V（固定开销 O(L²)）

Decode 每步：
  - 只算 1 个新 token 的 K/V：O(d)
  - Q 与全序列 K 的点积：O(L+k) ← 这部分仍然线性增长

每步计算量 = O(d + L + k)，n 步总计 = O(n·L + n²)

注意：Q·K^T 的计算量（O(n·L)项）无法消除
真正消除的是：历史 token K/V 的重复计算（从 O(n²) 到 O(L)）
```

换句话说，KV Cache 消除了"**每步重新投影历史 token**"的冗余，但注意力分数本身仍需全量计算。

---

## past_key_values 的数据结构

本步实现中，`past_key_values` 是一个列表，每个 Transformer 层保存一份 (K, V) 元组：

```python
# model.py 中的类型定义
KVCache = Tuple[Tensor, Tensor]  # (K, V)

# past_key_values 的结构：
past_key_values: List[KVCache]

# 以本步模型参数为例（2层，4头，d_head=32）：
past_key_values = [
    # 第 0 层
    (K_layer0,  V_layer0),   # 各形状 [total_seq_len, num_heads, d_head]
    # 第 1 层
    (K_layer1,  V_layer1),   # 各形状 [total_seq_len, num_heads, d_head]
]
```

每个 Decode 步，`total_seq_len` 加 1，K/V 张量通过 `torch.cat` 追加：

```python
# model.py: MultiHeadAttentionWithKVCache.forward()
if past_kv is not None:
    K_past, V_past = past_kv
    K_full = torch.cat([K_past, K], dim=0)   # [old_len+1, heads, d_head]
    V_full = torch.cat([V_past, V], dim=0)
else:
    K_full = K   # Prefill：直接使用全量
    V_full = V
```

---

## Attention 代码修改：step01 vs step03a

为了支持 KV Cache，Attention 层需要改动三处。对比两步的代码：

### 改动一：函数签名新增 past_kv 参数

```python
# step01 的 MultiHeadAttention.forward：
def forward(self, x: Tensor) -> Tensor:
    # 无历史缓存概念，每次从头算

# step03a 的 MultiHeadAttentionWithKVCache.forward：
def forward(
    self,
    x: Tensor,                         # [seq_len, d_model]
    past_kv: Optional[KVCache] = None, # ← 新增！None=Prefill，有值=Decode
) -> Tuple[Tensor, KVCache]:           # ← 新增！返回值多了新的 KV
```

`past_kv=None` 时是 Prefill（计算整个序列），有值时是 Decode（只计算新 token，历史从缓存读）。

### 改动二：KV 矩阵的拼接

```python
# step01：直接用当前输入的 K/V
K_full = K   # [seq_len, heads, d_head]
V_full = V

# step03a：把历史 K/V 和当前新 token 的 K/V 拼接
if past_kv is not None:
    K_past, V_past = past_kv           # 历史：[old_len, heads, d_head]
    K_full = torch.cat([K_past, K], dim=0)   # 拼接 → [old_len+1, heads, d_head]
    V_full = torch.cat([V_past, V], dim=0)
else:
    K_full = K   # Prefill，直接使用
    V_full = V
```

**这里是 KV Cache 的核心**：Decode 时 `x` 只有 1 个 token，算出 1 个新的 K/V，然后拼到历史上，注意力对全部历史 K/V 做点积。

### 改动三：因果掩码的调整

step01 中因果掩码的逻辑很简单——上三角全部屏蔽：

```python
# step01：seq_len × seq_len 的掩码，上三角置 -inf
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))
```

step03a 中需要处理 Prefill 和 Decode 两种情况：

```python
# step03a：
past_len = total_len - seq_len   # 历史长度（Decode 时 seq_len=1）
                                  # Prefill 时 past_len=0，Decode 时 past_len>0

# scores 形状：[seq_len, total_len]
# 例如 Decode 时：[1, old_len+1]
# mask[i, j] = True 表示位置 i 不能看到位置 j

mask = torch.ones(seq_len, total_len, dtype=torch.bool)
for i in range(seq_len):
    # 位置 i 可以看到：0 到 (past_len + i) 之间的所有历史
    mask[i, :past_len + i + 1] = False   # False = 允许看到

scores = scores.masked_fill(mask, float("-inf"))
```

**Decode 时（seq_len=1）**：
- `mask[0, :total_len]` 全部设为 False
- 新 token 能 attend 到所有历史 token（包括自己），完全不屏蔽
- 这是正确的：新 token 是当前序列末尾，可以看所有历史

**Prefill 时（seq_len=prompt_len，past_len=0）**：
- 退化为标准的因果掩码，与 step01 行为完全一致

### 改动四：返回值多了 new_kv

```python
# step01：只返回注意力输出
return self.W_o(concat)

# step03a：同时返回更新后的 KV 缓存
return self.W_o(concat), (K_full, V_full)
# K_full/V_full 包含历史+当前新 token，下一步 Decode 时作为 past_kv 传入
```

这四处改动合在一起，让 Attention 支持了 KV Cache 的增量计算。

---



`engine.py` 中 `KVCacheEngine.generate()` 清晰体现了两阶段：

```python
# Prefill：传入完整 prompt，past_key_values=None
logits, past_key_values = self.model(prompt_ids, past_key_values=None)
next_id = self._sample(logits[-1], temperature)

# Decode：每步只传 1 个 token，传入上一步的 past_key_values
for _ in range(max_new_tokens - 1):
    logits, past_key_values = self.model(
        next_id.unsqueeze(0),        # ← 形状 [1]，只有 1 个 token
        past_key_values=past_key_values,
    )
    next_id = self._sample(logits[-1], temperature)
```

对比 step01 的朴素引擎，每步 Decode 的输入从"完整序列"缩小到"1 个 token"。

---

## 显存代价：内存换时间的权衡

KV Cache 不是免费的午餐，它以**显存**换取**计算时间**：

```
每层每个 token 的 KV 缓存大小：
  K: [1, num_heads, d_head] = 1 × 4 × 32 = 128 个浮点数
  V: [1, num_heads, d_head] = 128 个浮点数
  合计：256 × 4字节(fp32) = 1 KB / token / 层

生成 100 个 token，2 层模型：
  缓存大小 = 100 × 2 × 1KB = 200 KB（本教程模型，很小）

真实大模型（如 70B 参数规模，80层，GQA 8个KV头，d_head=128）：
  每 token = 80层 × 8heads × 128 × 2(K+V) × 2字节(bf16) ≈ 320 KB
  生成 4096 个 token = 约 1.3 GB（仅一个请求！）
```

这就是为什么大模型推理时，**显存（HBM，显卡上的高带宽内存）常常是瓶颈**：
- 不开 KV Cache：计算密集，矩阵乘法充分利用计算单元
- 开 KV Cache：内存密集，每步都要从显存读出全部历史 K/V

随着请求数量增加，KV Cache 占用的显存会快速耗尽。这是 step03b 要面对的问题。

---

## 运行

```bash
python run.py
```

预期输出：
```
============================================================
KV Cache 效果 — NaiveEngine vs KVCacheEngine
============================================================
  生成长度      NaiveEngine   KVCacheEngine      加速比
------------------------------------------------------------
    10tokens          ...ms          ...ms         ...×
    30tokens          ...ms          ...ms         ...×
    50tokens          ...ms          ...ms         ...×

→ 序列越长，KV Cache 加速越明显 ✅
两种引擎生成结果完全一致 ✅

✅ step03a_kvcache_single 通过
```

`run.py` 同时验证两点：
1. **KVCacheEngine 比 NaiveEngine 更快**（序列越长，优势越明显）
2. **两个引擎输出完全相同**（KV Cache 只是优化，不改变数学结果）

---

## 下一步

step03a 解决了单个请求的重复计算问题。

但实际推理服务需要**同时服务多个用户**，每个请求的 prompt 长度不同、生成进度不同。
直接把多个请求的 KV Cache 拼在一起，会遇到显存碎片化和序列对齐的问题。

→ **step03b**：多请求 Batch 推理——如何让多个请求共享一次 GPU 前向，同时管理各自的 KV Cache？
