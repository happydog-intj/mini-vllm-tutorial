# step07 — 前缀缓存 Prefix Caching

## 为什么需要前缀缓存？

在实际部署中，很多请求共享相同的开头部分：

- **系统提示词（System Prompt）**：每个对话都以同一段角色设定开头，可能长达数百个 token
- **Few-shot 示例**：RAG 检索到的文档片段被拼在所有请求的前面
- **模板前缀**：固定格式的 prompt（如"请用中文回答，格式如下：……"）

没有前缀缓存时，每个请求都要对这段共享前缀做一次完整的 Prefill（前向计算），产生重复的矩阵乘法。当系统提示词占 prompt 总长度的大部分时，这些重复计算是纯粹的浪费。

前缀缓存的思路：**把已经算好的前缀 K/V 存下来，下一个请求命中时直接复用，跳过对应的 Prefill 计算。**

## 核心前提：为什么相同前缀的 K/V 完全相同？

在开始讨论如何实现之前，先确认一件事：相同前缀的 K/V 真的完全相同吗？

Transformer 中，第 `i` 个位置的 Key 和 Value：

```
K_i = W_K · x_i        （x_i 是位置 i 的隐状态，第一层 = token embedding + position embedding）
V_i = W_V · x_i
```

`K_i` 只取决于位置 `i` 的 token ID 和位置编码。如果两个请求在位置 `0..n-1` 的 token 完全相同，
这 `n` 个位置的 **第一层** K/V 数值上完全一致——这一点直接成立。

但 LLM 有多层（Qwen3-0.6B 有 28 层），后面各层的 x_i 经过注意力层汇聚了前面 token 的信息，
**只要前缀 token 序列完全相同，每一层的 x_i 值都相同，因此每层的 K_i/V_i 也都相同。**

```
请求A: [SYS(0..99)] + [问题A(100..119)]
请求B: [SYS(0..99)] + [问题B(100..114)]
           ↑
     位置 0~99 的 token 相同 + 位置编号相同
     → 所有 28 层的 K_0..K_99, V_0..V_99 完全相同
     → 只需计算一次，其余请求直接复用 ✅
```

这个性质要求前缀出现在**相同的起始位置**（位置 0 开始）。如果系统提示词被拼接在不同的偏移量处，
位置编码不同，K/V 就不同，无法复用。

## 实现前缀缓存需要解决的问题

有了这个思路之后，马上出现三个问题：

### 问题一：存在哪里？

K/V Cache 是张量，天然存在显存（GPU 内存）里。前缀缓存就是把「已经计算好的 K/V」**留在显存中不释放**，等下一个请求来时直接用。

代价是显存永久被占用。生产系统用 LRU 和引用计数（ref_count）管理淘汰，本步实现简化为永不淘汰。

### 问题二：存多细？

K/V Cache 是按 token 逐步增长的，理论上可以缓存任意长度的前缀。但太细（每个 token 一条缓存条目）会让字典很大、查找慢；太粗（整个 prompt 一条）命中率低。

实践中按**固定大小的 Block**（如 16 个 token）为单位缓存，命中时复用整块，未命中的尾部再临时计算。这与 step06 的 PagedAttention 分块完全对应，两者共用同一套 Block 管理机制。

### 问题三：如何快速判断「这段前缀是否已经缓存过」？

这是核心难题。对于一个新请求的 prompt，我们需要在毫秒级别内：

1. 判断它的前缀有没有缓存
2. 找到最长的已缓存前缀（而不只是判断有没有）

**方案 A：逐 token 对比**

把 prompt 的 token 序列和缓存字典里所有已存的 token 序列逐一比较。
代价：O(缓存条目数 × 前缀长度)，随缓存增大线性变慢，不可行。

**方案 B：直接用 token 序列作为字典 key**

```python
cache_dict[tuple(tokens[0:n])] = past_kv
```

查找 O(1)，但构造 key 本身需要 O(n)，且每次尝试不同长度都要重新构造 key。
更大的问题：内存里存了大量的 token 序列副本。

**方案 C：对每段前缀计算一个 hash，用 hash 作为 key（采用）**

```python
h = hash(tokens[0:n])
cache_dict[h] = past_kv
```

hash 是固定大小的整数（64位），构造 key O(n) 但只做一次，查找 O(1)。
需要解决的问题：用什么 hash 函数？如何处理相同 Block 内容但不同前缀的情况？

→ 这就引出了下一节的链式 xxhash 设计。



Transformer 中，第 `i` 个位置的 Key 和 Value 的计算方式是：

```
K_i = W_K · (token_embedding[i] + position_embedding[i])
V_i = W_V · (token_embedding[i] + position_embedding[i])
```

这里有两个关键点：

1. `K_i` 只取决于**位置 `i` 的 token ID** 和**位置 `i` 的位置编码**
2. 如果两个请求在位置 `0..n-1` 的 token 完全相同，那么这 `n` 个位置的 K/V 张量**数值上完全一致**

因此：

```
请求A: [SYS(0..99)] + [问题A(100..119)]
请求B: [SYS(0..99)] + [问题B(100..114)]
           ↑
     位置 0~99：token相同 + 位置相同
          → K_0..K_99, V_0..V_99 完全相同
          → 只需计算一次，其余请求直接复用
```

注意这个性质依赖于**非因果注意力位置编码**（旋转位置编码 RoPE 或绝对位置编码均满足此条件）——相同 token 在相同位置，K/V 就相同。

## 链式 xxhash：推导过程

### 第一步：为什么需要 hash？

前缀缓存的核心操作是：**给定一段 token 序列，快速查找是否有对应的 K/V 已经缓存。**

最直接的方案是把 token 序列本身作为字典的 key：

```python
cache[(72, 101, 108, 108, 111, ...)] = past_kv  # token tuple 作为 key
```

问题：token 序列可能有几百上千个元素，每次查找都要比较整个元组，效率低。
更重要的是，我们需要找**最长的匹配前缀**，要从长到短逐一尝试，每次都要构造一个新的 key。

用 hash 可以把任意长度的 token 序列映射为一个固定大小的整数（64位），查找 O(1)。

### 第二步：为什么不对每个 Block 独立 hash？

把 prompt 按 `block_size=16` 切块，对每块独立计算 hash：

```python
h_block0 = xxhash64(tokens[0:16])   # 只看这16个token
h_block1 = xxhash64(tokens[16:32])  # 只看这16个token
h_block2 = xxhash64(tokens[32:48])
```

乍看可行，但存在一个根本问题：**相同 Block 内容 + 不同前缀 = 相同 hash，但缓存不能复用。**

举个例子：

```
请求 A: [系统提示词版本1(block0)] + [用户问题X(block1)] + [更多内容...]
请求 B: [系统提示词版本2(block0)] + [用户问题X(block1)] + [更多内容...]
                  ↑ block0 不同                ↑ block1 的 token 完全相同
```

`h_block1` 在两个请求中相同（block1 的 token 一模一样），但能把请求A的 block1 缓存给请求B用吗？

**不能。**

原因在于前缀缓存的使用方式：调用模型时，你传入的是 `past_key_values`，它代表从位置 0 到当前位置的**连续 K/V 序列**：

```python
# 请求B复用请求A的block1时，会这样调用：
logits, new_kv = model(
    tokens_from_block2,
    past_key_values=cached_kv_from_A_block1  # 这里的kv是基于请求A的block0计算出来的
)
```

这意味着：请求B用了请求A的 block1 KV Cache，但请求A的 block1 KV 是在**请求A的 block0 计算完之后**的上下文里生成的（序列位置 16-31，且模型内部的残差流包含了 block0 的信息流）。如果两个请求的 block0 不同，直接复用 block1 的 KV 是错误的——**K/V 并不只取决于当前 block 的 token，还隐含了整个序列的上下文依赖。**

> 等等——之前不是说 K_i 只取决于 token_i 和 position_i，与其他位置无关吗？
>
> 没错，在**单层注意力**中 K_i = W_K · x_i 确实如此。但 x_i 本身（第 i 个位置的隐状态）经过多层 Transformer 之后，已经包含了前面所有 token 通过注意力传递过来的信息。所以在第 L 层，`K_i^L` 实际上是依赖整个前缀的——只有第 1 层的输入（embedding）是独立的。

因此，**如果两个请求从某个 Block 开始之前的内容不同，从那个 Block 起的所有 K/V 都必须重新计算，不能复用。**

独立 Block hash 无法表达这种前缀依赖：即使 block1 的 token 一样，我们也不知道这个缓存条目对应的是哪种 block0。

### 第三步：链式 hash 的设计

将 token 序列按固定大小（`block_size=16`）切块，每块的 hash 把**前一块的 hash 值**纳入计算：

```
Block 0: h0 = xxh64( b"\x00" * 8  ||  tokens[0:16] )
Block 1: h1 = xxh64( str(h0).encode()  ||  tokens[16:32] )
Block 2: h2 = xxh64( str(h1).encode()  ||  tokens[32:48] )
...
```

图示：

```
tokens: [t0 t1 ... t15 | t16 ... t31 | t32 ... t47]
              Block 0         Block 1        Block 2
                ↓                ↓               ↓
h0 = xxh64(0 ‖ B0)   h1 = xxh64(h0 ‖ B1)   h2 = xxh64(h1 ‖ B2)
```

这样保证：

- **不同前缀 → 不同 hash**：如果 Block 0 的内容不同，h0 不同，进而 h1、h2 全部不同
- **相同前缀 → 相同 hash**：只要前缀 token 序列一致，链式计算结果必然一致
- **误命中概率极低**：xxhash64 的输出空间为 2^64，实际中碰撞概率可忽略不计

### engine.py 中的实现

```python
def _compute_hash(self, tokens: List[int], up_to: int) -> int:
    h = 0
    for start in range(0, up_to, self.block_size):
        end = min(start + self.block_size, up_to)
        if end - start < self.block_size:
            break          # 只对完整 Block 计算，不足一块的尾部忽略
        hh = xxhash.xxh64()
        hh.update(str(h).encode())   # 纳入前一块的 hash
        hh.update(bytes(tokens[start:end]))
        h = hh.intdigest()
    return h
```

注意：只对**完整 Block 边界**计算 hash。不足一个 block_size 的尾部 token 不参与缓存键，这是有意为之——避免频繁更新缓存条目（decode 阶段每步都会延伸序列）。

## 缓存查找与存储流程

```
收到新请求 prompt = [t0, t1, ..., t_{n-1}]
                    ↓
从最长边界向短边界逐级查找缓存
   查 hash(tokens[0:48])  → 未命中
   查 hash(tokens[0:32])  → 未命中
   查 hash(tokens[0:16])  → 命中！cached_len=16, cached_kv=KV[0:16]
                    ↓
Prefill 只需计算 tokens[16:n]（跳过 tokens[0:16]）
模型调用：model(tokens[16:], past_key_values=cached_kv)
                    ↓
Prefill 完成后，将新的 KV 按 Block 边界存入缓存
   _prefix_kv_cache[hash(tokens[0:16])] = past_kv（已存在，跳过）
   _prefix_kv_cache[hash(tokens[0:32])] = past_kv（新增）
   _prefix_kv_cache[hash(tokens[0:48])] = past_kv（新增，若满足边界）
                    ↓
进入 Decode 阶段，逐步生成新 token
```

### 命中 vs 未命中

| 情况 | 发生时机 | 代价 |
|------|----------|------|
| **完全未命中** | 第一个请求（缓存为空） | 完整 Prefill，同时填充缓存 |
| **部分命中** | 前缀匹配了部分 Block | Prefill 仅处理未命中的尾部 |
| **完全命中** | 同一系统提示词的后续请求 | Prefill 几乎为零（只处理 prompt 末尾不足一块的部分） |

## 运行示例

```bash
python run.py
```

`run.py` 模拟的场景：100 token 的系统提示词 + 10 个不同用户问题（每个 10~28 token）。

输出示例（具体数字取决于运行环境）：

```
============================================================
Prefix Caching：系统提示词场景
============================================================
系统提示词: 100 tokens（所有请求共享）
用户问题平均: 19 tokens

  无前缀缓存: prefill 总计 1190 tokens，耗时 Xms
  有前缀缓存: prefill 总计 290 tokens，耗时 Xms
  节省计算: 75% ✅

  缓存命中次数: 9/10
```

其中：
- 第 1 个请求：缓存未命中，完整 Prefill 并填充缓存
- 第 2~10 个请求：系统提示词部分的 K/V 命中缓存，只 Prefill 各自的用户问题部分

## 实际场景的权衡

前缀缓存并非没有代价：

**内存占用**：缓存的 K/V 需要常驻显存（或内存）。缓存条目越多，占用越大。生产系统（如 vLLM）会用 LRU 或引用计数（`ref_count`）管理缓存生命周期——`ref_count > 0` 时 Block 被锁定不可驱逐，降为 0 时才可以被 LRU 淘汰。本步实现为简化版，不驱逐任何条目。

**粒度选择**：`block_size` 越大，缓存条目越少（内存友好），但命中率也可能下降（尾部浪费更多）；`block_size` 越小，命中更细粒度，但管理开销增加。

**适用前提**：前缀必须在**相同位置**出现。如果不同请求的系统提示词内容相同但被拼接在不同偏移量处，K/V 就不再相同，缓存失效。

**不适用场景**：每个请求的 prompt 完全不同（如搜索引擎场景），前缀缓存命中率接近零，反而引入额外的 hash 计算和查找开销。

## 本步实现 vs 生产系统

| 特性 | 本步实现 | 生产系统（如 vLLM） |
|------|----------|---------------------|
| 缓存粒度 | 整段 past_key_values | 物理 Block（固定大小的显存块） |
| 驱逐策略 | 无（永不驱逐） | LRU + ref_count |
| 跨请求共享 | 同进程内字典 | 统一的 Block 池，多请求共享物理 Block |
| 哈希算法 | 链式 xxhash64 | 链式 xxhash（相同思路） |
| Attention 计算 | 直接使用 seq.token_ids | 通过 block_table 索引物理 KV Block |

本步省略了 Block 池和驱逐机制，专注于展示**链式 hash + KV 复用**的核心逻辑。

### engine.py 中 Attention 计算的教学简化

`engine.py` 里的 attention 计算直接使用 `seq.token_ids`，没有真正用到 `block_table`：

```python
# 教学版：attention 使用完整 token_ids，block_table 只参与分配/释放
logits, seq.past_kv = model(seq.token_ids, past_kv=seq.past_kv)
```

这是有意识的简化——block 的分配和释放逻辑（`allocate`、`append_slot`、`free`、hash 命中时的 ref_count 管理）都是真实的，但 attention 计算没有把 block_table 真正接入。

**真实 vLLM 的做法**：attention kernel 直接接收 `block_table`，从分散在物理显存中的非连续 Block 里读取 KV Cache：

```python
# 真实 vLLM：FlashAttention PagedAttention 接口
flash_attn_with_kvcache(
    Q,
    kv_cache,           # 全局 KV Cache 大张量 [total_blocks, block_size, heads, dim]
    block_table=seq.block_table,  # ← 告诉 kernel 去哪些物理 block 读 K/V
    ...
)
```

`block_table` 中记录的物理 Block ID 直接传入 GPU kernel，kernel 内部按 `block_table[block_idx] * block_size + slot_in_block` 计算物理地址，一次 kernel 调用完成对所有非连续 Block 的 attention 计算。

把 block_table 真正接入 attention 需要 FlashAttention 的 PagedAttention 接口，将在 step09 引入。

## 下一步

前缀缓存解决了"重复 prompt 的重复计算"问题，但每个请求仍然是串行处理的。下一步将引入**连续批处理（Continuous Batching）**：如何在请求到达时间不同的情况下，把多个处于不同阶段（有的在 Prefill，有的在 Decode）的请求动态合并到同一个批次里，进一步提升整体吞吐量。
