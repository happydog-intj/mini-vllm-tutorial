# step07 — 前缀缓存 Prefix Caching

## 为什么需要前缀缓存？

在实际部署中，很多请求共享相同的开头部分：

- **系统提示词（System Prompt）**：每个对话都以同一段角色设定开头，可能长达数百个 token
- **Few-shot 示例**：RAG 检索到的文档片段被拼在所有请求的前面
- **模板前缀**：固定格式的 prompt（如"请用中文回答，格式如下：……"）

没有前缀缓存时，每个请求都要对这段共享前缀做一次完整的 Prefill（前向计算），产生重复的矩阵乘法。当系统提示词占 prompt 总长度的大部分时，这些重复计算是纯粹的浪费。

前缀缓存的思路：**把已经算好的前缀 K/V 存下来，下一个请求命中时直接复用，跳过对应的 Prefill 计算。**

## 核心数学：为什么相同前缀的 K/V 完全相同？

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

## 链式 xxhash：为什么不能用普通 hash？

### 普通 hash 的问题

假设用 `hash(tokens[0:n])` 来索引缓存，会遇到两个问题：

**问题1：碰撞风险**
`hash([1, 2, 3])` 和 `hash([7, 8, 9])` 理论上可能碰撞，导致缓存误命中——拿到错误的 K/V。

**问题2：无法区分前缀关系**
`hash([A, B, C])` 和 `hash([X, A, B, C])` 是完全独立的两个 hash 值，无法表达"后者的后缀包含前者"这一关系。

### 链式 hash 的设计

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

本步省略了 Block 池和驱逐机制，专注于展示**链式 hash + KV 复用**的核心逻辑。

## 下一步

前缀缓存解决了"重复 prompt 的重复计算"问题，但每个请求仍然是串行处理的。下一步将引入**连续批处理（Continuous Batching）**：如何在请求到达时间不同的情况下，把多个处于不同阶段（有的在 Prefill，有的在 Decode）的请求动态合并到同一个批次里，进一步提升整体吞吐量。
