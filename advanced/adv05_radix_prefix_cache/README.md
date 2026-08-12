# adv05 — Radix Attention + Copy-on-Write 前缀缓存

## 1. 教学目标

本章用一棵**基数树（Radix Tree / Compressed Trie）**替代 step13 的链式 hash 方案，实现跨请求的前缀 KV 共享与 Copy-on-Write (CoW) 分叉。

学完本章你将能够：

- 理解基数树如何在 O(前缀长度) 时间内完成最长前缀匹配，避免 step13 逐 Block hash 重算的线性扫描
- 解释 CoW 分叉语义：多请求共享前缀节点（ref=2），后续 token 不同时 split 产生独立后缀子节点
- 阅读 SGLang RadixAttention 论文/代码时，能对应本章的 `RadixNode` / `insert` / `match_prefix` / `_split`

---

## 2. 问题

主系列 step13 用**链式 xxhash + 字典**做前缀缓存：

```
h0 = xxh64(b"\x00"*8 || tokens[0:16])
h1 = xxh64(str(h0)   || tokens[16:32])
h2 = xxh64(str(h1)   || tokens[32:48])
...
prefix_cache[h2] = past_kv
```

这个方案有两个效率缺陷：

**缺陷一：长前缀查找需要线性扫描**
查询最长命中前缀时，必须从最长可能长度逐级向短探测（O(块数)次 hash 计算 + 字典查找）。块数随前缀增长线性增加。

**缺陷二：无法表达"树状"共享结构**
多个请求可能共享公共前缀，然后分叉出不同后缀。hash 字典是扁平的，无法直观表达这种分叉关系，也无法细粒度地追踪哪些节点被几个请求共享。

**基数树的解法**：把所有已缓存的 token 序列组织成一棵压缩前缀树，insert/match 均沿树路径单程遍历，无需重算 hash，分叉点自然就是 CoW 触发点。

---

## 3. 原理

### 基数树（Radix Tree）结构

基数树是压缩的前缀树（Trie）：把只有单个子节点的链状路径合并为一个边，减少节点数量。

```
插入序列 A = ["sys", "prompt", "A"]
插入序列 B = ["sys", "prompt", "B"]

树结构:
  root
  └─── ["sys","prompt"]  (ref=2, value=None)   ← 共享前缀节点
         ├─── ["A"]  (ref=1, value=KV_A)        ← 请求 A 的后缀
         └─── ["B"]  (ref=1, value=KV_B)        ← 请求 B 的后缀

节点含义:
  key   — 该边对应的 token 子序列（压缩后可含多个 token）
  value — 到此节点为止的完整前缀 KV 引用（内部节点为 None）
  ref   — 有多少请求路径经过或终止于此节点
```

### CoW 分叉示意

```
初始（只有请求 A）:

  root
  └─── ["sys","prompt","A"]  ref=1, value=KV_A


插入请求 B = ["sys","prompt","B"]，在 idx=2 处分叉:

  step 1: 遍历发现 child.key=["sys","prompt","A"]，
          在位置 j=2 与 tokens[2]="B" 不同

  step 2: _split(child, idx=2):
          ┌──────────────────────────────────────────┐
          │ Before:  child.key = ["sys","prompt","A"] │
          │ After:   child.key = ["sys","prompt"]     │  ← 公共前缀留在原节点
          │          child.value = None               │  ← 内部节点不持有 KV
          │          suffix_child.key = ["A"]         │  ← 原后缀成为子节点
          │          suffix_child.value = KV_A        │
          └──────────────────────────────────────────┘

  step 3: 继续为 "B" 创建新叶节点

  root
  └─── ["sys","prompt"]  ref=2, value=None
         ├─── ["A"]  ref=1, value=KV_A
         └─── ["B"]  ref=1, value=KV_B
```

`ref=2` 表示两个请求共享该前缀节点 —— 物理 KV 只存一份，零拷贝。

### ❓ Q1：Radix Tree 的"压缩"到底压缩了什么？

**问题**：普通 Trie 每个边对应一个 token，Radix Tree 说"压缩"，压缩了什么？

**答案**：压缩的是**只有单个子节点的链状路径**：

```
普通 Trie（未压缩）:
  root → "s" → "sy" → "sys" → "sys,p" → "sys,pr" → "sys,pro" → "sys,prompt"
  7 个节点，但路径上没有分叉 —— 纯粹浪费

Radix Tree（压缩后）:
  root → ["sys","prompt"]
  只有 1 个节点，把整条链合并成一条边

**只有出现分叉时才创建新节点**。这就是"基数树"（Radix/Compressed Trie）的核心优化：
节点数 = 分叉点数，而不是 token 数。当共享前缀很长时，节点数大幅减少。
```

### ❓ Q2：ref 计数什么时候减少？教学版说"未实现"，那会内存泄漏吗？

**问题**：insert 时 `ref += 1`，但请求结束时没做 `ref -= 1`。教学版会泄漏吗？

**答案**：是的，教学版**没有实现引用递减和节点驱逐**，树会无限增长。真实框架的做法：

```python
# 请求结束时:
def release_request(node_path):
    for node in node_path:
        node.ref -= 1
        if node.ref == 0 and all(child.ref == 0 for child in node.children):
            evict(node)  # LRU 驱逐：从 GPU/CPU 显存中移除 KV
```

当 `ref == 0` 时，说明没有任何请求经过该节点，可以安全驱逐。SGLang 用 LRU 策略：显存满了就从 `ref == 0` 的节点中选最久未访问的驱逐。

### ❓ Q3：`_split` 为什么必须先保存 `split_tok` 再截断？

**问题**：代码注释说"① 先保存分叉键（必须在截断前）"，不保存会怎样？

**答案**：因为 `node.key[idx]` 依赖于**截断前**的 `node.key`：

```python
# 正确顺序:
split_tok = node.key[idx]       # ① 此时 node.key = ["sys","prompt","A"], idx=2 → split_tok = "A"
node.key = node.key[:idx]       # ② 现在 node.key = ["sys","prompt"]
node.children = {split_tok: suffix_child}  # ③ {"A": suffix_child} ✓

# 错误顺序（先截断）:
node.key = node.key[:idx]       # ① node.key 变成 ["sys","prompt"]
split_tok = node.key[idx]       # ② IndexError! idx=2 超出范围（现在只有 2 个元素）
```

这是一个经典的**先读后写**顺序依赖 bug。

---

## 4. 实现细节

### RadixNode

```python
class RadixNode:
    key:      List       # 该节点代表的 token 子序列（压缩后可多 token）
    value:    Any        # 叶节点/完整前缀对应的 KV；内部节点为 None
    children: Dict       # first_token -> RadixNode
    ref:      int        # 引用计数
```

### insert

```
insert(tokens, value):
  node = root, i = 0
  while i < len(tokens):
    tok = tokens[i]
    if tok not in node.children:
      创建新叶节点 leaf(tokens[i:], value), ref=1
      return
    child = node.children[tok]
    j = 公共前缀长度(child.key, tokens[i:])
    if j < len(child.key):          ← 分叉点 → CoW 触发
      _split(child, j)              ← 将 child 在 j 处分裂
    child.ref += 1                  ← 当前路径经过 child
    i += j
    if i >= len(tokens): break      ← tokens 恰好在 child 处结束
    node = child
  node.value = value
```

### match_prefix

```
match_prefix(tokens) -> (hit_len, hit_val):
  node = root, i = 0
  hit_len = 0, hit_val = None
  while i < len(tokens) and tokens[i] in node.children:
    child = node.children[tokens[i]]
    j = 公共前缀长度(child.key, tokens[i:])
    if j < len(child.key): break    ← tokens 在节点内部中断，停止
    i += j
    if child.value is not None:
      hit_len, hit_val = i, child.value   ← 更新命中点
    node = child
  return hit_len, hit_val
```

### _split（CoW 核心，修复 plan 中的 bug）

```python
def _split(self, node, idx):
    split_tok = node.key[idx]              # ① 先保存分叉键（必须在截断前）
    suffix_child = RadixNode(node.key[idx:], node.value)
    suffix_child.children = node.children
    suffix_child.ref = node.ref            # 已有引用路径终止于此
    node.key   = node.key[:idx]            # ② 截断原节点为公共前缀
    node.value = None
    node.children = {split_tok: suffix_child}  # ③ 唯一子节点（只赋值一次）
```

### ref 引用计数

| 操作 | ref 变化 |
|------|---------|
| `insert` 遍历到 child | `child.ref += 1` |
| `insert` 创建新叶节点 | `leaf.ref = 1` |
| `_split` 创建 suffix_child | `suffix_child.ref = node.ref`（继承）|
| 请求结束（教学版未实现） | `ref -= 1`，归零可驱逐 |

---

## 5. 教学版 vs 真实框架

### 与 SGLang RadixAttention 对比

SGLang（2024）的 RadixAttention 是本章思路的生产实现：

| 特性 | 本章（教学版） | SGLang RadixAttention |
|------|--------------|----------------------|
| 数据结构 | `RadixTree`（Python dict 树） | 相同的 Radix Tree（C++/Python） |
| KV 存储 | Python dict 模拟 | GPU 显存 Page Pool |
| 并发请求 | 串行 | 批处理，多请求同时共享树 |
| 驱逐策略 | 无（永不驱逐） | LRU + ref_count |
| 跨会话复用 | 同进程内 | 跨会话持久化（多轮对话） |
| prefix match 粒度 | 任意 token | 对齐到 Block 粒度 |

### step13 hash 方案 vs adv05 radix 方案

| 维度 | step13 链式 hash | adv05 RadixTree |
|------|-----------------|-----------------|
| 数据结构 | 扁平 dict（hash → KV） | 树形（节点 → KV） |
| 最长前缀查找 | O(块数)次 hash 重算 | O(前缀长度) 单程树遍历 |
| 分叉可见性 | 无（hash 独立） | 明确（`_split` 产生子节点） |
| ref 计数 | 手动 ref_count 字段 | 节点 `ref` 属性，insert 自动维护 |
| CoW 实现 | 需额外机制 | 树分裂自然实现 |
| 代码复杂度 | 低 | 稍高（split 逻辑） |

**何时用 hash，何时用 radix tree？**
- 请求少、前缀简单（系统提示词单一固定）→ hash 够用
- 多样化前缀（RAG 多文档、多轮对话、多系统提示词）→ radix tree 命中率更高、结构更清晰

---

## 6. 运行

```bash
cd advanced/adv05_radix_prefix_cache
python run.py
```

预期输出：

```
============================================================
adv05_radix_prefix_cache — Radix Attention + CoW 验证
============================================================

请求 A ['sys', 'prompt', 'A']:
  命中长度=0  miss 长度=3
  KV={0: 'sys', 1: 'prompt', 2: 'A'}

请求 B ['sys', 'prompt', 'B']:
  命中长度=0  miss 长度=3
  KV={0: 'sys', 1: 'prompt', 2: 'B'}

[断言①] match_prefix(['sys','prompt','A']) → hit_len=3
  ✓ 命中长度正确 (3)

[断言②] 共享前缀节点 ref=2
  ✓ 共享前缀节点 ref=2（两请求均经过该节点）

[断言③] 分叉子节点 keys={'B', 'A'}
  ✓ 后缀节点 'A' 独立: key=['A'], KV末token='A'
  ✓ 后缀节点 'B' 独立: key=['B'], KV末token='B'

[附加]  match_prefix(['sys','prompt','B']) → hit_len=3 ✓

树结构概览:
  root
  └─ key=['sys', 'prompt']  ref=2  value=None
     └─ key=['A']  ref=1  value=<kv>
     └─ key=['B']  ref=1  value=<kv>

✅ adv05_radix_prefix_cache 通过
```

---

## 7. 下一步

adv06 — **Pipeline Parallel（流水线并行）**

基数树解决的是单机显存的前缀复用问题。当模型规模超过单卡容量时，需要把模型的不同层切分到多张 GPU 上，形成流水线：卡 0 计算第 1~N 层，卡 1 计算第 N+1~2N 层……前向传播在设备间流动。adv06 将展示：

- Pipeline Stage 划分与 micro-batch 调度
- 如何消除 bubble（流水线气泡）
- 与 Tensor Parallel（张量并行）的区别
