# step14_4 — torch.tril causal mask：消除逐行 Python 循环

## 问题

`Paged Prefix Cache` 每次 forward 都用 Python 循环逐行构造 causal mask：

```python
mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device)
for i in range(seq_len):
    mask[i, :start_pos + i + 1] = False
scores = scores.masked_fill(mask, float("-inf"))
```

**问题：**
1. `seq_len` 次 Python 循环，每次一次 tensor slice 赋值
2. 每次 forward 都重新 allocate 一个 `[seq_len, total_len]` 的 bool tensor
3. decode 阶段 `seq_len=1`，整个 mask 全是 `False`，构造和 masked_fill 都在浪费时间

## 解决方案：broadcast 比较，一次生成

causal mask 的本质是：位置 `(i, j)` 被 mask 当且仅当 `j > start_pos + i`。
用 broadcast 一次生成，无需 Python 循环：

```python
q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)    # [seq_len, 1]
k_idx = torch.arange(total_len, device=x.device).unsqueeze(0)  # [1, total_len]
causal_mask = k_idx > (start_pos + q_idx)                       # [seq_len, total_len]
scores = scores.masked_fill(causal_mask, float("-inf"))
```

**decode 阶段的特殊优化：**

decode 时 `seq_len=1`，当前 token 可以 attend 到所有历史，mask 全为 `False`，直接跳过：

```python
if seq_len == 1:
    # decode：无需 mask
    weights = torch.softmax(scores, dim=-1)
else:
    # prefill：构造 causal mask
    q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
    k_idx = torch.arange(total_len, device=x.device).unsqueeze(0)
    causal_mask = k_idx > (start_pos + q_idx)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
```

## torch.tril 介绍

`torch.tril(input, diagonal=0)` 取矩阵的**下三角部分**（含对角线），上三角置零：

```python
x = torch.ones(4, 4)
torch.tril(x)
# tensor([[1, 0, 0, 0],
#         [1, 1, 0, 0],
#         [1, 1, 1, 0],
#         [1, 1, 1, 1]])
```

下三角矩阵天然对应因果掩码的语义：位置 `i` 只能看到位置 `0..i`（即第 `i` 行的前 `i+1` 列）。

## 两种等价实现

### 方式 1：torch.tril

```python
full  = torch.ones(total_len, total_len, dtype=torch.bool, device=x.device)
allow = torch.tril(full, diagonal=0)          # [total_len, total_len]，下三角为 True
allow = allow[start_pos:start_pos + seq_len]  # [seq_len, total_len]，只取 query 对应的行
causal_mask = ~allow                          # True=需要屏蔽
```

可视化（total_len=4，start_pos=0，seq_len=4）：

```
allow（可以 attend 的位置）:      causal_mask（需要屏蔽）:
T F F F                           F T T T
T T F F                           F F T T
T T T F                           F F F T
T T T T                           F F F F
```

**局限**：先分配 `[total_len, total_len]` 的完整矩阵再切片，decode 时 total_len 很大但只用 1 行，有浪费。

### 方式 2：broadcast 比较（本章采用）

直接生成 `[seq_len, total_len]`，无多余分配：

```python
q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)    # [seq_len, 1]
k_idx = torch.arange(total_len, device=x.device).unsqueeze(0)  # [1, total_len]
causal_mask = k_idx > (start_pos + q_idx)                       # [seq_len, total_len]
```

位置 `(i, j)` 的值为 `j > start_pos + i`：key 在 query 的未来则为 True（屏蔽）。

**优势**：直接生成目标大小，decode（seq_len=1）时可完全跳过。

### 对比

| | torch.tril | broadcast 比较 |
|---|---|---|
| 中间分配大小 | `[total_len, total_len]` | `[seq_len, total_len]`（直接目标大小）|
| decode（seq_len=1）| 仍需分配和切片 | 直接跳过整个操作 |
| 代码可读性 | 直观（下三角=因果）| 需理解 broadcast |
| 适用场景 | prefill | prefill + decode 均适用 |

## Python 循环次数对比

| | Paged Prefix Cache | step14_4 |
|---|---|---|
| mask 构造 | seq_len 次 Python 循环 | 0 次（纯 tensor broadcast）|
| tensor 分配 | 每次 forward 1 次 | 1 次（可进一步缓存复用）|
| decode 特殊处理 | 无（seq_len=1 也走循环）| 跳过整个 mask 操作 |

## 与 vLLM 的对比

vLLM 使用 FlashAttention，causal mask 在 kernel 内部隐式处理，完全不需要显式构造 mask tensor，也不产生任何额外显存分配。`is_causal=True` 参数告知 kernel 使用下三角掩码即可。

本章是在保持标准 softmax attention 的前提下，消除 Python 循环的最低成本方案，也是通向 `F.scaled_dot_product_attention(is_causal=True)` 的过渡。

## 实现

见 `model.py` — `PagedMultiHeadAttention.forward` 中 causal mask 构造部分。

## 运行

```bash
python run.py
```
