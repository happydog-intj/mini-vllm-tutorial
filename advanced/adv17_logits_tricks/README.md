# adv17 — Logits Tricks：模型输出控制工具箱

## 1. 教学目标

- 掌握 5 种 logits 层面的输出控制技术
- 理解每种 trick 的数学原理、使用场景、与 API 的对应关系
- 能区分 logits trick（硬约束）与 prompt engineering（软约束）的适用边界
- 理解 logits tricks 在整个解码流水线中的位置

---

## 2. 背景：为什么需要 Logits Tricks？

Prompt 是**软约束**——你写 "只回答 yes 或 no"，模型可能输出 "Yes, I believe so because..."

Logits trick 是**硬约束**——在物理层面让不合法的 token 概率为零，**不可能**被选中。

```
"请只回答 yes 或 no"          →  模型可能不听话  （软约束）
force_tokens([yes_id, no_id])  →  物理上只能选这两个（硬约束）
```

---

## 3. 在解码流水线中的位置

```
模型 forward pass
       │
       ▼
  原始 logits [vocab_size]
       │
       │  ┌──────────────────────────────┐
       ├──│ ★ Logits Tricks（本章）       │  ← 最先执行
       │  │   Bias / Force / Ban / Prefix │
       │  └──────────────────────────────┘
       │
       │  ┌──────────────────────────────┐
       ├──│ Guided Decode（adv15）        │  ← regex/JSON FSM 动态约束
       │  └──────────────────────────────┘
       │
       │  ┌──────────────────────────────┐
       ├──│ Temperature 缩放             │
       │  └──────────────────────────────┘
       │
       │  ┌──────────────────────────────┐
       ├──│ TopK / TopP / MinP（adv02）  │  ← 采样策略
       │  └──────────────────────────────┘
       │
       ▼
    softmax → sample → output token
```

**关键区别：**
- Logits Tricks: 决定"哪些 token 有资格参与竞争"
- 采样策略 (adv02): 决定"合法候选中如何选择"
- Guided Decode (adv15): 根据语法状态动态调整白名单
- 三者可以组合使用！

---

## 4. 五种 Trick 详解

### 4.1 Logit Bias（偏置注入）

**原理：** 给指定 token 的 logit 加一个常数

```python
logits[token_id] += bias_value
```

**数值直觉：**

```
原始: logits = [2.0, 1.5, 1.0]  → probs = [0.50, 0.33, 0.17]

+5 偏置: logits = [7.0, 1.5, 1.0]  → probs = [0.99, 0.003, 0.002]
+2 偏置: logits = [4.0, 1.5, 1.0]  → probs = [0.84, 0.11, 0.05]
-5 偏置: logits = [-3., 1.5, 1.0]  → probs = [0.004, 0.59, 0.37]
```

**使用场景：**
- OpenAI API 的 `logit_bias` 参数
- 鼓励模型输出 JSON 格式词（给 `{` 加正偏置）
- 轻微抑制某些词（比 ban 更柔和）

**代码：**
```python
def logit_bias(logits, bias_map: dict[int, float]):
    result = logits.clone()
    for tid, bias in bias_map.items():
        result[tid] += bias
    return result
```

---

### 4.2 Force Tokens（强制分类输出）

**原理：** 只保留白名单 token，其余全部 -inf

```python
mask = torch.full_like(logits, float('-inf'))
mask[allowed_ids] = 0
logits = logits + mask
# → softmax 后非白名单概率 = 0
```

**使用场景：**

| 场景 | allowed_ids |
|------|-------------|
| 是非判断 | `["yes", "no"]` |
| 多选题 | `["A", "B", "C", "D"]` |
| 情感分类 | `["positive", "negative", "neutral"]` |
| 评分 | `["1", "2", "3", "4", "5"]` |

**为什么不用 prompt 写"只回答 yes 或 no"？**

| 方式 | 保证率 | 问题 |
|------|--------|------|
| Prompt 约束 | ~85% | 模型可能输出 "Yes, because..." |
| Force Tokens | **100%** | 物理上不可能输出其他内容 |

**代码：**
```python
def force_tokens(logits, allowed_ids: list[int]):
    mask = torch.full_like(logits, float('-inf'))
    mask[allowed_ids] = 0.0
    return logits + mask
```

---

### 4.3 Logprobs 提取（置信度打分）

**原理：** 不修改 logits，读取目标 token 的 log probability 作为分数

```python
log_probs = log_softmax(logits)
score_yes = log_probs[yes_id]   # -0.59 → P=0.55
score_no  = log_probs[no_id]    # -0.79 → P=0.45
# yes 的 logprob 更高 → 答案是 yes
```

**使用场景：**
- **不需要生成文本！** 一次 forward pass 直接得到分类结果
- 比生成完整 "yes" 快很多倍
- 可设置阈值：差距 < 0.1 时拒绝回答（低置信度）

**实际 API 用法（OpenAI）：**
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    max_tokens=1,
    logprobs=True,
    top_logprobs=5,  # 返回 top-5 token 的 logprob
)
# 直接从 logprobs 中读取 yes/no 的分数
```

**代码：**
```python
def extract_logprobs(logits, target_ids: list[int]):
    log_probs = F.log_softmax(logits, dim=-1)
    return {tid: log_probs[tid].item() for tid in target_ids}
```

---

### 4.4 Prefix Forcing（前缀强制）

**原理：** 前 N 步不走 logits 选择，直接注入指定 token

```python
prefix_ids = [token_lbrace, token_quote]  # 强制以 '{"' 开头

for step in range(max_len):
    logits = model.forward(...)
    if step < len(prefix_ids):
        next_token = prefix_ids[step]   # 不看 logits，直接注入
    else:
        next_token = sample(logits)     # 正常解码
```

**为什么需要？**
```
不加 prefix force:
  → "Sure! Here's the JSON you requested:\n{"name": "..."}"

加 prefix force（强制以 '{"' 开头）:
  → '{"name": "..."}'
```

**使用场景：**
- JSON 输出时强制以 `{` 或 `[` 开头
- 代码生成时强制以 `def ` 或 `class ` 开头
- 强制回答以 "Answer: " 前缀开始

**代码：**
```python
def prefix_force(step, prefix_ids, logits):
    if step < len(prefix_ids):
        forced = torch.full_like(logits, float('-inf'))
        forced[prefix_ids[step]] = 0.0
        return forced, True  # is_forced=True
    return logits, False
```

---

### 4.5 Ban Tokens（禁止词）

**原理：** 将黑名单 token 的 logit 设为 -inf

```python
logits[banned_ids] = float('-inf')
```

**常见用法：**

| 禁止什么 | 效果 |
|----------|------|
| EOS token | 强制模型继续生成到指定长度 |
| 换行符 `\n` | 强制单行输出 |
| "I'm sorry" 的首 token | 减少模型拒绝 |
| System prompt 关键词 | 防止泄露指令 |
| 特定语言的 token | 强制单语输出 |

**代码：**
```python
def ban_tokens(logits, banned_ids: list[int]):
    result = logits.clone()
    result[banned_ids] = float('-inf')
    return result
```

---

## 5. 组合使用

多种 trick 可以叠加：

```python
# 场景：情感分类，只允许 positive/negative，
# 同时偏置 positive（因为大多数评论是正面的先验）
logits = force_tokens(logits, [pos_id, neg_id])   # 只允许两个选项
logits = logit_bias(logits, {pos_id: 1.0})        # 轻微偏向 positive
probs = softmax(logits)
```

```python
# 场景：JSON 生成，禁止废话 token，前缀强制 {"
logits = ban_tokens(logits, [sorry_id, sure_id, newline_id])
logits, forced = prefix_force(step, [lbrace_id, quote_id], logits)
if not forced:
    logits = guided_decode(logits, json_fsm_state)  # adv15 的 FSM 约束
```

---

## 6. Logit Tricks vs Prompt Engineering

| 维度 | Prompt Engineering | Logits Tricks |
|------|-------------------|---------------|
| 保证率 | ~85-95% | **100%** |
| 灵活性 | 极高（自然语言描述） | 有限（需要知道 token id） |
| 可解释性 | 高 | 中 |
| 性能开销 | 无额外开销 | 极小（向量操作） |
| 适用场景 | 复杂指令、多轮对话 | 格式强制、分类、安全约束 |

**最佳实践：** 两者结合使用
- Prompt 告诉模型**意图**（"请判断这是否正面评论"）
- Logits trick 保证**格式**（只能输出 yes/no）

---

## 7. 运行

```bash
cd advanced/adv17_logits_tricks
python run.py
```

期望输出：

```
============================================================
Logits Tricks 工具箱 — 5 个实用技巧演示
============================================================

原始 logits: {'yes': 0.337, 'no': 0.129, ...}
原始 probs:  {'yes': '0.225', 'no': '0.183', ...}

------------------------------------------------------------
Trick 1: Logit Bias — 给 'yes' 加 +5 偏置
  → 'yes' 概率从 0.225 提升到 0.977

Trick 2: Force Tokens — 只允许 'yes'(0) 和 'no'(1)
  → 模型选择: 'yes' (只能在 yes/no 中选)

Trick 3: Logprobs 提取 — 比较 'yes' vs 'no' 的置信度
  → 模型更倾向 'yes'，差距 = 0.2079

Trick 4: Prefix Forcing — 强制前两步输出 '{' + '"'
  → 前 2 步强制输出 '{"'，之后模型自由选择

Trick 5: Ban Tokens — 禁止 'maybe'(2) 和 'hello'(3)
  → 'maybe' 和 'hello' 概率归零
...
✅ 所有 Logits Tricks 演示通过
```

---

## 8. 下一步

这些 logits trick 是构建更复杂系统的基础模块：

- **adv15 Guided Decode** = 每步动态计算 force_tokens（基于 FSM 状态）
- **adv16 Function Call** = prefix_force + guided_decode（强制输出函数调用 JSON）
- **RLHF/DPO 训练** = logprobs 提取是奖励模型的核心接口
- **Speculative Decoding (adv03)** = 用 logprobs 验证草稿模型的输出
