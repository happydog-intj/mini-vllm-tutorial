# adv15 — Guided Decoder：JSON/Regex 结构化输出

## 1. 教学目标

- 理解"引导解码"(Guided Decoding)的核心思想：在采样阶段用正则约束裁剪候选 token
- 掌握 Regex Partial Match 的兼容性处理策略（Python 版本差异 / 第三方库回退）
- 能将本教学版扩展到 JSON Schema、CFG 等更复杂约束场景

---

## 2. 问题

LLM 自由生成时常出现格式错误：

```
# 期望: {"price": 3.14}
# 实际输出:
{"price": "3.14 dollars (approx)"}   # 多余文字
{"price": 3.1.4}                     # 非法数字
{price: 3.14}                        # 缺少引号
```

**后处理脆弱**：写正则修复 → 漏洞多；让模型重试 → 浪费算力。

根本原因：模型在每步自由选 token，没有语法约束。

---

## 3. 原理

每一个解码步骤，在 softmax/argmax 之前插入一个掩码层：

```
                  模型输出 logits
                       |
                 RegexGuide.next_allowed()
                  |         |
            允许的 token   不允许的 token
            logit 不变     logit -> -inf
                  |
             softmax / argmax / sample
                  |
             确定性合法 token
```

**核心：Regex Partial Match**

```
已生成:  "3."
候选:    "1"  -> trial="3.1" -> partial match r'-?\d+(\.\d+)?' -> True  (允许)
候选:    "-"  -> trial="3.-" -> partial match                  -> False (屏蔽)
候选:    "a"  -> trial="3.a" -> partial match                  -> False (屏蔽)
```

只要候选拼接后"仍可能扩展为完整合法串"，就允许它；否则屏蔽。

---

## 4. 实现细节

### `mask_logits(logits, allowed_token_ids)`

```python
mask = torch.full_like(logits, float('-inf'))
mask[allowed_token_ids] = 0
return logits + mask
```

把不在 `allowed_token_ids` 集合的位置加 `-inf`，经 softmax 后概率为 0。

### `RegexGuide.next_allowed(logits)`

```
for each (tid, tok) in vocab:
    trial = self.generated + tok
    if partial_match(pattern, trial):
        allowed.append(tid)
return mask_logits(logits, tensor(allowed))
```

时间复杂度 O(|vocab|)，单字符词表下性能可接受；真实词表需索引加速（见第 5 节）。

### `RegexGuide.consume(token_str)`

每步采样后调用，将选中 token 追加到 `self.generated`，推进内部状态。

### `RegexGuide.is_complete()`

```python
return bool(re.fullmatch(self.pattern, self.generated))
```

判断当前已生成串是否已经完整匹配 regex，用于决定何时停止解码。

---

### 兼容性：Partial Match 三级策略

| 优先级 | 实现 | 条件 |
|--------|------|------|
| 1 | `re.match(pattern, trial, re.PARTIAL_MATCH)` | Python 3.11+ 且标准库已合并该特性 |
| 2 | `regex.match(pattern, trial, partial=True)` | `pip install regex` 已安装 |
| 3 | **教学回退**：试探拼接后能否 fullmatch | 无任何依赖；对数字/JSON 字段场景足够准确 |

> **说明**：`re.PARTIAL_MATCH` 是社区提案（CPython issue #64381），Python 3.13 标准库
> 仍未合并。本模块在运行时自动检测并选择最优策略，通过 `PARTIAL_MATCH_STRATEGY`
> 变量暴露当前使用的实现名称（便于调试）。

### ❓ Q1：partial match 为什么要"试探拼接后 fullmatch"？

**问题**：为什么不直接检查当前生成的串是否部分匹配，而要拼接每个候选 token 再 fullmatch？

**答案**：因为 `re` 标准库没有原生的 partial match API。教学版用暴力试探法：对每个候选 token，拼接后检查是否"仍有可能"扩展为合法串。三级回退策略就是因为 Python 标准库支持不够好。

### ❓ Q2：O(vocab) 的逐 token 检查在真实词表（10万+）下会不会很慢？

**答案**：**会非常慢！** 生产版用 **FSM（有限状态机）**：

```
教学版: 每步 50000 次正则匹配 → 很慢
生产版: 编译 regex → DFA → 每步 O(1) 查表
```

Outlines 库就是这样做的——把 regex/JSON Schema/CFG 都编译成 FSM。

### ❓ Q3：mask_logits 用 -inf，和直接删掉候选 token 有什么区别？

**答案**：**-inf mask 保持词表形状和对齐**：

```python
# -inf mask: logits = [2.0, -inf, 1.0] → softmax → [0.73, 0.0, 0.27] ✓
# 直接删除: logits = [2.0, 1.0] → 但 token id 映射断了！
```

此外，-inf mask 对 GPU 友好——可在 CUDA kernel 里并行做。

---

## 5. 教学版 vs 真实框架

| 特性 | 本教学版 | vLLM / Outlines / lm-format-enforcer |
|------|---------|--------------------------------------|
| 约束表达 | Python regex | JSON Schema / CFG / EBNF |
| 核心算法 | 逐 token 前缀检查 O(V) | **有限状态机(FSM)** + token 索引，O(1) per step |
| 加速 | 无 | 预计算 token -> 状态转移表；batch GPU mask |
| 支持场景 | 数字、简单模式 | JSON、SQL、代码、任意 CFG |
| 多步状态 | `self.generated` 字符串 | FSM 当前状态节点 |

**真实实现核心差异**：将 regex 编译为 DFA/NFA，每个状态预计算可接受的 token 集合
（索引），解码时 O(1) 查表而非 O(vocab_size) 逐字检查。

---

## 5.5 深入：模型具体怎么输出 JSON？

很多人好奇：模型明明是一个概率分布采样器，它怎么"知道"要输出合法 JSON？

**答案：它不知道。是我们在每一步强制它只能选合法 token。**

### 完整流程：生成 `{"val": 42}`

把 JSON 语法看成一个**有限状态机 (FSM)**，当前状态决定下一步允许什么字符：

```
┌─────────┐    {     ┌──────────┐    "     ┌──────────┐
│  START   │────────▶│  OBJ_KEY │────────▶│  IN_STR  │
└─────────┘         └──────────┘         └──────────┘
                                               │ "
                                               ▼
┌─────────┐    }     ┌──────────┐    :     ┌──────────┐
│   END   │◀────────│ OBJ_VAL  │◀────────│  COLON   │
└─────────┘         └──────────┘         └──────────┘
```

**逐 token 的决策过程：**

| Step | 已生成 | 当前状态 | 允许的 token | 选中 | 原因 |
|------|--------|----------|-------------|------|------|
| 1 | `""` | START | `{` | `{` | JSON 对象必须以 `{` 开头 |
| 2 | `{` | OBJ_KEY | `"` | `"` | key 必须是字符串 |
| 3 | `{"` | IN_STR | `a-z, A-Z, 0-9, "` | `v` | 字符串内容或结束引号 |
| 4 | `{"v` | IN_STR | `a-z, A-Z, 0-9, "` | `a` | 继续字符串 |
| 5 | `{"va` | IN_STR | `a-z, A-Z, 0-9, "` | `l` | 继续字符串 |
| 6 | `{"val` | IN_STR | `"` | `"` | 结束 key（模型概率最高） |
| 7 | `{"val"` | COLON | `:` | `:` | key 后**只能**是冒号 |
| 8 | `{"val":` | OBJ_VAL | `0-9, ", {, [, t, f, n` | `4` | 值开始，数字/字符串/对象/数组/布尔/null |
| 9 | `{"val":4` | IN_NUM | `0-9, }, ,` | `2` | 继续数字或结束 |
| 10 | `{"val":42` | IN_NUM | `}, ,` | `}` | 结束对象 |

**关键洞察：** Step 7 只有一个合法选择 `:`，模型别无选择。Step 8 有多个合法选择，这时模型的概率分布在合法候选中自由竞争。

### 数学：mask 如何工作

每一步，词表中所有 token 都有一个 logit 值（模型的原始打分）。Guided decoder 的工作：

```python
# 假设词表 = ["{", "}", '"', "a", "5", ":", " ", "\n"]
# 模型输出 logits = [2.1, 0.3, 1.8, -0.5, 0.9, -1.2, 0.4, -0.8]

# 当前状态 START → 只允许 "{"
allowed = [True, False, False, False, False, False, False, False]

# mask 操作
masked_logits = [2.1, -inf, -inf, -inf, -inf, -inf, -inf, -inf]

# softmax 后
probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# → 必然选中 "{"
```

当允许多个 token 时（如 Step 8），mask 保留多个候选，模型在它们之间按原始概率分布选择：

```python
# 状态 OBJ_VAL → 允许值的起始字符
# logits = [0.1, -0.5, 2.3, 0.8, 1.9, -1.2, 0.4, -0.8]
# allowed: '"', 数字(0-9), '{', '[', 't', 'f', 'n'

# 假设简化为: allowed = [False, False, True, False, True, False, False, False]
masked_logits = [-inf, -inf, 2.3, -inf, 1.9, -inf, -inf, -inf]

# softmax → P('"')=0.60, P("5")=0.40
# 模型倾向输出字符串，但数字也有机会
```

### 为什么不后处理修 JSON？

| 方案 | 问题 |
|------|------|
| 生成后正则修复 | 结构性错误无法修，如嵌套括号不匹配 |
| 让模型重试 | 浪费 2-3x 算力，且不保证成功 |
| **引导解码** | **一次生成，0% 失败率，零额外算力** |

---

## 5.6 动画演示

提供 3Blue1Brown 风格的 Manim 动画，可视化整个 JSON 引导解码过程：

```bash
# 安装 manim (如果未安装)
pip install manim

# 低质量预览（开发时用）
cd advanced/adv15_guided_decoder
manim -pql json_guided_animation.py JSONGuidedScene

# 高质量渲染
manim -pqh json_guided_animation.py JSONGuidedScene
```

动画包含 4 幕：
1. **自由生成 vs 引导生成** — 对比错误率
2. **JSON 有限状态机** — 状态转移的可视化
3. **逐 token 掩码流程** — 实际生成 `{"val":42}` 的每一步
4. **Logits Masking 数学** — 柱状图展示 -inf mask 和 softmax 效果

---

## 6. 运行

```bash
# 安装依赖（可选，提供更好的 partial match 支持）
pip install regex torch

# 运行演示
cd advanced/adv15_guided_decoder
python run.py
```

期望输出（数字因随机种子固定）：

```
=======================================================
adv15_guided_decoder — Regex Guided Decoding Demo
=======================================================
Pattern        : -?\d+(\.\d+)?
Vocab size     : 13 (chars: ['-', '0', ..., '.'] + EOS)
Partial match  : _try_partial_match_regex_lib

[Case 1] target hint='3.14'
  step  1: token='1'    generated='1'
  => fullmatch OK
...
✅ adv15_guided_decoder 通过
```

---

## 7. 下一步

**adv16 — Function Call（工具调用解码）**

Guided Decoding 可直接应用于 Function Call 场景：
- 约束模型输出严格合法的 JSON（函数名 + 参数）
- 结合 JSON Schema 生成参数值时逐字段约束类型
- adv16 将展示如何在 mini-vLLM 中端到端集成工具调用流程

**adv17 — Logits Tricks 工具箱**

引导解码是 logits 操控的一种形式，更多轻量技巧见 adv17：
- Logit Bias（偏置注入）、Force Tokens（强制 yes/no）、Ban Tokens（禁止词）
- Logprobs 提取（置信度打分）、Prefix Forcing（前缀强制）
