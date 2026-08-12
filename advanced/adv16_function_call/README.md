# Function Call / Tool Call：让模型会"调工具" — Function Call

## 教学目标

理解 Function Call / Tool Call 的本质：模型本身只会"生成文本"，要让它"采取行动"（查天气、算数、查数据库），就要让模型输出**结构化的工具调用**，执行后把结果回填到上下文，继续生成最终答案。本步实现一个 ReAct 风格的工具调用循环。

## 问题：模型只会"说"，不会"做"

主系列 step20 实现了 OpenAI 兼容的 HTTP 服务，但只做了 `messages` + `stream` 基础功能，README 里明确标注"不支持 function calling、tools"。

为什么不直接让模型"自由生成"工具调用？因为模型自由生成常输出格式错误的 JSON：

```
用户："北京天气如何？"
模型自由生成：
  "北京今天天气不错，温度大概25度..."   ← 这是一段话，下游程序没法解析执行
  或 "{name: get_weather, args: {city: 北京}}"  ← JSON 格式错误（key 没引号）
```

下游解析脆弱，一个标点错误就崩。Function Call 的解法：用 **adv15 Guided Decoder** 把模型输出约束成**永远合法的 JSON 工具调用格式**，然后执行、回填、循环。

## 原理：ReAct 循环 model ↔ tool

```
用户提问："北京天气如何？人口多少？"
            │
            ▼
┌───────────────────────────────────────────────┐
│  循环（最多 max_iters 轮）                      │
│                                                 │
│  1. 模型（经 guided decoder 约束）输出工具调用    │
│     → {"name":"get_weather","args":{"city":"北京"}}│
│                                                 │
│  2. parse_tool_call 解析出 {name, args}           │
│                                                 │
│  3. execute_tool 执行（查天气/计算器/数据库）     │
│     → "北京: 晴 25°C"                            │
│                                                 │
│  4. 把结果拼回 prompt：[tool:get_weather]->...    │
│                                                 │
│  5. 模型看到结果，决定：继续调工具 or 给最终答案   │
└───────────────────────────────────────────────┘
            │
            ▼
  最终答案："北京晴 25°C，人口约 2189 万。"
```

关键：每轮模型输出都是**合法 JSON 工具调用**（由 adv15 的 regex/grammar 约束保证），`parse_tool_call` 一定能解析，不会崩。

## 实现细节

### TOOL_SCHEMA：工具注册表

```python
TOOL_SCHEMA = {
    "get_weather": {"args": ["city"], "returns": "str"},
    "calculator":  {"args": ["expr"], "returns": "str"},
}
```

真实框架里工具用 JSON Schema 描述参数类型，guided decoder 据此约束输出。教学版用简化的字典。

### parse_tool_call：从文本里抠出工具调用

```python
def parse_tool_call(text):
    m = re.search(r'\{.*\}', text, re.S)   # 抠出第一个 {...}
    obj = json.loads(m.group(0))
    if 'name' in obj and 'args' in obj: return obj
```

教学版用正则抠 JSON；真实版由 guided decoder 保证整段输出就是合法 JSON，无需正则。

### execute_tool：模拟执行

```python
def execute_tool(call):
    name, args = call['name'], call['args']
    if name == "get_weather":  return f"{args.get('city','?')}: 晴 25°C"
    if name == "calculator":   return str(eval(args.get('expr','0')))  # 教学用 eval
```

教学版用本地函数模拟；真实版调外部 API（HTTP/数据库）。

### tool_loop：ReAct 主循环

```python
def tool_loop(user_query, max_iters=3):
    prompt = user_query
    for _ in range(max_iters):
        out = scripted_model(prompt)          # 教学版：脚本驱动多轮不同工具
        call = parse_tool_call(out)
        if call is None:
            return prompt + " → 最终答案"      # 不再调工具，收敛
        result = execute_tool(call)
        prompt += f"\n[tool:{call['name']}]->{result}"
    return prompt
```

教学版用 `scripted_model` 按轮次返回不同工具调用（先 calculator 再 get_weather），最后一轮返回非 JSON 给出最终答案，体现"多轮工具 → 收敛"的完整轨迹。真实版由模型自己决定下一步。

### ❓ Q1：模型怎么"知道"有哪些工具可以调？

**问题**：模型不是只会生成文本吗，它怎么知道 `get_weather` 和 `calculator` 的存在？

**答案**：工具描述被**写进 system prompt**：

```
System: 你有以下工具可用：
  - get_weather(city): 查询指定城市的天气
  - calculator(expr): 计算数学表达式

请根据需要调用工具。如果不需要，直接回答用户问题。
```

模型在训练时见过大量"工具调用"的对话样本，学会了根据上下文选择合适的工具。**它不是真的"调用"函数，而是生成一个看起来像工具调用的 JSON 字符串**。下游解析器拿到这个 JSON 后，才真正执行函数。

### ❓ Q2：`eval()` 在教学版里用，生产环境有什么安全问题？

**问题**：`eval(args.get('expr','0'))` 可以执行任意 Python 代码！

**答案**：**绝对不要在生产环境用 eval 处理用户输入！** 教学版只是为了演示。生产系统的做法：

```python
# 安全替代方案：
import ast
# 方案 1：ast.literal_eval（只允许字面量表达式）
result = ast.literal_eval(safe_expr)  # 不能调用函数/导入模块

# 方案 2：专用计算器库
import numexpr
result = numexpr.evaluate(expr)  # 只支持数学运算

# 方案 3：沙箱执行
# 在隔离容器（Docker / gVisor）中执行，即使被注入也影响有限
```

OpenAI 的 calculator tool 内部也是沙箱执行，不是直接 eval。

### ❓ Q3：多工具并行（parallel tool calls）是怎么实现的？

**问题**：OpenAI 支持一次调多个工具（parallel tool calls），教学版是串行的。

**答案**：Parallel tool calls 的核心是**模型一次性输出多个工具调用**：

```json
{
  "tool_calls": [
    {"name": "get_weather", "args": {"city": "北京"}},
    {"name": "get_weather", "args": {"city": "上海"}}
  ]
}
```

执行时并行调用：
```python
results = asyncio.gather(
    execute_tool(call) for call in tool_calls
)
```

把结果一起回填到 prompt。收益：两个独立工具调用可以并发，延迟取 max 而非 sum。教学版是串行的，没有展示这个优化。

---

## 教学版 vs 真实框架

| 维度 | 教学版 | 真实框架 |
|------|--------|----------|
| 模型 | `fake_model_output` 脚本驱动 | 真 LLM 生成工具调用 |
| 输出约束 | 假设已是合法 JSON | adv15 guided decoder / JSON Schema 强约束 |
| 工具执行 | 本地函数 | 外部 HTTP API / 数据库 / 代码沙箱 |
| 循环 | 固定 max_iters | 模型自主决定何时收敛（生成 `final_answer`） |
| 多工具并行 | 串行 | OpenAI parallel tool calls |

真实框架参考：
- **OpenAI function calling**：`tools` + `tool_choice` 参数，模型输出 `tool_calls` 数组
- **vLLM guided tool calling**：用 Outlines/lm-format-enforcer 把输出约束成工具 schema
- **LangChain / LlamaIndex agent**：ReAct / Tool Calling agent 框架，封装上述循环

## 运行

```bash
python run.py
```

预期输出（循环轨迹）：

```
用户提问: 北京天气如何？人口多少？
[轮 1] 模型输出: {"name":"calculator","args":{"expr":"1899+29000000"}}
       执行 → 29001899
[轮 2] 模型输出: {"name":"get_weather","args":{"city":"北京"}}
       执行 → 北京: 晴 25°C
[轮 3] 模型输出: (无工具调用，给最终答案)
       最终答案: 北京晴 25°C，人口约 2900 万。

✅ adv16_function_call 通过
```

## 下一步

adv01–adv16 至此全部完成。回到主系列 [`SUMMARY.md`](../../SUMMARY.md) 查看完整优化手段速查表，或复习主系列任意一步对比进阶版与教学版差异。

本进阶系列覆盖的 16 项与主系列 15 步合起来，构成了一个相对完整的 LLM 推理优化知识地图：精度控制 → 解码加速 → 缓存结构 → 大规模并行 → 分离式架构 → 模型架构变体 → 服务工程。
