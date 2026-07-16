"""
adv16_function_call/tool_loop.py

ReAct 风格工具调用循环（教学版）。

核心思路
--------
1. 模型（fake_model_output）输出一个 JSON 工具调用——在真实系统中这由
   adv15 的 Guided Decoder 约束，保证输出合法 JSON。
2. parse_tool_call 从模型输出中提取 {name, args}。
3. execute_tool 模拟执行工具（get_weather / calculator）。
4. 把执行结果拼回 prompt，进入下一轮。
5. 模型在最后一轮不再输出工具调用 JSON，而是给出自然语言最终答案，循环终止。

多轮调用策略
------------
script 参数（list[str | None]）驱动每轮模型输出：
- 字符串：当轮输出的 forced JSON（含工具调用）
- None：当轮不输出工具 JSON，触发循环提前结束
"""

import json
import re

# ---------------------------------------------------------------------------
# 工具 Schema（供外部引用）
# ---------------------------------------------------------------------------
TOOL_SCHEMA = {
    "get_weather": {"args": ["city"], "returns": "str"},
    "calculator":  {"args": ["expr"],  "returns": "str"},
}


# ---------------------------------------------------------------------------
# 模型输出模拟
# ---------------------------------------------------------------------------
def fake_model_output(prompt: str, forced_json: str | None) -> str:
    """
    教学版：不跑真模型。

    - forced_json 非 None：模拟 guided decoder 约束下的工具调用 JSON 输出。
    - forced_json 为 None：模拟模型决定直接回答，不调用工具。
    """
    if forced_json is None:
        # 最终回答轮：直接返回自然语言摘要（不含 JSON）
        return "根据以上信息，可以综合回答用户的问题。"
    return forced_json


# ---------------------------------------------------------------------------
# 解析 / 执行
# ---------------------------------------------------------------------------
def parse_tool_call(text: str) -> dict | None:
    """从模型输出解析 {name, args}。失败返回 None。"""
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if 'name' in obj and 'args' in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


def execute_tool(call: dict) -> str:
    """模拟执行工具，返回结果字符串。"""
    name = call['name']
    args = call.get('args', {})
    if name == 'get_weather':
        city = args.get('city', '?')
        return f"{city}: 晴 25°C"
    if name == 'calculator':
        expr = args.get('expr', '0')
        try:
            # 仅允许简单算术表达式，eval 在教学环境下可接受
            return str(eval(expr, {"__builtins__": {}}))  # noqa: S307
        except Exception as e:
            return f"err: {e}"
    return "unknown tool"


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def tool_loop(
    user_query: str,
    script: list[str | None] | None = None,
    max_iters: int = 3,
    verbose: bool = False,
) -> str:
    """
    ReAct 风格循环：模型输出工具调用 → 执行 → 把结果拼回 prompt → 直到最终答案。

    Parameters
    ----------
    user_query : str
        用户问题。
    script : list[str | None] | None
        每轮 forced JSON 脚本（None 表示该轮给出最终答案）。
        默认脚本：先调用 calculator，再调用 get_weather，第三轮给出最终答案。
    max_iters : int
        最大循环轮数，防止无限循环。
    verbose : bool
        是否打印每轮轨迹。

    Returns
    -------
    str
        含工具调用历史的 prompt（最终状态）。
    """
    if script is None:
        # 默认多轮脚本：两轮工具调用 + 一轮最终回答
        script = [
            '{"name":"calculator","args":{"expr":"1400*10000"}}',
            '{"name":"get_weather","args":{"city":"北京"}}',
            None,  # 最终回答轮，不调用工具
        ]

    prompt = user_query

    for i in range(max_iters):
        forced = script[i] if i < len(script) else None
        out = fake_model_output(prompt, forced)
        call = parse_tool_call(out)

        if call is None:
            # 模型决定直接回答（无工具调用 JSON）
            final_answer = f"\n[最终答案] {out}"
            prompt += final_answer
            if verbose:
                print(f"  轮次 {i+1}: 无工具调用 -> 给出最终答案")
            break

        result = execute_tool(call)
        entry = f"\n[tool:{call['name']}({json.dumps(call['args'], ensure_ascii=False)})]->{result}"
        prompt += entry

        if verbose:
            print(f"  轮次 {i+1}: 调用 {call['name']}({call['args']}) -> {result}")
    else:
        # 达到 max_iters 仍未给出最终答案
        prompt += "\n[最终答案] 已达最大轮次，以上为工具执行记录。"

    return prompt
