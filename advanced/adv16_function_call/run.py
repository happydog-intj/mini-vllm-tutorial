"""
adv16_function_call/run.py

演示并验证工具调用循环（ReAct 风格）。

断言
----
① parse_tool_call 能从含 JSON 文本中解析出 {name, args}
② execute_tool 对每个工具均返回非空字符串
③ 最终 prompt 含 [tool: 标记（说明执行过工具）
④ 循环能在 max_iters 内收敛到最终答案
"""

import json
from tool_loop import (
    TOOL_SCHEMA,
    parse_tool_call,
    execute_tool,
    tool_loop,
)


def run_assertions() -> None:
    """单元断言：逐一验证各核心函数。"""
    print("=" * 60)
    print("adv16_function_call — 单元断言")
    print("=" * 60)

    # ① parse_tool_call：从含噪文本中提取 JSON
    sample_texts = [
        '好的，我来查天气：{"name":"get_weather","args":{"city":"上海"}} 请稍候',
        '{"name":"calculator","args":{"expr":"2+2"}}',
        '这是一段不含工具调用的纯文本回答。',
    ]
    parsed = [parse_tool_call(t) for t in sample_texts]

    assert parsed[0] is not None and parsed[0]['name'] == 'get_weather', \
        f"断言①失败: 应解析出 get_weather，实际={parsed[0]}"
    assert parsed[0]['args']['city'] == '上海', \
        f"断言①失败: city 应为上海，实际={parsed[0]['args']}"

    assert parsed[1] is not None and parsed[1]['name'] == 'calculator', \
        f"断言①失败: 应解析出 calculator，实际={parsed[1]}"

    assert parsed[2] is None, \
        f"断言①失败: 纯文本应解析为 None，实际={parsed[2]}"

    print("  ✓ ① parse_tool_call：从含 JSON 文本中解析工具调用")

    # ② execute_tool 对各工具均返回非空字符串
    result_weather = execute_tool({'name': 'get_weather', 'args': {'city': '北京'}})
    assert result_weather and isinstance(result_weather, str), \
        f"断言②失败: get_weather 返回空或非字符串，实际={result_weather!r}"

    result_calc = execute_tool({'name': 'calculator', 'args': {'expr': '6 * 7'}})
    assert result_calc and isinstance(result_calc, str), \
        f"断言②失败: calculator 返回空或非字符串，实际={result_calc!r}"
    assert result_calc == '42', \
        f"断言②失败: 6*7 应为 42，实际={result_calc!r}"

    result_unknown = execute_tool({'name': 'unknown_tool', 'args': {}})
    assert result_unknown and isinstance(result_unknown, str), \
        f"断言②失败: 未知工具应返回非空字符串，实际={result_unknown!r}"

    print("  ✓ ② execute_tool：各工具返回非空字符串（get_weather / calculator / unknown）")

    print()


def run_tool_loop_demo() -> None:
    """端到端演示：多轮工具调用循环，打印轨迹。"""
    print("=" * 60)
    print("adv16_function_call — 端到端工具调用循环")
    print("=" * 60)
    print()

    user_query = "北京天气如何？北京人口大约多少？"
    print(f"用户问题: {user_query}")
    print()

    # 自定义脚本：轮次1 → calculator（估算人口），轮次2 → get_weather，轮次3 → 最终答案
    script = [
        '{"name":"calculator","args":{"expr":"1400*10000"}}',   # 估算人口（1400万*1万）
        '{"name":"get_weather","args":{"city":"北京"}}',
        None,  # 最终回答轮
    ]

    print("循环轨迹:")
    final_prompt = tool_loop(
        user_query,
        script=script,
        max_iters=5,
        verbose=True,
    )

    print()
    print("最终 prompt（含工具调用历史）:")
    print("-" * 60)
    print(final_prompt)
    print("-" * 60)
    print()

    # ③ 最终 prompt 含 [tool: 标记
    assert '[tool:' in final_prompt, \
        f"断言③失败: 最终 prompt 中未发现 [tool: 标记\n{final_prompt}"
    print("  ✓ ③ 最终 prompt 含 [tool: 标记（执行过工具）")

    # ④ 含最终答案标记（说明循环在 max_iters 内收敛）
    assert '[最终答案]' in final_prompt, \
        f"断言④失败: 未发现 [最终答案] 标记，循环可能未收敛\n{final_prompt}"
    print("  ✓ ④ 循环在 max_iters 内收敛到最终答案")

    # 额外验证：两种工具都被调用过
    assert '[tool:calculator' in final_prompt, \
        "断言附加: 未发现 calculator 调用记录"
    assert '[tool:get_weather' in final_prompt, \
        "断言附加: 未发现 get_weather 调用记录"
    print("  ✓ 附加: calculator 和 get_weather 均在轨迹中")

    print()


def show_tool_schema() -> None:
    """打印工具 Schema（教学信息）。"""
    print("=" * 60)
    print("注册的工具 Schema")
    print("=" * 60)
    for name, spec in TOOL_SCHEMA.items():
        print(f"  {name}: args={spec['args']}, returns={spec['returns']}")
    print()


if __name__ == "__main__":
    show_tool_schema()
    run_assertions()
    run_tool_loop_demo()
    print("\n✅ adv16_function_call 通过")
