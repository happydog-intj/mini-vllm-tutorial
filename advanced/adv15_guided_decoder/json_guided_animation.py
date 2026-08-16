"""
JSON 结构化输出动画 — 3Blue1Brown 风格

演示内容：
1. 模型自由生成 vs 约束生成的对比
2. JSON 格式的有限状态机(FSM)约束过程
3. 逐 token 掩码的完整流程可视化
4. 为什么 -inf mask 能保证 100% 合法 JSON

运行: manim -pql json_guided_animation.py JSONGuidedScene
高质量: manim -pqh json_guided_animation.py JSONGuidedScene
"""

from manim import *


class JSONGuidedScene(Scene):
    """JSON 结构化输出的引导解码动画"""

    def construct(self):
        self.scene_title()
        self.free_vs_guided()
        self.json_fsm_states()
        self.token_mask_step_by_step()
        self.logits_masking_math()
        self.summary()

    def scene_title(self):
        """标题页"""
        title = Text("模型如何输出合法 JSON？", font_size=42)
        subtitle = Text(
            "Guided Decoding — 每一步都在语法轨道上",
            font_size=24,
            color=GRAY,
        )
        subtitle.next_to(title, DOWN, buff=0.5)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

    def free_vs_guided(self):
        """第一幕：自由生成 vs 约束生成的对比"""
        header = Text("自由生成 vs 引导生成", font_size=32)
        header.to_edge(UP)
        self.play(Write(header))

        # 左侧：自由生成（错误百出）
        free_label = Text("自由生成", font_size=24, color=RED)
        free_label.move_to(LEFT * 3.5 + UP * 2)

        free_outputs = VGroup(
            Text('{"price": 3.1.4}', font_size=18, color=RED_B),
            Text('{price: "hello"}', font_size=18, color=RED_B),
            Text('{"price": maybe 3}', font_size=18, color=RED_B),
        )
        free_outputs.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        free_outputs.next_to(free_label, DOWN, buff=0.5)

        # X 标记
        x_marks = VGroup()
        for out in free_outputs:
            x_mark = Text("✗", font_size=22, color=RED)
            x_mark.next_to(out, LEFT, buff=0.2)
            x_marks.add(x_mark)

        self.play(Write(free_label))
        self.play(
            LaggedStart(
                *[FadeIn(o, shift=DOWN * 0.2) for o in free_outputs],
                lag_ratio=0.3,
            ),
            run_time=1.2,
        )
        self.play(LaggedStart(*[Write(x) for x in x_marks], lag_ratio=0.2))
        self.wait(0.5)

        # 右侧：约束生成（正确）
        guided_label = Text("引导生成", font_size=24, color=GREEN)
        guided_label.move_to(RIGHT * 3.5 + UP * 2)

        guided_output = Text('{"price": 3.14}', font_size=18, color=GREEN_B)
        guided_output.next_to(guided_label, DOWN, buff=0.5)

        check_mark = Text("✓", font_size=22, color=GREEN)
        check_mark.next_to(guided_output, LEFT, buff=0.2)

        guided_note = Text("每步只允许语法合法的 token", font_size=18, color=YELLOW)
        guided_note.next_to(guided_output, DOWN, buff=0.4)

        self.play(Write(guided_label))
        self.play(FadeIn(guided_output, shift=DOWN * 0.2), Write(check_mark))
        self.play(FadeIn(guided_note, shift=UP * 0.2))
        self.wait(2)

        self.play(
            FadeOut(header),
            FadeOut(free_label),
            FadeOut(free_outputs),
            FadeOut(x_marks),
            FadeOut(guided_label),
            FadeOut(guided_output),
            FadeOut(check_mark),
            FadeOut(guided_note),
        )

    def json_fsm_states(self):
        """第二幕：JSON 格式本质是一个有限状态机"""
        header = Text("JSON = 有限状态机", font_size=32)
        header.to_edge(UP)
        self.play(Write(header))

        # FSM 状态节点
        states = {
            "start": (LEFT * 5, "等待\n{", BLUE),
            "key": (LEFT * 2.5, '等待\n"key"', TEAL),
            "colon": (ORIGIN, "等待\n:", YELLOW),
            "value": (RIGHT * 2.5, "等待\nvalue", GREEN),
            "end": (RIGHT * 5, "等待\n}", RED_B),
        }

        state_circles = {}
        state_labels = {}

        for name, (pos, label_text, color) in states.items():
            circle = Circle(radius=0.55, color=color, fill_opacity=0.15)
            circle.move_to(pos + DOWN * 0.3)
            label = Text(label_text, font_size=14, color=color)
            label.move_to(circle.get_center())
            state_circles[name] = circle
            state_labels[name] = label

        # 绘制状态
        all_states = VGroup(*state_circles.values(), *state_labels.values())
        self.play(
            LaggedStart(
                *[Create(c) for c in state_circles.values()],
                lag_ratio=0.15,
            ),
            LaggedStart(
                *[FadeIn(l) for l in state_labels.values()],
                lag_ratio=0.15,
            ),
            run_time=2,
        )

        # 状态转移箭头
        state_names = ["start", "key", "colon", "value", "end"]
        arrows = VGroup()
        arrow_labels_group = VGroup()
        token_texts = ["{", '"key"', ":", "3.14", "}"]

        for i in range(len(state_names) - 1):
            src = state_circles[state_names[i]]
            dst = state_circles[state_names[i + 1]]
            arrow = Arrow(
                src.get_right(),
                dst.get_left(),
                buff=0.1,
                color=WHITE,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows.add(arrow)

            tok_label = Text(token_texts[i], font_size=14, color=GRAY_A)
            tok_label.next_to(arrow, UP, buff=0.1)
            arrow_labels_group.add(tok_label)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2),
            LaggedStart(
                *[FadeIn(l) for l in arrow_labels_group], lag_ratio=0.2
            ),
            run_time=2,
        )
        self.wait(0.5)

        # 高亮当前状态转移过程
        highlight = Circle(radius=0.65, color=YELLOW, stroke_width=4)
        highlight.move_to(state_circles["start"].get_center())
        self.play(Create(highlight), run_time=0.5)

        for i, name in enumerate(state_names[1:]):
            self.play(
                highlight.animate.move_to(
                    state_circles[name].get_center()
                ),
                run_time=0.6,
            )
            self.wait(0.3)

        # 底部说明
        explanation = Text(
            "每个状态只允许特定字符 → 非法 token 直接屏蔽",
            font_size=22,
            color=YELLOW,
        )
        explanation.to_edge(DOWN, buff=0.6)
        self.play(Write(explanation))
        self.wait(2)

        self.play(
            FadeOut(all_states),
            FadeOut(arrows),
            FadeOut(arrow_labels_group),
            FadeOut(highlight),
            FadeOut(explanation),
            FadeOut(header),
        )

    def token_mask_step_by_step(self):
        """第三幕：逐 token 掩码的完整流程"""
        header = Text("逐步生成: 模型如何输出 {\"val\": 42}", font_size=28)
        header.to_edge(UP)
        self.play(Write(header))

        # 已生成文本（逐步增长）
        steps = [
            # (已生成, 当前允许tokens, 选中token, 说明)
            ('', ['{'], '{', '开始: JSON 必须以 { 开头'),
            ('{', ['"'], '"', '对象内: 必须是 key 的引号'),
            ('{"', list('val"'), 'v', '字符串内: 合法字符'),
            ('{"v', list('al"'), 'a', '继续 key'),
            ('{"va', list('l"'), 'l', '继续 key'),
            ('{"val', ['"'], '"', '结束 key 引号'),
            ('{"val"', [':'], ':', 'key 之后必须是冒号'),
            ('{"val":', list(' 0123456789'), '4', '值: 数字开头'),
            ('{"val":4', list('0123456789,}'), '2', '继续数字或结束'),
            ('{"val":42', ['}', ','], '}', '对象结束'),
        ]

        # 已生成区域
        gen_label = Text("已生成:", font_size=20, color=GRAY)
        gen_label.move_to(LEFT * 5 + UP * 1.5)

        gen_text = Text('""', font_size=22, color=WHITE)
        gen_text.next_to(gen_label, RIGHT, buff=0.3)

        self.play(Write(gen_label), Write(gen_text))

        # 逐步展示前 5 步（避免太长）
        display_steps = [0, 1, 6, 7, 9]  # 关键步骤

        for step_idx in display_steps:
            generated, allowed, chosen, note = steps[step_idx]

            # 更新已生成文本
            new_gen = Text(
                f'"{generated}"' if generated else '""',
                font_size=22,
                color=WHITE,
            )
            new_gen.next_to(gen_label, RIGHT, buff=0.3)
            self.play(Transform(gen_text, new_gen), run_time=0.4)

            # 显示允许的 tokens
            allowed_label = Text("允许:", font_size=18, color=GREEN)
            allowed_label.move_to(LEFT * 5 + DOWN * 0.2)

            allowed_tokens_str = " ".join(
                [f"[{t}]" for t in allowed[:8]]
            )
            if len(allowed) > 8:
                allowed_tokens_str += " ..."
            allowed_display = Text(
                allowed_tokens_str, font_size=16, color=GREEN_B
            )
            allowed_display.next_to(allowed_label, RIGHT, buff=0.2)

            # 选中的 token（高亮）
            chosen_label = Text("选中:", font_size=18, color=YELLOW)
            chosen_label.move_to(LEFT * 5 + DOWN * 0.9)
            chosen_display = Text(
                f"[{chosen}]", font_size=20, color=YELLOW
            )
            chosen_display.next_to(chosen_label, RIGHT, buff=0.2)

            # 说明
            note_text = Text(note, font_size=18, color=GRAY_A)
            note_text.move_to(DOWN * 1.8)

            step_group = VGroup(
                allowed_label,
                allowed_display,
                chosen_label,
                chosen_display,
                note_text,
            )
            self.play(FadeIn(step_group), run_time=0.5)
            self.wait(1)
            self.play(FadeOut(step_group), run_time=0.3)

        # 最终结果
        final_result = Text(
            '最终输出: {"val": 42}  ← 100% 合法 JSON',
            font_size=24,
            color=GREEN,
        )
        final_result.move_to(DOWN * 1.0)
        self.play(Write(final_result))
        self.wait(2)

        self.play(
            FadeOut(header),
            FadeOut(gen_label),
            FadeOut(gen_text),
            FadeOut(final_result),
        )

    def logits_masking_math(self):
        """第四幕：logits mask 的数学原理"""
        header = Text("核心机制: Logits Masking", font_size=32)
        header.to_edge(UP)
        self.play(Write(header))

        # 展示 logits 向量（柱状图形式）
        token_names = ["{", "}", '"', "a", "5", ":", " ", "\\n"]
        logit_vals = [2.1, 0.3, 1.8, -0.5, 0.9, -1.2, 0.4, -0.8]
        allowed_mask = [True, False, False, False, False, False, False, False]
        # 当前状态: 开头, 只允许 {

        # 原始 logits 柱状图
        bars = VGroup()
        labels_row = VGroup()

        bar_width = 0.55
        start_x = -3.5

        for i, (name, val) in enumerate(zip(token_names, logit_vals)):
            height = max(abs(val) * 0.6, 0.08)
            color = BLUE if val >= 0 else BLUE_E
            bar = Rectangle(
                width=bar_width,
                height=height,
                fill_opacity=0.7,
                fill_color=color,
                stroke_width=1,
            )
            x_pos = start_x + i * (bar_width + 0.15)
            y_base = UP * 0.3
            if val >= 0:
                bar.move_to(RIGHT * x_pos + y_base + UP * height / 2)
            else:
                bar.move_to(RIGHT * x_pos + y_base + DOWN * height / 2)
            bars.add(bar)

            label = Text(name, font_size=14)
            label.move_to(RIGHT * x_pos + DOWN * 0.8)
            labels_row.add(label)

        logit_title = Text("模型输出 logits", font_size=20, color=GRAY_A)
        logit_title.move_to(UP * 2.2)
        self.play(Write(logit_title))
        self.play(
            LaggedStart(
                *[GrowFromEdge(b, DOWN) for b in bars],
                lag_ratio=0.08,
            ),
            FadeIn(labels_row),
            run_time=1.5,
        )
        self.wait(0.8)

        # mask 步骤：不合法的 token 变红
        mask_label = Text(
            "当前状态: 开头 → 只允许 { ", font_size=18, color=YELLOW
        )
        mask_label.move_to(DOWN * 1.5)
        self.play(Write(mask_label))
        self.wait(0.5)

        # 将不合法 token 变红并缩小
        fade_anims = []
        for i, is_allowed in enumerate(allowed_mask):
            if not is_allowed:
                fade_anims.append(
                    bars[i].animate.set_fill(RED_D, opacity=0.2).stretch_to_fit_height(0.05)
                )
        self.play(*fade_anims, run_time=1)

        # -inf 标注
        inf_label = Text(
            "非法 token: logit → -∞", font_size=22, color=RED
        )
        inf_label.move_to(DOWN * 2.2)
        self.play(Write(inf_label))
        self.wait(0.5)

        # softmax 结果
        softmax_eq = Text(
            "softmax:  P({) = 1.0,  P(其余) = 0",
            font_size=22,
            color=GREEN,
        )
        softmax_eq.move_to(DOWN * 3.0)
        self.play(Write(softmax_eq))

        # 高亮存活的 bar
        surviving = SurroundingRectangle(bars[0], color=GREEN, buff=0.08)
        self.play(Create(surviving))
        self.wait(2)

        self.play(
            FadeOut(header),
            FadeOut(logit_title),
            FadeOut(bars),
            FadeOut(labels_row),
            FadeOut(mask_label),
            FadeOut(inf_label),
            FadeOut(softmax_eq),
            FadeOut(surviving),
        )

    def summary(self):
        """总结页"""
        title = Text("总结: 模型如何保证输出合法 JSON", font_size=32)
        title.to_edge(UP)

        points = VGroup(
            Text("1. JSON 语法 → 有限状态机, 每个状态定义合法后继", font_size=22),
            Text("2. 每步解码前, 对 logits 做 mask: 非法 token → -∞", font_size=22),
            Text("3. softmax 后非法 token 概率=0, 永远不会被选中", font_size=22),
            Text("4. 结果: 每个 token 都在语法轨道上 → 100% 合法", font_size=22),
        )
        points.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        points.next_to(title, DOWN, buff=0.8)
        points.shift(LEFT * 0.5)

        points[0].set_color(BLUE_B)
        points[1].set_color(TEAL_B)
        points[2].set_color(GREEN_B)
        points[3].set_color(YELLOW)

        self.play(Write(title))
        for point in points:
            self.play(FadeIn(point, shift=RIGHT * 0.3), run_time=0.8)
            self.wait(0.5)

        # 核心公式
        formula = Text(
            "P(t) = exp(z_t) / Σ exp(z_v)  ·  𝟙[t ∈ V_valid]",
            font_size=24,
        )
        formula.to_edge(DOWN, buff=0.8)
        self.play(Write(formula))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])
