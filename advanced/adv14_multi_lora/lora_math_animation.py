"""
LoRA 数学原理动画 — 3Blue1Brown 风格

演示内容：
1. 全量微调 vs 低秩分解的几何直觉
2. 瓶颈结构：压缩再展开
3. 多 LoRA 切换的加法优雅性

运行: manim -pql lora_math_animation.py LoRAMathScene
高质量: manim -pqh lora_math_animation.py LoRAMathScene

注意: 本动画不依赖 LaTeX，使用 Text 替代 MathTex 以避免 LaTeX 安装要求。
"""

from manim import *


class LoRAMathScene(Scene):
    """LoRA 低秩适配的数学直觉动画"""

    def construct(self):
        self.scene_title()
        self.full_finetune_problem()
        self.low_rank_intuition()
        self.bottleneck_geometry()
        self.multi_lora_switch()
        self.summary()

    def scene_title(self):
        """标题页"""
        title = Text("LoRA: 低秩适配的数学直觉", font_size=42)
        subtitle = Text(
            "Low-Rank Adaptation — 为什么 0.4% 参数就够了？",
            font_size=24,
            color=GRAY,
        )
        subtitle.next_to(title, DOWN, buff=0.5)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

    def full_finetune_problem(self):
        """第一幕：全量微调的参数爆炸问题"""
        header = Text("全量微调：修改整个权重矩阵", font_size=32)
        header.to_edge(UP)
        self.play(Write(header))

        # 大矩阵 W0
        w_matrix = self._create_matrix_grid(8, 8, cell_size=0.4, color=BLUE_D)
        w_label = Text("W₀", font_size=36)
        w_label.next_to(w_matrix, UP, buff=0.3)

        w_group = VGroup(w_matrix, w_label)
        w_group.move_to(LEFT * 3)

        self.play(FadeIn(w_group))
        self.wait(0.5)

        # 展示 ΔW 也是满秩大矩阵
        delta_matrix = self._create_matrix_grid(8, 8, cell_size=0.4, color=RED_D)
        delta_label = Text("ΔW", font_size=36, color=RED)
        delta_label.next_to(delta_matrix, UP, buff=0.3)

        delta_group = VGroup(delta_matrix, delta_label)
        delta_group.move_to(RIGHT * 3)

        plus_sign = Text("+", font_size=48)
        plus_sign.move_to(ORIGIN)

        self.play(FadeIn(plus_sign), FadeIn(delta_group))
        self.wait(0.5)

        # 参数量标注
        param_text = Text(
            "d=4096 → ΔW: d² = 16,777,216 参数",
            font_size=26,
            color=RED,
        )
        param_text.next_to(VGroup(w_group, delta_group), DOWN, buff=0.8)

        self.play(Write(param_text))
        self.wait(2)

        self.play(
            FadeOut(w_group),
            FadeOut(plus_sign),
            FadeOut(delta_group),
            FadeOut(param_text),
            FadeOut(header),
        )

    def low_rank_intuition(self):
        """第二幕：低秩假设的直觉 — ΔW 的奇异值集中在前几个"""
        header = Text("核心假设：微调的变化是低秩的", font_size=32)
        header.to_edge(UP)
        self.play(Write(header))

        # 奇异值条形图
        bar_heights = [3.0, 2.2, 1.4, 0.7, 0.3, 0.15, 0.08, 0.04]
        bars = VGroup()
        bar_labels = VGroup()

        for i, h in enumerate(bar_heights):
            bar = Rectangle(
                width=0.5,
                height=h,
                fill_opacity=0.8,
                fill_color=interpolate_color(YELLOW, RED, i / len(bar_heights)),
                stroke_width=1,
            )
            bar.move_to(LEFT * 2.5 + RIGHT * i * 0.7 + DOWN * (2.0 - h / 2))
            bars.add(bar)

            label = Text(f"σ{i+1}", font_size=18)
            label.next_to(bar, DOWN, buff=0.1)
            bar_labels.add(label)

        axis_label = Text("ΔW 的奇异值分布", font_size=22)
        axis_label.next_to(bars, DOWN, buff=0.5)

        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.1),
            run_time=1.5,
        )
        self.play(FadeIn(bar_labels), Write(axis_label))
        self.wait(1)

        # 标注前 r 个占主导
        brace = Brace(VGroup(bars[0], bars[1], bars[2]), UP, color=GREEN)
        brace_text = Text("rank r=3\n捕捉 >95% 信息", font_size=20, color=GREEN)
        brace_text.next_to(brace, UP, buff=0.15)

        self.play(Create(brace), Write(brace_text))
        self.wait(1)

        # 右侧：公式
        formula_line1 = Text("ΔW ≈ B · A", font_size=36)
        formula_line1.move_to(RIGHT * 3.5 + UP * 0.8)
        formula_line1[3].set_color(RED)  # Δ
        formula_line1[4].set_color(RED)  # W

        dim_a = Text("A: [d × r]  降维", font_size=24, color=TEAL)
        dim_b = Text("B: [r × d]  升维", font_size=24, color=GREEN)
        dim_a.next_to(formula_line1, DOWN, buff=0.5)
        dim_b.next_to(dim_a, DOWN, buff=0.2)

        param_saving = Text(
            "2dr = 65,536 (仅 0.4%)",
            font_size=24,
            color=YELLOW,
        )
        param_saving.next_to(dim_b, DOWN, buff=0.4)

        self.play(Write(formula_line1))
        self.play(FadeIn(dim_a, shift=UP * 0.2), FadeIn(dim_b, shift=UP * 0.2))
        self.play(Write(param_saving))
        self.wait(2)

        self.play(
            FadeOut(bars),
            FadeOut(bar_labels),
            FadeOut(axis_label),
            FadeOut(brace),
            FadeOut(brace_text),
            FadeOut(formula_line1),
            FadeOut(dim_a),
            FadeOut(dim_b),
            FadeOut(param_saving),
            FadeOut(header),
        )

    def bottleneck_geometry(self):
        """第三幕：瓶颈结构的几何直觉 — 压缩再展开"""
        header = Text("几何直觉：压缩到瓶颈再展开", font_size=32)
        header.to_edge(UP)
        self.play(Write(header))

        # 左侧：输入空间 (大圆)
        input_space = Circle(radius=1.5, color=BLUE, fill_opacity=0.1)
        input_space.move_to(LEFT * 4)
        input_label = Text("输入空间\nd=4096 维", font_size=18)
        input_label.next_to(input_space, DOWN, buff=0.3)

        # 中间：瓶颈 (小圆)
        bottleneck = Circle(radius=0.4, color=YELLOW, fill_opacity=0.3)
        bottleneck.move_to(ORIGIN)
        bn_label = Text("瓶颈\nr=8 维", font_size=18, color=YELLOW)
        bn_label.next_to(bottleneck, DOWN, buff=0.3)

        # 右侧：输出空间 (大圆)
        output_space = Circle(radius=1.5, color=GREEN, fill_opacity=0.1)
        output_space.move_to(RIGHT * 4)
        output_label = Text("输出空间\nd=4096 维", font_size=18)
        output_label.next_to(output_space, DOWN, buff=0.3)

        self.play(Create(input_space), Write(input_label))
        self.play(Create(bottleneck), Write(bn_label))
        self.play(Create(output_space), Write(output_label))

        # 箭头 A: 降维
        arrow_a = Arrow(
            input_space.get_right(),
            bottleneck.get_left(),
            buff=0.1,
            color=TEAL,
            stroke_width=3,
        )
        a_label = Text("A", font_size=28, color=TEAL)
        a_label.next_to(arrow_a, UP, buff=0.15)

        # 箭头 B: 升维
        arrow_b = Arrow(
            bottleneck.get_right(),
            output_space.get_left(),
            buff=0.1,
            color=GREEN_D,
            stroke_width=3,
        )
        b_label = Text("B", font_size=28, color=GREEN_D)
        b_label.next_to(arrow_b, UP, buff=0.15)

        self.play(GrowArrow(arrow_a), Write(a_label))
        self.play(GrowArrow(arrow_b), Write(b_label))
        self.wait(1)

        # 动画：一个点从输入空间经过瓶颈到输出空间
        dot = Dot(input_space.get_center(), color=WHITE, radius=0.1)
        self.play(FadeIn(dot))

        path1 = Line(input_space.get_center(), bottleneck.get_center())
        path2 = Line(bottleneck.get_center(), output_space.get_center())

        self.play(MoveAlongPath(dot, path1), run_time=0.8)
        self.play(dot.animate.scale(0.4), run_time=0.3)  # 压缩
        self.play(MoveAlongPath(dot, path2), run_time=0.8)
        self.play(dot.animate.scale(2.5), run_time=0.3)  # 展开

        # 底部解释
        explanation = Text(
            "x → A(压缩到r维) → B(展开回d维) → Δy",
            font_size=24,
        )
        explanation.to_edge(DOWN, buff=0.5)
        self.play(Write(explanation))
        self.wait(2)

        self.play(
            *[
                FadeOut(m)
                for m in [
                    input_space, input_label, bottleneck, bn_label,
                    output_space, output_label, arrow_a, a_label,
                    arrow_b, b_label, dot, explanation, header,
                ]
            ]
        )

    def multi_lora_switch(self):
        """第四幕：Multi-LoRA 的加法切换"""
        header = Text("Multi-LoRA：同一个 base，不同的方向", font_size=32)
        header.to_edge(UP)
        self.play(Write(header))

        # base 输出向量 (中心)
        origin = DOWN * 0.5
        base_arrow = Arrow(
            origin, origin + UP * 2.5, color=WHITE, stroke_width=4, buff=0
        )
        base_label = Text("W₀·x", font_size=26)
        base_label.next_to(base_arrow, LEFT, buff=0.3)

        self.play(GrowArrow(base_arrow), Write(base_label))
        self.wait(0.5)

        # 三个 LoRA adapter 的偏移方向
        lora_configs = [
            (UP * 2.5 + LEFT * 1.8, RED, "客服"),
            (UP * 2.5 + RIGHT * 0.2, GREEN, "代码"),
            (UP * 2.5 + RIGHT * 2.2, PURPLE, "翻译"),
        ]

        lora_arrows = []
        for offset, color, task_name in lora_configs:
            tip = origin + offset
            arrow = Arrow(
                origin + UP * 2.5,
                tip,
                color=color,
                stroke_width=3,
                buff=0,
            )
            label = Text(task_name, font_size=20, color=color)
            label.next_to(arrow.get_end(), UP, buff=0.15)
            lora_arrows.append((arrow, label))

        # 逐个展示 adapter
        for arrow, label in lora_arrows:
            self.play(GrowArrow(arrow), Write(label), run_time=0.7)

        self.wait(1)

        # 公式
        formula = Text("y_i = W₀·x + B_i·A_i·x", font_size=28)
        formula.to_edge(DOWN, buff=1.0)

        switch_text = Text(
            "切换任务 = 换一对 (A, B) 指针，零拷贝",
            font_size=22,
            color=YELLOW,
        )
        switch_text.next_to(formula, DOWN, buff=0.3)

        self.play(Write(formula))
        self.play(FadeIn(switch_text, shift=UP * 0.2))
        self.wait(2)

        self.play(
            FadeOut(base_arrow),
            FadeOut(base_label),
            *[FadeOut(a) for a, _ in lora_arrows],
            *[FadeOut(l) for _, l in lora_arrows],
            FadeOut(formula),
            FadeOut(switch_text),
            FadeOut(header),
        )

    def summary(self):
        """总结页"""
        title = Text("总结", font_size=36)
        title.to_edge(UP)

        points = VGroup(
            Text("1. W' = W₀ + B·A，rank(BA) = r ≪ d", font_size=26),
            Text("2. 参数量: 2dr vs d²（节省 256×）", font_size=26),
            Text("3. 切换 adapter = O(1)，base 模型不动", font_size=26),
        )
        points.arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        points.next_to(title, DOWN, buff=0.8)

        self.play(Write(title))
        for point in points:
            self.play(FadeIn(point, shift=RIGHT * 0.3), run_time=0.8)
            self.wait(0.5)

        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ─── 辅助方法 ───

    def _create_matrix_grid(self, rows, cols, cell_size=0.4, color=BLUE):
        """创建一个网格表示矩阵"""
        grid = VGroup()
        for i in range(rows):
            for j in range(cols):
                cell = Square(
                    side_length=cell_size,
                    fill_opacity=0.6,
                    fill_color=color,
                    stroke_width=0.5,
                    stroke_color=WHITE,
                )
                cell.move_to(RIGHT * j * cell_size + DOWN * i * cell_size)
                grid.add(cell)
        grid.move_to(ORIGIN)
        return grid
