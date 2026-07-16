"""
adv14: Multi-LoRA 动态切换 — 演示与验证脚本

验证项:
  1. set_adapter(0) 与 set_adapter(1) 对同一输入产生不同输出
     (A/B 矩阵随机初始化,两个 adapter 参数不同)
  2. 切换 adapter 后立即生效(同输入,不同 adapter → 不同输出)
  3. base 权重的 requires_grad=False(已冻结)
"""

import torch
import torch.nn as nn

from lora import LoRALinear, MultiLoRAEngine


def main() -> None:
    torch.manual_seed(42)

    # ── 构造基础线性层并包裹为 LoRALinear ──────────────────────────────────
    in_features, out_features, r, num_adapters = 8, 8, 2, 2
    base_linear = nn.Linear(in_features, out_features, bias=False)
    lora_layer = LoRALinear(base_linear, r=r, num_adapters=num_adapters)

    # 构造包含 LoRALinear 的简单模型
    model = nn.Sequential(lora_layer)
    engine = MultiLoRAEngine(model, num_adapters=num_adapters)

    # ── 准备测试输入 ────────────────────────────────────────────────────────
    x = torch.randn(1, in_features)  # [1, 8]

    # ── 验证 1: base 权重已冻结 ─────────────────────────────────────────────
    for name, param in base_linear.named_parameters():
        assert not param.requires_grad, (
            f"base 权重 {name} 应该 requires_grad=False,但实际为 True"
        )
    print("✓ 验证 1: base 权重 requires_grad=False (已冻结)")

    # ── 验证 2: adapter 切换立即生效,输出不同 ─────────────────────────────
    out_0 = engine.generate(x, adapter_idx=0)
    out_1 = engine.generate(x, adapter_idx=1)

    print(f"\nadapter=0 输出: {out_0.squeeze().tolist()}")
    print(f"adapter=1 输出: {out_1.squeeze().tolist()}")

    assert not torch.allclose(out_0, out_1), (
        "adapter=0 与 adapter=1 的输出应该不同(A/B 初始化不同),但结果相同!"
    )
    print("✓ 验证 2: adapter=0 与 adapter=1 产生不同输出")

    # ── 验证 3: 同一 adapter 重复调用结果一致 ─────────────────────────────
    out_0_again = engine.generate(x, adapter_idx=0)
    assert torch.allclose(out_0, out_0_again), (
        "相同 adapter 对相同输入应产生相同输出"
    )
    print("✓ 验证 3: 切换回 adapter=0 后输出与首次一致")

    # ── 验证 4: LoRA 增量形状正确(base + delta 形状匹配) ─────────────────
    # 手动检查 forward 中的形状
    lora_layer.set_adapter(0)
    A, B = lora_layer.adapters['0']
    with torch.no_grad():
        x_A = A(x)        # [1, r]
        x_BA = B(x_A)     # [1, out_features]
        base_out = lora_layer.base(x)  # [1, out_features]

    assert x_A.shape == (1, r), f"A 输出形状应为 [1, {r}],实际 {x_A.shape}"
    assert x_BA.shape == (1, out_features), (
        f"B(A(x)) 形状应为 [1, {out_features}],实际 {x_BA.shape}"
    )
    assert base_out.shape == x_BA.shape, "base 输出与 LoRA 增量形状不匹配!"
    print(f"✓ 验证 4: 形状链正确 — x[1,8] → A → [{x_A.shape[1]}] → B → [1,8]")

    # ── 差值对比(直观展示两个 adapter 的差异) ─────────────────────────────
    diff = (out_0 - out_1).abs()
    print(f"\nadapter 输出差值 (L∞): {diff.max().item():.6f}")
    print(f"adapter 输出差值 (L2): {diff.norm().item():.6f}")

    print("\n✅ adv14_multi_lora 通过")


if __name__ == "__main__":
    main()
