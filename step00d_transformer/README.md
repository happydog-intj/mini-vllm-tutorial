# step00d — Transformer：完整 Decoder 层

## 教学目标

理解 Attention + MLP + 残差连接 + RMSNorm 组成的完整 Decoder 层。

## Pre-Norm 结构（现代 LLM 标准）

```
输入 x [seq_len, d_model]
  │
  ├─→ RMSNorm → MultiHeadAttention → (+x) ← 残差
  │
  ├─→ RMSNorm → MLP → (+x)               ← 残差
  │
输出 x [seq_len, d_model]
```

## 残差连接：梯度的高速公路

```python
x = x + attention(norm(x))   # 不是 x = attention(x)
#   ↑
#   ∣── 梯度可以直接从输出流到输入，绕过 attention
#       解决了深层网络训练不稳的问题
```

## MLP (SwiGLU)

```
x → W_gate → SiLU ─┐
                     × → W_down → output
x → W_up   ────────┘

d_model=128 → d_ff=512 → d_model=128
```

## RMSNorm vs LayerNorm

```
LayerNorm: y = (x - mean(x)) / std(x) * γ + β   # 需要算均值+方差
RMSNorm:   y = x / sqrt(mean(x²)) * γ            # 只算均方根，更快
```

## 运行

```bash
python run.py
```
