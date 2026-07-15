# adv01 量化 Quantization：W4A16 / W8A16 权重低比特推理

## 1. 教学目标

理解权重量化（Weight-Only Quantization）的核心原理，并能手写对称 per-tensor 量化与反量化，在 TinyTransformer 上验证 FP32 → INT8 → INT4 的精度与存储折衷。

---

## 2. 问题：主系列为什么没解决这个？

主系列（step01–step16）始终用 **BF16 / FP32** 存储权重，专注于讲解 KV Cache、调度、分页内存、CUDA Graph 等推理系统机制。量化是模型压缩的独立维度：

- 主系列的模型权重每个参数占 4 字节（FP32），或 2 字节（BF16）
- 真实 70B 模型以 FP16 存储需约 140 GB 显存，单机根本跑不起来
- 量化把权重压到 INT8（1 byte/param）或 INT4（0.5 byte/param），70B 模型降至约 35 GB 或 17.5 GB

主系列没有量化，是刻意的教学分层：先让读者理解推理系统结构，再通过进阶章节单独拆解模型压缩。

---

## 3. 原理

### 对称量化公式

```
FP32 权重 w（任意实数）
          │
          ▼
   scale = max|w| / qmax        qmax = 7 (INT4) 或 127 (INT8)
          │
          ▼
   q = round(w / scale)         整数，范围 [-qmax, qmax]
          │
          ▼
   反量化：w_approx = q * scale  ≈ w（有舍入误差）
```

### 存储布局对比

```
FP32 权重（每个参数 4 字节）
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│float32│float32│float32│float32│float32│float32│float32│float32│  → 32 bytes（8个参数）
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

INT8 量化（每个参数 1 字节，同时存 1 个 float32 scale）
┌──┬──┬──┬──┬──┬──┬──┬──┐  + ┌───────┐
│i8│i8│i8│i8│i8│i8│i8│i8│    │ scale │  → 8 + 4 = 12 bytes（节省约 3/4）
└──┴──┴──┴──┴──┴──┴──┴──┘    └───────┘

INT4 量化（每个参数理论 0.5 字节，打包存储）
┌────┬────┬────┬────┐          + ┌───────┐
│i4i4│i4i4│i4i4│i4i4│            │ scale │  → 4 + 4 = 8 bytes（节省约 7/8）
└────┴────┴────┴────┘            └───────┘
  两个 INT4 打包进一个字节

注：教学版用 int8 容器存 INT4 值（未真正打包），bytes_int4_stored = bytes_int8_stored
    生产实现（bitsandbytes、AutoAWQ）会真正打包，节省额外 2×
```

### W4A16 / W8A16 推理流程

```
推理时（每次 forward）

权重（INT8/INT4 低比特存储）
        │
        │  dequantize: w = q * scale
        ▼
权重（FP16，与激活同精度）
        │
        │  matmul: out = x @ w.T
        ▼
输出（FP16）

激活 x 始终保持 FP16（A16），matmul 精度不降；
仅权重低比特存储（W8/W4），显存大幅减少。
```

---

## 4. 实现细节

### `quantize_weight` — 量化

```python
def quantize_weight(w: torch.Tensor, bits: int = 8):
    qmax = (1 << (bits - 1)) - 1          # INT8: 127  INT4: 7
    scale = w.abs().max() / qmax           # per-tensor scale
    q = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale
```

- `qmax` 由 `bits` 决定：INT8 为 127，INT4 为 7
- `scale` 是单个标量（per-tensor），教学版最简单；生产用 per-channel / groupwise
- `torch.int8` 作容器：INT4 值域 `[-7, 7]` 完全在 int8 范围内，无溢出

### `dequantize_weight` — 反量化

```python
def dequantize_weight(q: torch.Tensor, scale: torch.Tensor, bits: int = 8):
    return q.to(torch.float32) * scale
```

- 先转 float32 再乘 scale，避免整型溢出
- 调用方再 `.to(x.dtype)` 匹配激活精度（FP16 / BF16 / FP32）

### `QuantizedLinear` — 量化线性层

```python
class QuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int = 8):
        super().__init__()
        q, scale = quantize_weight(linear.weight.data, bits)
        self.register_buffer('q', q)        # 随模型 .to(device) 迁移
        self.register_buffer('scale', scale)
        self.bias = linear.bias

    def forward(self, x):
        w = dequantize_weight(self.q, self.scale).to(x.dtype)
        out = x @ w.t()
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out
```

- 继承 `nn.Module`：PyTorch 要求子模块必须是 `nn.Module`，否则 `setattr` 报错
- `register_buffer`：量化权重不是可训练参数，用 buffer 注册以便随 `.to(device)` 迁移
- bias 不量化：bias 参数量极少（相对权重矩阵），量化收益微小，复杂度不值得

### `quantize_model` — 递归替换

```python
def quantize_model(model, bits=8):
    for name, mod in list(model.named_children()):
        if isinstance(mod, nn.Linear):
            setattr(model, name, QuantizedLinear(mod, bits))
        else:
            quantize_model(mod, bits)   # 递归子模块
    return model
```

- `named_children()` 只遍历直接子模块，`setattr` 替换直接属性
- 递归处理嵌套结构（如 `TransformerDecoderLayerWithKV` 内部的 `MultiHeadAttentionWithKVCache`）
- 对 TinyTransformer 的 `ModuleList` 中每层直接子 `nn.Linear` 均可覆盖

---

## 5. 教学版 vs 真实框架

```
维度              教学版（本章）                   真实框架（vLLM / AWQ / GPTQ）
────────────────────────────────────────────────────────────────────────────────
量化粒度          per-tensor（全层共享 1 个 scale）  per-channel：每行 1 个 scale
                                                    groupwise：每 128 个参数 1 个 scale
                                                    → 显著减少量化误差

INT4 存储         int8 容器（未打包）               真正打包：2个INT4 → 1个uint8
                                                    存储量再减半

校准方法          无（直接 round+clamp）            AWQ：激活感知缩放，保护显著权重
                                                    GPTQ：Hessian 最小二乘优化

支持格式          INT8、INT4（对称）                INT8、INT4、FP8（e4m3 / e5m2）
                                                    NF4（bitsandbytes 4-bit NormalFloat）

matmul 加速       无（反量化后普通 matmul）          CUDA kernel 直接在 INT8/INT4 空间做
                                                    GEMM（cutlass / Marlin kernel）
                                                    → 比 FP16 还快

vLLM 接入         —                                 --quantization awq / gptq / fp8
                                                    AutoAWQ、bitsandbytes 库集成
```

真实量化推理的性能收益来自两个叠加效应：
1. **显存节省**：更小的权重 → 更大 batch / 更长序列
2. **带宽节省**：GPU 的推理瓶颈常在显存带宽，读取 INT8 权重比 FP16 快 2×，INT4 快 4×

---

## 6. 运行

```bash
cd advanced/adv01_quantization
python run.py
```

预期输出（关键部分）：

```
========================================================
logits 对比（最后一个 token 位置，前 8 个词元）
========================================================
  FP32  : [-0.082, 1.112, 0.487, -0.275, -0.881, -0.038, 0.947, -0.238]
  INT8  : [-0.082, 1.107, 0.487, -0.277, -0.879, -0.036, 0.950, -0.239]
  INT4  : [-0.109, 1.139, 0.484, -0.289, -0.985, -0.028, 0.971, -0.313]

  相对误差 FP32 vs INT8 : 4.2868e-03
  相对误差 FP32 vs INT4 : 8.2490e-02
  ✓ INT8 相对误差 < 1%，INT4 相对误差 < 15%（qmax=7，教学模型正常范围）

========================================================
权重存储大小对比
========================================================
  FP32  : 5120 bytes (参考基准)
  INT8  : 1280 bytes 存储  (0.25x，理论 0.25x)
  INT4  : 1280 bytes (int8 容器)  | 打包后理论 640 bytes (0.12x，理论 0.125x)
  ✓ 权重存储大小：INT8 < FP32，INT4 打包约 1/8 FP32

✅ adv01_quantization 通过
```

---

## 7. 下一步

adv01 展示了**权重**的低比特压缩。推理系统还有另一个常被忽视的优化维度——**采样策略**。

→ **adv02 采样进阶**：Temperature / Top-K / Top-P / Repetition Penalty 的实现与对比，以及如何在 vLLM 的 `SamplingParams` 中组合使用。
