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

### ❓ Q1：`qmax` 的公式为什么是 `(1 << (bits - 1)) - 1`？丢了 `-8` 这个值？

**问题**：`bits=4` 时 `qmax=7`，范围是 `[-7, 7]` 共 15 个值。但 4-bit 有符号整数能表示 16 个值 `[-8, 7]`，为什么丢掉 `-8`？

**答案**：这是**对称量化（symmetric quantization）**的取舍。对称量化的核心约束是量化范围关于零点对称：`[-qmax, +qmax]`。

- INT8（8-bit signed）：理论范围 `[-128, 127]`，对称范围取 `[-127, 127]`（舍掉 -128）
- INT4（4-bit signed）：理论范围 `[-8, 7]`，对称范围取 `[-7, 7]`（舍掉 -8）

舍掉一个值（INT8 的 -128，INT4 的 -8）是为了保证**零值精确映射为零**——`quantize(0) = 0`。如果范围不对称（比如 `[-8, 7]`），零点可能偏移，推理时 bias 需要额外补偿。

**非对称量化（asymmetric quantization）**确实能利用全部 16 个值，公式变为 `q = round((w - zero_point) / scale)`，多一个 `zero_point` 参数。Google 的 TPU 推理用的就是这种方法，但教学版先讲对称的，非对称留给进阶。

### ❓ Q2：一个 outlier 会不会毁掉整个 scale？

**问题**：1000 个权重里 999 个在 `[-0.1, 0.1]`，1 个 outlier 是 `10.0`。`scale = 10/127 ≈ 0.0787`，原来 `0.1` 的值量化后变成 `round(0.1/0.0787) = 1`，反量化回来是 `0.0787`，**相对误差 21%**！

**答案**：你的直觉完全正确。per-tensor 量化对 outlier 极度敏感，这就是为什么**生产环境几乎不用 per-tensor**：

| 量化粒度 | scale 数量 | outlier 敏感度 | 典型场景 |
|---------|-----------|---------------|---------|
| **per-tensor** | 整个层 1 个 | 极高：1 个 outlier 毁所有 | 教学版 |
| **per-channel** | 每行 1 个 | 中：outlier 只影响同行的权重 | vLLM / AWQ 默认 |
| **groupwise** | 每 128 个参数 1 个 | 低：outlier 局限在小组内 | GPTQ、bitsandbytes |

**per-channel 直观效果**：假设权重矩阵每行的 max 不同：
```
行0: max=0.1  → scale=0.1/127  → 该行正常量化
行1: max=10.0 → scale=10/127   → 该行正常量化
```

outlier 只影响它所在的那一行，不会拖垮整个 tensor。AWQ（Activation-aware Weight Quantization）更进一步：先分析激活分布，找出哪些权重通道对激活值影响大，给这些"重要"通道用更大的 scale 保护它们不被 outlier 压缩。

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

### ❓ Q4：为什么 `model.py` 要用 `importlib.import_module` 这么绕的方式？

**问题**：直接写 `from model import TinyTransformerWithKVCache` 不行吗？为什么要动态导入？

**答案**：这是**跨章节代码复用**的技术选择。`adv01` 的模型定义不在本目录，而是复用 `step07_kvcache_for_single_request/model.py`。

```python
# 方法1：直接 import（简单，但路径可能有问题）
from model import TinyTransformerWithKVCache  # ❌ 当前目录没有 model.py

# 方法2：修改 sys.path + import（常用）
sys.path.insert(0, '../../step07_kvcache_for_single_request')
from model import TinyTransformerWithKVCache  # ✅ 能工作

# 方法3：importlib 动态导入（当前方式，最灵活）
_path = os.path.join(os.path.dirname(__file__), '..', '..', 'step07_kvcache_for_single_request')
sys.path.insert(0, os.path.abspath(_path))
_mod = importlib.import_module('model')  # ✅ 等价于方法2，但更灵活
```

方法2和方法3在功能上等价。选择 importlib 的原因：

1. **避免命名冲突**：如果当前目录恰好也有 `model.py`，`from model import ...` 会导入本地文件（Python 先搜索当前目录）。importlib 通过绝对路径确保导入的是 step07 的 model。
2. **延迟加载**：importlib 可以在运行时条件判断后再导入（虽然本例没用到）。
3. **热重载潜力**：可以用 `importlib.reload(_mod)` 重新加载修改后的模块，适合交互式开发。

**教学建议**：如果你只是本地跑实验，方法2（sys.path + from import）完全够用。importlib 更"工程化"，但多一层间接性增加了理解成本。

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

### ❓ Q3：`register_buffer` 和 `nn.Parameter` 到底有什么区别？

**问题**：如果错误地用 `self.q = nn.Parameter(q)` 会怎样？

**答案**：关键区别在于 **optimizer 是否尝试更新它**：

| | `register_buffer` | `nn.Parameter` |
|---|---|---|
| 出现在 `model.parameters()` | ❌ 否 | ✅ 是 |
| optimizer 会更新 | ❌ 否 | ✅ 是 |
| `.to(device)` 跟随模型 | ✅ 是 | ✅ 是 |
| 参与反向传播 | ❌ 否 | ✅ 是 |

如果写成 `nn.Parameter(q)`，optimizer 会尝试用梯度更新这个 int8 tensor：
```python
# optimizer.step() 会做：param -= lr * param.grad
self.q.data -= 0.001 * self.q.grad  # 但 q 是 int8，梯度是 float32
```

这会导致 **类型不兼容错误**——int8 tensor 不能直接加减 float32 梯度。即使你强行 `.float()` 再转回来，训练过程中整型值也会溢出 `[-127, 127]` 范围。

**更深层的原因**：量化后的权重已经**丢失了精度信息**。反量化得到的是 `w_approx = q * scale ≈ w`，这个近似值不是原始的 FP32 权重。如果在这个近似值上反向传播，梯度方向可能完全错误——量化和训练需要联合优化（如 QAT, Quantization-Aware Training），不是简单换一下存储格式就行。

### ❓ Q5：bias 不量化真的是因为"参数量少"吗？

**问题**：教学模型 `in_features=4, out_features=4`，bias 只有 4 个参数，权重 16 个，bias 占比 20%。这时候不量化还合理吗？

**答案**：即使 bias 占比不小，仍有三个理由不量化：

**1. 量化 bias 的计算开销不划算**

```python
# 量化 bias 的 forward 需要：
bias_fp32 = q_bias.to(torch.float32) * bias_scale  # 额外反量化操作
out = x @ w.t() + bias_fp32
```

bias 反量化本身的计算量虽然小，但每个 token 每次 forward 都要做。权重反量化是 `O(d_model²)` 的矩阵操作，反量化开销可忽略；但 bias 只有 `O(d_model)`，反量化开销相对不可忽略。

**2. bias 的数值分布特殊**

权重的值域通常比较集中（初始化时接近 0），但 bias 可能有任意大小——比如 LayerNorm 的 bias 初始为 0，但某些层（如 embedding 的 bias）初始值可能较大。给 bias 做量化需要单独校准 scale，增加了校准复杂度。

**3. 实际占比计算**

| 模型 | 权重参数量 | bias 参数量 | bias 占比 |
|------|-----------|------------|----------|
| 教学（4×4） | 16 | 4 | 20% |
| GPT-2 small (768×768) | 589,824 | 768 | 0.13% |
| LLaMA-70B (4096×4096) | 16,777,216 | 4096 | 0.024% |

教学模型确实 bias 占比大，但**教学版的目的是演示核心方法（权重量化）**，bias 是边情况。等理解了权重量化，bias 量化是顺手的事——读者可以自己练习加上。

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

### ❓ Q6：`list(model.named_children())` 为什么要包一层 `list()`？

**问题**：不包 `list()` 直接遍历会怎样？是因为遍历时修改模型结构导致迭代器失效吗？

**答案**：这是 Python 的**经典陷阱**。`named_children()` 返回的是**惰性迭代器（iterator）**，不是列表。迭代器的行为是：每次 `next()` 时才从模型中取下一个子模块。

```python
# 不包 list()：
for name, mod in model.named_children():   # 迭代器
    setattr(model, name, new_mod)          # 修改了模型结构！
    # 下一轮 next() 时，模型的子模块列表已经变了，迭代器可能跳过或重复
```

当你 `setattr` 替换一个子模块后，模型的内部 `_modules` 字典被修改了。迭代器依赖这个字典，修改后行为**未定义**——可能跳过某些模块，也可能重复访问。

```python
# 包 list()：
for name, mod in list(model.named_children()):  # 先取出一份快照列表
    setattr(model, name, new_mod)              # 修改不影响已取出的列表
    # 安全：遍历的是快照，不受 setattr 影响
```

**类比理解**：就像遍历列表时删除元素——`for x in my_list: my_list.remove(x)` 会跳过元素。`list()` 就是先做一份快照。

**额外注意**：这只能替换**直接子模块**的属性名。如果某个 Linear 被深埋在 `model.encoder.layers[3].self_attn.q_proj`，`named_children()` 只能看到 `encoder` 这一层，需要通过递归深入到每个嵌套模块——这就是代码里 `quantize_model(mod, bits)` 递归调用的原因。

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

### ❓ Q7：INT4 用 int8 容器存，和 INT8 占的字节一样，图啥？

**问题**：教学版 INT4 和 INT8 都按 `numel() * 1` 计算，存储大小一模一样。不打包的话，INT4 除了误差更大，还有什么好处？

**答案**：教学版唯一的好处是**教学**——让你在不写 bit 打包代码的前提下，直观对比 INT4 比 INT8 误差更大。真实收益必须等打包后才出现：

```
FP32:  4 bytes/elem
INT8:  1 byte/elem   → 省 3/4
INT4:  1 byte/elem（教学版，未打包） → 省 3/4（和 INT8 一样，没有额外好处）
INT4:  0.5 byte/elem（生产打包，2个INT4塞进1字节）→ 省 7/8
```

**生产打包示例**：两个 4-bit 值 `a` 和 `b`（各占 [-7,7]）塞进一个 uint8：

```python
# 打包：packed = (a & 0x0F) | ((b & 0x0F) << 4)
# 解包：a = packed & 0x0F;  b = (packed >> 4) & 0x0F
```

bitsandbytes、AutoAWQ 等库就是这么做的。**不打包的 INT4 在生产中没有任何价值**，教学版只是避免 bit 操作分散注意力。

### ❓ Q10：W4A16 只是显存省了，计算并没有加速？

**问题**：每次 forward 都要反量化回 FP32 再 matmul，计算量没减少。量化到底加速了啥？

**答案**：教学版（反量化后 matmul）的收益**仅有显存节省，没有计算加速**。真实的 INT4/INT8 推理加速来自两个层面：

**1. 显存带宽（Memory Bandbound）是 GPU 推理的真正瓶颈**

大模型推理时，GPU 算力（FLOPS）通常跑不满——**权重从显存读到 SRAM 的速度才是瓶颈**。读 17.5 GB（INT4 70B）比读 140 GB（FP16 70B）快 8 倍，GPU 等待权重的时间大幅缩短。即使计算时仍用 FP16 matmul，**吞吐量也受限于带宽而非算力**。

```
FP16 70B:  140 GB / 900 GB/s ≈ 155ms（只读权重就要这么久）
INT4 70B:   17.5 GB / 900 GB/s ≈ 19ms
```

**2. 真正的 INT4 GEMM 需要 CUDA kernel 直接在整数空间计算**

NVIDIA 的 Tensor Cores 支持 `DP4A` 指令（4-bit 乘加），cutlass 库的 Marlin kernel 专门优化了 INT4 GEMM。这种 kernel 读入 INT4 数据后**不做反量化**，直接在 INT4 空间做矩阵乘法，最后再把累加结果转回 FP16。计算量相同（仍然 N³ 次乘加），但**每次乘加的位宽从 16-bit 降到 4-bit**，吞吐提升 2-4 倍。

**总结**：
- 教学版（反量化 matmul）= 仅省显存，不加速计算
- 生产版（INT4 GEMM kernel）= 省显存 + 带宽减半 + 计算加速


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

### ❓ Q8：相对误差用 `max|a-b| / max|a|` 合理吗？

**问题**：如果 FP32 logits 里有一个很大的值 `100.0` 和一个接近零的值 `0.001`，INT8 把 `0.001` 变成了 `0.1`（差了两个数量级），但因为分母是 `max|a| = 100`，这个局部大误差被淹没了。有没有场景下相对误差很小但实际生成结果完全变了？

**答案**：**有的。** 这个指标叫 **全局无穷范数相对误差（global ∞-norm relative error）**，它衡量的是"整体最大偏差相对于整体最大值的比例"。它的盲区是：

```python
# 场景：logits 分布极度不均
fp32_logits  = [100.0, 0.001, 0.002, -0.001, ...]   # max|a| = 100
int4_logits  = [100.1, 0.1,    0.1,   -0.1,    ...]   # 小值全变了

# 全局相对误差：max|diff| / max|fp32| = 0.1 / 100 = 0.001（0.1%，很小！）
# 但小值的相对误差：|0.001 - 0.1| / |0.001| = 99（9900%！）
```

**为什么实际生成可能变？** softmax 函数对 logits 的**相对大小**敏感，不是绝对大小：

```python
# FP32: 小值 logits 经过 softmax 后的概率
softmax([100.0, 0.001, 0.002]) ≈ [1.0, ~0, ~0]  # token 0 几乎 100%

# INT4: 小值被扰动了
softmax([100.0, 0.1, 0.1]) ≈ [1.0, ~0, ~0]       # 结果可能一样（token 0 仍占优）

# 但如果大值也接近：
softmax([1.0, 0.9, 0.8])  → [0.35, 0.32, 0.29]   # 三个 token 概率接近
softmax([1.0, 1.1, 0.6])  → [0.34, 0.38, 0.23]   # INT4 扰动后，top-1 变了！
```

**所以当 logits 分布平坦（多个 token 概率接近）时，小的量化误差就可能改变 top-1 token 的选择。** 这解释了为什么有些模型量化后"看似误差小，但生成质量降得多"。

**更好的指标**：
- **Cosine similarity**：衡量整体方向一致性，对小值不敏感
- **Top-K accuracy**：直接比较 top-K token 是否一致
- **Perplexity (PPL)**：在验证集上跑真实生成任务

教学版用全局相对误差是因为**计算简单、结果直观**——但它确实是粗糙的指标。

### ❓ Q9：`assert err_int8 < 1e-2` 这个阈值是拍脑袋定的吗？

**问题**：1% 的相对误差在 `d_model=4, vocab_size=256` 的玩具模型上成立，换到真实模型还成立吗？

**答案**：**是的，这个阈值是教学经验值**，不是理论推导出来的。但它有几个依据：

**1. INT8 的理论量化误差上界**

对于对称均匀量化，最大绝对误差是 `scale / 2`（半个量化步长）。当 `qmax = 127` 时，scale = max|w| / 127，所以最大相对误差约 `1/(2×127) ≈ 0.4%`。这解释了为什么 INT8 阈值设在 1%——理论上限是 0.4%，留了 2.5× 余量。

**2. INT4 为什么是 15%？**

INT4 的 `qmax = 7`，理论最大相对误差约 `1/(2×7) ≈ 7%`。但因为：
- 教学模型 `d_model=4` 太小，权重分布不稳定
- round+clamp 的非线性误差叠加
- 多层变换后误差累积

所以阈值放宽到 15%。

**3. 真实模型上的表现**

| 模型 | INT8 误差 | INT4 误差（AWQ） | INT4 误差（GPTQ） |
|------|----------|-----------------|------------------|
| LLaMA-2-7B | ~0.3% | ~1-3% | ~1-2% |
| LLaMA-2-70B | ~0.2% | ~2-5% | ~1-3% |

真实模型上 INT8 误差通常 < 0.5%，因为权重大小分布更稳定（经过预训练收敛）。INT4 误差取决于校准方法——AWQ/GPTQ 用激活感知或 Hessian 优化，比直接 round+clamp 好 3-5 倍。

**结论**：教学版的阈值是合理的保守估计。真实模型上 INT8 几乎总是 < 1%，但 INT4 如果没有 AWQ/GPTQ 校准，误差可能达到 10-20%——这也是为什么生产上 INT4 必须配合校准方法使用。

---

## 7. 下一步

adv01 展示了**权重**的低比特压缩。推理系统还有另一个常被忽视的优化维度——**采样策略**。

→ **adv02 采样进阶**：Temperature / Top-K / Top-P / Repetition Penalty 的实现与对比，以及如何在 vLLM 的 `SamplingParams` 中组合使用。
