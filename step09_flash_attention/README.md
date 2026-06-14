# Step 09: FlashAttention 封装

## 本节目标

封装 FlashAttention，在 CUDA GPU 上使用 flash_attn 加速注意力计算，CPU/MPS 自动回退到 PyTorch SDPA。

## 核心概念

### 标准注意力的内存瓶颈

标准注意力需要存储完整的 `seq_len × seq_len` attention 矩阵：

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

对于序列长度 4096，这个矩阵是 4096×4096 = 16M 个元素，严重占用显存带宽。

### FlashAttention 原理

FlashAttention 通过分块（tiling）计算，避免将完整 attention 矩阵写入 HBM：

1. 将 Q/K/V 分成小块，逐块计算
2. 用在线 softmax（online softmax）保证数值等价
3. 内存复杂度从 O(N²) 降至 O(N)

**速度提升**：2-4x（取决于序列长度和 GPU 型号）

### PyTorch SDPA 作为回退

`torch.nn.functional.scaled_dot_product_attention` 在 PyTorch 2.0+ 内置多种优化后端：
- CUDA：自动选择 FlashAttention / memory-efficient attention
- CPU：标准实现

## 文件说明

| 文件 | 功能 |
|------|------|
| `attention.py` | FlashAttention 封装 + SDPA 回退 |
| `run.py` | 正确性验证（max_diff < 0.02） |

## 运行

```bash
python run.py
```

## 注意事项

- `flash_attn` 仅支持 CUDA，需单独安装：`pip install flash-attn`
- 输入形状：`[batch, num_heads, seq_len, head_dim]`（与 flash_attn 期望的不同，内部做了 transpose）
- bfloat16 / float16 均支持，float32 不支持 FlashAttention
