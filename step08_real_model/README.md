# Step 08: Real Model — Qwen3ForCausalLM

## 本节目标

从零实现 Qwen3 模型结构，并从 HuggingFace safetensors 权重文件加载参数，完成真实推理。

## 核心概念

### RMSNorm
比 LayerNorm 更轻量的归一化方法，去掉了均值中心化步骤：

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

### RoPE（旋转位置编码）
将位置信息编码为旋转矩阵，作用于 Q/K：

```
apply_rope(x, cos, sin) = [x1*cos - x2*sin, x1*sin + x2*cos]
```

### GQA（分组查询注意力）
Qwen3 使用 GQA，KV head 数量少于 Q head 数量，推理时用 `repeat_interleave` 展开：

```
num_groups = num_heads // num_kv_heads
k_expanded = k.repeat_interleave(num_groups, dim=1)
```

### safetensors 加载
支持单文件和分片两种格式：
- `model.safetensors` — 小模型单文件
- `model.safetensors.index.json` + 多个分片 — 大模型分片

## 文件说明

| 文件 | 功能 |
|------|------|
| `model.py` | Qwen3ForCausalLM 完整实现 |
| `loader.py` | safetensors 权重加载器 |
| `engine.py` | 封装模型+tokenizer 的推理引擎 |
| `run.py` | 测试入口，自动跳过无模型环境 |

## 运行

```bash
# 设置模型路径（默认 ~/huggingface/Qwen3-0.6B）
export QWEN3_MODEL_PATH=/path/to/Qwen3-0.6B

python run.py
```

## 关键设计

- `past_key_values`：KV Cache，decode 阶段每步只处理 1 个 token
- `positions`：显式传入位置索引，支持 prefill 和 decode 两种模式
- `torch.bfloat16`：推理精度，平衡显存与数值稳定性
