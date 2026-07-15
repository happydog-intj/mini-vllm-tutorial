# mini-vllm-tutorial 进阶系列

主系列 step01–20 覆盖了 LLM 推理引擎的核心路径（Tokenizer → Attention → KV Cache → Continuous Batching → PagedAttention → FlashAttention → CUDA Graph → Tensor Parallel → 真实模型接入）。然而，工业级推理服务仍有 16 项关键优化手段未讲。本进阶系列在主系列基础上补齐这些内容，覆盖精度控制、解码加速、缓存结构、大规模并行、分离式架构、模型架构变体与服务工程等方向。

## 前置

建议先学完主系列 step01–20，再进入进阶系列。

## 学习路线

```
Phase A — 精度与采样进阶
  adv01_quantization        ← W4A16/W8A16 权重量化
  adv02_sampling_advanced   ← MinP / 惩罚项 / Beam Search

Phase B — 解码加速
  adv03_speculative_decoding ← 投机解码（草稿+验证）
  adv04_flash_decoding       ← 长序列 decode 切分（+FlashInfer 简介）

Phase C — 缓存结构进阶
  adv05_radix_prefix_cache   ← Radix Attention + Copy-on-Write

Phase D — 并行进阶（承接主系列 step17 TP）
  adv06_pipeline_parallel    ← Pipeline Parallel (PP, 微批次/1F1B)
  adv07_sequence_parallel    ← Sequence Parallel (SP)
  adv08_data_parallel_dplb   ← Data Parallel + DPLB 负载均衡
  adv09_tbo_dbo_overlap      ← TBO/DBO 计算-通信重叠

Phase E — 分离式架构
  adv10_pd_disaggregation    ← Prefill/Decode 分离
  adv11_afd_attention_ffn    ← Attention/FFN 分离

Phase F — 模型架构变体
  adv12_moe_eplb             ← MoE Top-k 路由 + EPLB
  adv13_linear_attention     ← 线性注意力 / SSM

Phase G — 服务与输出控制
  adv14_multi_lora           ← Multi-LoRA 动态切换
  adv15_guided_decoder       ← JSON/regex 结构化输出
  adv16_function_call        ← Function Call / Tool Call
```

## 学习方式

每步都是独立可运行的代码，沿用主系列风格：

1. 先读当步 `README.md`（含 ASCII 图解与原理说明）
2. 运行 `python run.py` 看效果
3. 阅读核心实现文件，理解代码
4. 对比相邻两步的 diff：`diff ../advXX_xxx/engine.py engine.py`
