# mini-vllm-tutorial

从零实现 LLM 推理引擎：15步学懂 vLLM 核心原理

## 学习路线

```
Phase 0 — 基础概念（CPU，只需 torch）
  step01_tokenizer    ← Token 是什么，BPE 编码
  step02_embedding    ← 向量空间，词表查表
  step03_attention    ← 手写 Scaled Dot-Product Attention
  step04_transformer  ← 完整 Transformer Decoder 层

Phase 1 — 朴素推理
  step05_naive         ← 自回归生成，O(n²) 问题展示

Phase 2 — 采样算法
  step06_sampler       ← Greedy / Temperature / Top-k / Top-p / Gumbel-Max

Phase 3 — KV Cache
  step07_kvcache_single ← 单请求 KV Cache，O(n²) → O(n)
  step08_kvcache_batch  ← 多请求 Batch + Padding 问题

Phase 4 — 调度器
  step09_scheduler       ← Continuous Batching
  step10_chunked_prefill ← 长 Prompt 分块
  step11_preemption     ← 抢占避免 OOM

Phase 5 — PagedAttention
  step12_paged_attention ← 分页内存，利用率 17% → 96%
  step13_prefix_cache    ← 前缀缓存，节省 77% 计算

Phase 6 — 真实模型（需要 GPU 推荐）
  step14_paged_prefix_cache ← 分页前缀缓存（待实现）
  step15_real_model      ← 接入 Qwen3-0.6B

Phase 7 — 高级优化
  step16_flash_attention ← IO-aware 分块注意力
  step17_cuda_graph      ← CUDA Graph 录制重放
  step18_tensor_parallel ← 多 GPU Tensor 并行

Phase 8 — 工程化
  step19_benchmark       ← 吞吐量/延迟测量
  step20_serve           ← OpenAI 兼容 HTTP 服务
```

## 快速开始

```bash
# 安装依赖（CPU 版，前 8 步够用）
pip install -r requirements-cpu.txt

# 从第一步开始
cd step01_tokenizer
python run.py
```

## 学习方式

每步都是独立可运行的代码。学习方法：

1. 先读 `README.md`（含 ASCII 图解）
2. 运行 `python run.py` 看效果
3. 阅读核心实现文件，理解代码
4. 对比相邻两步的 diff：`diff ../stepXX/engine.py engine.py`

## 与 nano-vllm 的关系

本项目是 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的教学版本：
- nano-vllm：生产就绪，~1400 tok/s，代码精简但跳跃
- mini-vllm-tutorial：教学优先，每步增量清晰，注释详尽

## 总结

学完所有步骤后，可以查看 [SUMMARY.md](SUMMARY.md)——汇总了推理服务的关键配置、监控指标和优化方法。
