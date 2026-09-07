---
layout: home

hero:
  name: '从零实现 LLM 推理引擎'
  text: '20步学懂 vLLM 核心原理'
  tagline: 从 Tokenizer 到 PagedAttention 到 HTTP 服务——每步可运行的代码 + ASCII 图解
  actions:
    - theme: brand
      text: 开始阅读 →
      link: /step01_tokenizer/
    - theme: alt
      text: GitHub ⭐
      link: https://github.com/happydog-intj/mini-vllm-tutorial

features:
  - icon: 🧠
    title: 核心公式
    details: 'LLM 推理引擎 = Tokenizer + Attention + KV Cache + Scheduler + PagedAttention'
  - icon: 🎯
    title: 面向实践
    details: 每章解决一个具体问题，配套可运行的 Python 代码
  - icon: 📐
    title: 20 步 + 17 进阶
    details: 主系列 20 步覆盖核心路径，进阶系列 17 章覆盖量化、投机解码、MoE、分布式等
  - icon: 🚀
    title: 从 CPU 到 GPU
    details: 前 8 步只需 CPU + PyTorch，后续逐步引入 GPU、FlashAttention、CUDA Graph、Tensor Parallel
---

## 为什么写这个教程？

vLLM 是目前最主流的 LLM 推理引擎之一，但它的代码量超过 10 万行，直接阅读源码门槛很高。

这个教程用 **20 步**把 vLLM 的核心原理拆解成可独立运行的最小实现：每步只加一个概念，代码量控制在几百行以内，配合 ASCII 图解和对比实验，让你真正理解每个优化**为什么有效**。

## 学习路线

| 阶段 | 章节 | 核心问题 |
|------|------|----------|
| **基础概念** | step01–04 | Token → Embedding → Attention → Transformer |
| **朴素推理** | step05–06 | 自回归生成 + 采样策略 |
| **KV Cache** | step07–08 | O(n²) → O(n)，Static Batching |
| **调度** | step09–11 | Continuous Batching → Chunked Prefill → Preemption |
| **PagedAttention** | step12–14 | 分页内存 + 前缀缓存 |
| **高性能内核** | step15–16 | FlashAttention + CUDA Graph |
| **分布式** | step17 | Tensor Parallelism |
| **工程落地** | step18–20 | Benchmark + Real Model + HTTP 服务 |

## 快速开始

```bash
# 安装依赖（CPU 版，前 8 步够用）
pip install -r requirements-cpu.txt

# 从第一步开始
cd step01_tokenizer
python run.py
```

## 与 nano-vllm 的关系

本项目是 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的教学版本：
- **nano-vllm**：生产就绪，~1400 tok/s，代码精简但跳跃
- **mini-vllm-tutorial**：教学优先，每步增量清晰，注释详尽
