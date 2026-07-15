# 进阶系列设计:mini-vllm-tutorial `advanced/` 子系列

- **日期**: 2026-07-15
- **状态**: 设计已确认,待写实现计划
- **背景**: 主系列(step01–20)对照知乎《大模型推理核心概念与术语总结》分析后,有 16 项推理优化手段未覆盖。本设计新增一个进阶子系列补齐。

## 1. 目标

在主系列基础上,新增 16 步进阶教程,覆盖主系列缺失的推理优化技术。风格与主系列一致:每步独立可运行、增量清晰、注释详尽、README 含 ASCII 图解与"教学版 vs 真实框架"对比。尽可能通俗易懂、循序渐进。

## 2. 非目标 (YAGNI)

- 不替换或重写主系列任何现有步骤。
- 不追求生产级性能(仍用 TinyTransformer 教学,模型小、CPU 可跑)。
- 不实现真正的多机分布式训练/推理框架,分布式类步骤用单机模拟器讲清原理。

## 3. 目录与编号

- 新建 `advanced/` 目录,内部从 `adv01` 起独立编号,与主系列物理隔离。
- 自带 `advanced/README.md`:学习路线 + Phase 分组 + 与主系列的衔接说明。
- 顶层 `README.md` 末尾追加一行指针:`→ 进阶系列见 advanced/README.md`。
- base model 复用:`advanced/advXX/model.py` 用 importlib 从 `../step07_kvcache_for_single_request` 加载 `TinyTransformerWithKVCache`,沿用主系列 `step12_paged_attention/scheduler.py` 的 shim 写法,避免循环导入。

## 4. 16 步清单

代码深度:🟢 = 真代码可跑;🟡 = 模拟器/图解(单机 CPU 难以真实实现,用 Python 模拟讲清原理)。

### Phase A — 精度与采样进阶
- `adv01` 量化 Quantization (W4A16 / W8A16) 🟢
  - weight-only 量化,在 TinyTransformer 上做 INT8/INT4 权重量化+反量化,测显存占用与速度。
- `adv02` 采样进阶 (MinP / Frequency·Presence·Repetition Penalty / Beam Search) 🟢
  - 扩展 step06,新增 MinP、三种惩罚项、Beam Search。

### Phase B — 解码加速
- `adv03` 投机解码 Speculative Decoding 🟢
  - 草稿模型(更小 transformer)+ 目标模型验证/接受,演示 decode 2–3×。
- `adv04` Flash-Decoding (+ FlashInfer 简介) 🟡
  - 长序列 decode 的 split-K 切分原理演示,简介 FlashInfer 库定位。

### Phase C — 缓存结构进阶
- `adv05` Radix Attention + Copy-on-Write 🟢
  - 基数树组织 KV cache 实现自动前缀复用 + 写时复制,对比主系列 hash+ref_count 方案。

### Phase D — 并行进阶(承接主系列 step17 TP)
- `adv06` Pipeline Parallel (PP, 微批次 / 1F1B) 🟡
- `adv07` Sequence Parallel (SP, 序列维切分) 🟡
- `adv08` Data Parallel + DPLB (多副本 + 负载均衡) 🟢(Python 调度模拟)
- `adv09` TBO/DBO (计算-通信重叠) 🟡

### Phase E — 分离式架构
- `adv10` PD Disaggregation (Prefill/Decode 分离) 🟡
- `adv11` AFD (Attention-FFN 分离) 🟡

### Phase F — 模型架构变体
- `adv12` MoE + EPLB (Top-k 路由 + 专家负载均衡) 🟢(小 MoE 层)
- `adv13` Linear Attention / SSM (线性注意力) 🟢

### Phase G — 服务与输出控制
- `adv14` Multi-LoRA (多适配器动态切换) 🟢
- `adv15` Guided Decoder (JSON / regex 结构化输出) 🟢
- `adv16` Function Call / Tool Call 🟢(基于 adv15)

## 5. 每步文件结构(与主系列一致)

```
advanced/advXX_name/
  README.md   ← 教学目标 / 问题 / 原理(ASCII 图)/ 实现细节 / 真实框架对比 / 运行 / 下一步
  run.py      ← 可独立运行,结尾打印 "✅ advXX_name 通过"
  *.py        ← 核心实现(model.py 多为复用 step07 的 shim)
```

README 固定含"教学版做法"vs"真实 vLLM/SGLang 做法"对比段落,延续主系列风格。

## 6. 排序理由

从"可直接在现有模型上跑的改写"(量化、采样)起步 → 解码加速 → 缓存结构 → 并行(承接 TP)→ 架构分离 → 模型变体 → 服务层。每步只引入一个新概念,前置依赖都已在前面铺好。

## 7. 依赖与运行环境

- 主系列 CPU 环境(torch)足以跑完所有 🟢 步骤。
- 🟡 模拟器步骤只用 torch + 标准库,不引入真分布式依赖。
- 不新增 `requirements.txt`(全部基于现有 torch)。若后续 adv12/adv13 需额外包,届时再加并在此更新。

## 8. 顶层改动

- 新增 `advanced/` 整个目录(16 个 step 子目录 + 1 个 `advanced/README.md`)。
- 顶层 `README.md` 末尾追加进阶系列指针(一行)。
- 顶层 `SUMMARY.md` 可选追加"进阶系列索引"小节(实现完成后再补)。

## 9. 验收标准

- 每个 advXX 目录 `python run.py` 能跑通并打印通过标记。
- 每个 README 含:教学目标、ASCII 原理图、教学版 vs 真实框架对比、运行说明。
- 不破坏主系列现有步骤(重命名后 step07/08/09 importlib 链仍 OK)。
- 顶层 README 含进阶系列指针。

## 10. 风险与备注

- 🟡 模拟器步骤的"速度提升"是模拟出来的,README 需诚实标注为原理演示而非真实加速。
- adv10/adv11 的 KV/激活跨"设备"传递用模拟时延注入,不跨真进程(除非后续决定升级为多进程)。
- adv03 投机解码的草稿模型可用更小 transformer 或同模型浅层近似,需在 README 说明取舍。
