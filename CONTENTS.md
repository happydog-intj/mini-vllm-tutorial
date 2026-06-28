# 目录 — mini-vllm-tutorial

每章说明：**要解决的问题** → **解决方案** → **直观类比**

---

## 为什么需要 LLM 推理引擎？

ChatGPT 让数亿人体验到了大语言模型的能力，但它背后的工程挑战鲜少被讨论：
**一个模型参数量 700 亿、每次对话都要生成几百个 token 的系统，如何同时服务成千上万个用户？**

这不是一个简单的"把模型跑起来"的问题。LLM 推理有几个根本性的特殊之处，
使得通用深度学习推理系统（如早期的 Triton Inference Server）完全不够用：

### LLM 推理的三个根本特征

**1. 自回归生成：每个 token 依赖前一个**

```
输入:  "今天天气"
步骤1: 模型 → "很"       （把"很"加入序列，重新输入）
步骤2: 模型 → "好"       （把"好"加入序列，重新输入）
步骤3: 模型 → "。"       （结束）

生成过程是严格串行的——必须先有第 N 个 token，才能算第 N+1 个
```

这意味着无法像图片分类那样"一次前向传播完成推理"。
每个请求是一个持续进行的、长度不可预知的循环过程。

**2. 计算量和内存需求都随序列长度增长**

```
Prefill（处理 prompt）：计算量 O(n²)，n 是 prompt 长度
Decode（逐步生成）：每步 O(n)，但需要缓存所有历史 K/V 向量（KV Cache）

生成 1000 个 token 的请求，KV Cache 占用约：
  1000 token × 28层 × 2(K+V) × 8头 × 64维 × 2字节 = 57 MB（Qwen3-0.6B）
  1000 token × 80层 × 2(K+V) × 8头 × 128维 × 2字节 = 327 MB（LLaMA-70B）
```

显存是有限的——系统必须精心管理每一块显存，才能支撑足够多的并发请求。

**3. 请求的生成长度在开始时完全未知**

用户问一句话，模型可能回答 10 个字，也可能回答 2000 个字。
这个不确定性让所有"提前分配资源"的策略都面临困境：
分配少了会 OOM，分配多了大量浪费。

---

### 从朴素实现到 vLLM：五个核心问题

把一个 LLM 从"能跑"做到"能高效服务大量用户"，需要依次解决五个问题。
每个问题在操作系统设计里都有对应的经典解法——LLM 推理系统的演进，本质上是在 GPU 上重走了一遍 OS 走过的路：

| # | LLM 推理问题 | 解决方案 | OS 类比 |
|---|------------|---------|--------|
| 1 | 每步重算整个历史序列，O(n²) 计算量 | **KV Cache**：缓存已算的 K/V，每步只算新 token | CPU 寄存器 / L1 Cache：已算的中间结果存起来，不重算 |
| 2 | 一个请求完成，GPU 槽位空转等其他请求 | **Continuous Batching**：完成即释放，新请求随时补入 | 分时操作系统：进程完成即释放 CPU，就绪队列立刻调度下一个 |
| 3 | 长 Prefill 独占 GPU，Decode 请求"卡住" | **Chunked Prefill**：prefill 切成小块，每步只用一个 chunk | 时间片抢占：CPU 密集型进程不能无限占 CPU，用完时间片强制切换 |
| 4 | 预留 max_len 连续显存，碎片无法共用，利用率 ~60% | **PagedAttention**：固定大小 Block 按需分配，block_table 翻译地址 | 虚拟内存分页：page table 把虚拟地址映射到散落的物理页帧，消除外部碎片 |
| 5 | QK^T 中间矩阵反复在 HBM/SRAM 间搬运，带宽成瓶颈 | **FlashAttention**：分块计算，中间结果留在 SRAM，HBM 读写从 O(n²) 降到 O(n) | Cache-blocking 分块矩阵乘法：把小块数据装进 L1 Cache，在 Cache 内算完再写回，减少主存访问 |


---

### 设计的演进逻辑

这五个问题不是独立的——后一个问题往往是解决前一个问题时暴露出来的新瓶颈：

```
朴素推理（朴素自回归推理）
  └→ 问题：每步重算全部历史
      └→ KV Cache（多请求 KV Cache + Static Batching）：缓存历史 K/V
          └→ 新问题：多请求时 GPU 槽位空转
              └→ Continuous Batching（Continuous Batching 调度器）：动态调度请求
                  └→ 新问题：长 Prefill 阻塞 Decode
                      └→ Chunked Prefill（Chunked Prefill：切片长 Prompt）：切片执行
                          └→ 新问题：KV Cache 显存碎片、OOM
                              ├→ Preemption（Preemption：抢占避免 OOM）：优雅降级
                              └→ PagedAttention（PagedAttention：分页内存管理）：分页管理
                                  └→ 新问题：注意力计算带宽成瓶颈
                                      └→ FlashAttention（FlashAttention：SRAM-aware 注意力计算）：SRAM 分块
                                          └→ 工程化：多卡、服务化...
```

**每一步都是对前一步遗留问题的精准回应。** 这条演进路径不是事后诸葛亮——
它是 2017-2023 年间学术界和工业界真实走过的路，
从原始 Transformer 论文到 Orca、vLLM、FlashAttention 的发表，每篇论文都在解决上一个系统暴露的新瓶颈。

本教程按照这条历史演进路径组织，每一步都先展示当前系统的问题，再引入解决方案。
学完全部步骤，你不只是学会了 vLLM 的代码，而是理解了**为什么**它会被这样设计。

---



## Phase 0 — 基础组件

> 从零搭积木。Transformer 由四个基础组件组成，理解它们才能看懂后面每一步的优化。

---

### step01 — Tokenizer：文字变数字

**问题：** LLM 是数学函数，只能处理整数，但输入是自然语言文字。如何把文字转成数字？

**方案：** BPE（字节对编码）——把高频字符组合合并成一个 token，词表大小约 15 万。"你好世界" → `[15267, 1207, 110008]`

**类比：** 摩尔斯电码表。每个字母对应一串点划，查表翻译，双向可逆。不同的是 BPE 的"码表"是从语料中统计出来的，频率越高的组合越可能成为一个独立条目。

---

### step02 — Embedding：数字变向量

**问题：** Token ID 是整数（如 42、15267），整数之间没有语义关系——42 和 43 不比 42 和 10000 更"相近"。网络需要有语义的连续表示。

**方案：** Embedding 矩阵——一个大查找表，每个 token ID 对应一行向量（如 1024 维）。这些向量是训练出来的，语义相近的词向量距离也近。

**类比：** 地图上的坐标。"北京"和"上海"的 ID 可能相差很远，但它们的向量坐标在"城市"这个语义区域里彼此相近，都远离"苹果""跑步"等词的区域。

---

### step03 — Attention：词语间的关联计算

**问题：** 句子中每个词的含义依赖上下文——"苹果很好吃"和"苹果发布新品"里，"苹果"的含义不同。如何让模型动态感知上下文？

**方案：** Scaled Dot-Product Attention——每个词生成 Q（我想找什么）、K（我能提供什么）、V（我的内容）。Q·K^T 算相关度，加权 V 得到融合了上下文的新表示。

**类比：** 图书馆检索。你带着一个问题（Q）来查资料，每本书的封面（K）说明了它的主题，内容（V）是实际信息。相关度高的书被多参考，不相关的书被忽略。

---

### step04 — Transformer Decoder 层：组装完整计算单元

**问题：** 光有 Attention 还不够——还需要 MLP 做非线性变换、残差连接保留原始信息、归一化稳定训练。如何把这些组件组装成一个完整的层？

**方案：** Decoder 层 = RMSNorm → Attention → 残差 → RMSNorm → MLP → 残差。叠加 N 层（如 28 层）形成完整模型。

**类比：** 流水线工厂的一道工序。原材料进来（词向量），经过注意力车间（感知上下文）和加工车间（非线性变换），残差连接保证即使某道工序"失效"，原始材料也能直接传递下去，不会损失信息。

---

## Phase 1 — 朴素推理

> 先让系统跑起来，再暴露问题。

---

### step05 — 朴素自回归推理

**问题：** 如何用 Transformer 生成文字？

**方案：** 自回归——每次把已有序列全部输入模型，预测下一个 token，追加到序列末尾，循环直到生成结束符。

**暴露的问题：** 每生成一个 token，都要对整个历史序列重新计算注意力，计算量是 O(n²)——生成第 100 个 token 时的计算量是生成第 1 个 token 时的 100 倍。

**类比：** 每写一个新字就重新把全文朗读一遍，再决定下一个字是什么。速度随文章长度平方级下降。

---

## Phase 2 — 采样算法

> 不一定每次都选最可能的词——创意来自随机性。

---

### step06 — 采样算法：logits → next_token

**问题：** 模型输出的是每个 token 的概率分布。每次都选最高概率（贪心）会导致输出重复、缺乏创意。如何在合理范围内引入随机性？

**方案：** Temperature（调节分布的"尖锐"程度）+ Top-k（只从概率最高的 k 个里选）+ Top-p / Nucleus（只从累积概率达到 p 的最小集合里选）。

**类比：**
- Temperature=0（纯贪心）：每次点菜都选菜单排第一的，永远没有惊喜。
- Temperature 高：随机翻菜单，可能选到奇怪的菜。
- Top-p=0.9：只在"你今天可能想吃的"那几道菜里随机选，既有随机性又不离谱。

---

## Phase 3 — KV Cache：消除重复计算

> 已经算过的东西，记下来不要重算。

---

### step07 — 单请求 KV Cache

**问题：** 朴素自回归推理 每步都重算全部历史 token 的 K/V，大量重复计算。

**方案：** 把每层每步算出的 K/V 向量缓存下来（past_key_values），下一步只算新 token 的 K/V，再拼接到缓存里。计算量从 O(n²) 降到 O(n)。

**类比：** 做笔记。第一次读完整段落（prefill），把要点记在纸上（KV Cache）。之后每读一个新句子，只需把新笔记加到纸上，不用重读已有笔记。

---

### step08 — 多请求 KV Cache + Static Batching

**问题：** 多个用户同时请求时，逐条处理效率低，GPU 的并行算力被浪费。

**方案：** Static Batching——把多个请求的 prompt padding 到同一长度，拼成一个 batch tensor，一次矩阵乘法处理所有请求。

**暴露的问题：** Batch 内最短请求完成后，它的槽位只能空等最长请求结束，GPU 有大量空转时间。整个 batch 必须作为整体处理——就像班车必须等最后一个乘客才能出发，先到的乘客只能等。

---

## Phase 4 — 调度器：让 GPU 永不空转

> 不要让 GPU 等人，有人完成立刻补进新人。

---

### step09 — Continuous Batching 调度器

**问题：** Static Batching 里，一个请求完成后槽位空转，新请求必须等整批结束才能进入。

**方案：** Continuous Batching——每完成一个 decode 步骤就检查哪些请求完成了，立刻把新请求补进来做 prefill，GPU 槽位从不空转。每个请求用 Sequence 状态机（WAITING → RUNNING → FINISHED）追踪状态。

**类比（OS 进程调度）：** 早期批处理 OS 一批作业全跑完才接受下一批；分时 OS 的就绪队列里永远有备用进程，一个进程完成就立刻调度下一个。Continuous Batching 就是 LLM 推理系统的"分时调度"。

---

### step10 — Chunked Prefill：切片长 Prompt

**问题：** 新请求的超长 prompt（如 4096 token）做 prefill 时会独占 GPU 长达 1.3 秒，期间所有正在 decode 的请求的输出完全停止——用户看到屏幕上的字突然"卡住"。

**方案：** 把长 prefill 切成固定大小的 chunk（如 512 token），每步只处理一个 chunk，剩余时间留给 decode 请求。

**类比（OS 时间片抢占）：** CPU 密集型进程不能无限占用 CPU，OS 给它一个时间片，用完就强制切换给其他进程。`chunk_size` 就是 LLM 调度器给 prefill 的"时间片"，decode 请求不会被长期饿死。

---

### step11 — Preemption：抢占避免 OOM

**问题：** 生成长度在请求开始时未知，KV Cache 随 decode 持续增长，可能在运行中耗尽显存导致整个服务崩溃。

**方案：** 抢占——KV Cache 不足时，选择最晚进入的请求（LIFO），释放其 KV Cache，把它插回等待队列队首，稍后重新 prefill 恢复。

**类比（Linux OOM Killer）：** 内存耗尽时 Linux 不直接崩溃，而是主动 kill 一个进程释放内存，其他进程继续运行——优雅降级而非整体崩溃。Swap to CPU（保留 KV Cache 到内存）对应把脏页换出到磁盘，Recompute（重新 prefill）对应直接丢弃可重建的干净页。

---

## Phase 5 — PagedAttention：消除显存碎片

> 不要提前锁死空间——用多少拿多少，用完还回来。

---

### step12 — PagedAttention：分页内存管理

**问题：** 为每个请求预分配 max_len 大小的连续显存块，实际只用一部分，其余空着却被锁定无法给其他请求用——显存利用率约 60%，严重时降到 10%。

**方案：** 把 KV Cache 切成固定大小的 Block（如 16 个 token/块），按需动态分配，Block 不需要物理连续，通过 block_table 做逻辑地址→物理地址翻译。

**类比（OS 虚拟内存分页）：** 进程看到连续的虚拟地址空间，OS 用页表把它映射到散落的物理页帧。PagedAttention 完全照搬这套设计：block_table 就是页表，Block 就是物理页帧，显存利用率从 60% 提升到 ~96%。

---

### step13 — Prefix Caching：相同前缀只算一次

**问题：** 多个用户请求共享同一段 System Prompt，每次请求都重新计算这段 prompt 的 KV Cache，是纯粹的重复计算。

**方案：** 对内容相同的 Block 只保留一份物理拷贝，多个请求的 block_table 指向同一个物理 Block，通过引用计数管理生命周期。新请求如果前缀已缓存，直接跳过 prefill。

**类比（OS 共享库 / Copy-on-Write）：** libc.so 被所有进程共享，物理内存只有一份，每个进程的页表各有一条映射。System Prompt 的 KV Cache 就是 LLM 推理系统的"共享库"。

---

## Phase 6 — 真实模型

> 换掉玩具，接上真实世界。

---

### step14 — Paged Prefix Cache（待实现）

**问题：** 前缀缓存（Prefix Caching：相同前缀只算一次）在 PagedAttention 框架下如何高效实现？

**方案：** 待实现。

---

### step15 — 真实模型：加载 Qwen3-0.6B

**问题：** 前面所有步骤用的是随机初始化的 TinyTransformer，无法生成有意义的文字。如何接入真实的、从 HuggingFace 下载的模型权重？

**方案：** 加载 safetensors 格式的权重文件，把 HuggingFace 的 key 名称（`model.layers.0.self_attn.q_proj.weight`）映射到自己实现的模型属性结构，处理 BF16 精度。同时实现 Qwen3 特有的组件：RMSNorm、GQA、RoPE。

**核心约束：** `model.py` 里每个 `nn.Linear` 的属性名必须和权重文件的 key 路径精确对应——PyTorch 只看属性名，不看类名，名字写错就是 missing key 报错。

---

## Phase 7 — 高级优化

> 在正确的层级做正确的优化。

---

### step16 — FlashAttention：SRAM-aware 注意力计算

**问题：** 标准注意力每层要把 `QK^T` 中间矩阵（seq=2048 时约 32MB）写回显存，再读回来算 softmax，再写回，再读回乘 V——大量显存带宽浪费，GPU 的算术单元在等数据。

**方案：** 把 Q/K/V 切成小块（tile），每块载入 GPU 片上 SRAM（192KB，速度是显存的 10 倍），在 SRAM 内用 online softmax 完成完整计算，只把最终结果写回显存一次。HBM 读写量从 O(n²) 降到 O(n)。

**类比（CPU 分块矩阵乘法）：** 朴素矩阵乘法每次访问 B 矩阵的一列，cache miss 率极高。分块矩阵乘法把 A/B 的小块都装进 L1 Cache，在 Cache 内完成所有乘加，大幅降低主存访问次数。FlashAttention 就是注意力计算的"Cache-blocked matmul"。

同时提供 `flash_attn_varlen_func` 接口——把变长序列直接拼接（无 padding），用 `cu_seqlens` 数组标记每条序列的边界，让 Continuous Batching 下的混合长度 batch 零浪费。

---

### step17 — CUDA Graph：录制重放，跳过调度层

**问题：** Decode 阶段每步只处理 1 个 token，GPU 计算只需几微秒，但 Python → PyTorch → CUDA driver 的调度链路每步需要几毫秒——GPU 大部分时间在等 CPU 把 kernel 提交过来。

**方案：** 提前录制一张 CUDA Graph（把所有 kernel launch 记录成图），之后每步推理只需 `g.replay()` 直接重放，完全跳过 Python/PyTorch/driver 层。为每种 batch size（1, 2, 4, 8...）各录制一张图。

**类比（io_uring / DMA）：** io_uring 把 100 个 I/O 操作写入 submission queue，一次系统调用批量提交，不再每次都陷入内核。DMA 让 CPU 配置一次传输参数，硬件独立搬运数据，CPU 不再逐字节介入。CUDA Graph 做的是同一件事：CPU 配置一次，GPU 独立重放。

---

### step18 — Tensor Parallelism：多卡分布式推理

**问题：** 大模型（如 70B）的权重超过单卡显存上限；即使装得下，单卡矩阵乘法的计算吞吐也可能不足。

**方案：** 把每层的权重矩阵沿某个维度切分到多张 GPU：
- 列并行（Q/K/V 投影）：每张 GPU 负责一部分 head，各自独立计算，无需通信。
- 行并行（O 投影/MLP）：每张 GPU 算出部分结果，最后一次 `all_reduce` 求和。

整个 Attention 或 MLP 子层只需 1 次 all_reduce，通信量极低。

**类比（分布式矩阵乘法）：** 数学基础和 FlashAttention tiling 完全相同——矩阵乘法的分块可加性 `X@W = X0@W0 + X1@W1`。区别在于 FlashAttention 是同一张 GPU 内跨 SRAM/HBM 层级切分，Tensor Parallelism 是跨多张 GPU 切分。Ring Attention 则是两者的合并，用于超长序列（1M token）的分布式处理。

---

## Phase 8 — 工程化

> 把引擎包装成服务，量化优化效果。

---

### step19 — Benchmark：量化优化效果

**问题：** 做了这么多优化，到底快了多少？不同配置下的吞吐量和延迟是多少？

**方案：** 实现标准化的推理性能测试——固定并发请求数、prompt 长度和生成长度，测量：
- **吞吐量（tokens/s）**：GPU 每秒生成多少 token
- **延迟（TTFT / TPOT）**：首个 token 延迟 / 每个 token 的平均生成时间

**类比：** 装修完房子后的验收报告——不只是"感觉更好了"，而是有具体的数字：水压、电压、隔音分贝值，每项都有对照基准可以比较。

---

### step20 — HTTP Serve：OpenAI 兼容推理服务

**问题：** 推理引擎是 Python 对象，只能被同进程调用。如何让它被任意语言、任意机器、任意应用通过标准接口使用？

**方案：** 用 FastAPI 包装推理引擎，实现 OpenAI `/v1/chat/completions` 接口（包括流式输出 SSE）。任何兼容 OpenAI SDK 的客户端——无论是 Python、JavaScript、curl 还是 ChatGPT 插件——都能直接对接。

**类比：** 把自家厨房（推理引擎）改造成餐厅（HTTP 服务）——菜品（模型能力）没变，但现在有了标准化的点菜方式（REST API），任何人拿着菜单（OpenAI 格式请求）都能下单，不需要亲自进厨房操作。

---

## 纵向主线：一个问题如何在多个层级被解决

同一类问题在不同 phase 被反复解决，每次解决的是更深一层：

```
「重复计算」这个问题：
  单请求 KV Cache  KV Cache         → 缓存已算的 K/V，避免逐 token 重算
  Prefix Caching：相同前缀只算一次   Prefix Caching   → 缓存共享 prefix 的 K/V，避免逐请求重算
  FlashAttention：SRAM-aware 注意力计算   FlashAttention   → 避免 attention score 矩阵写回显存再读回

「内存浪费」这个问题：
  多请求 KV Cache + Static Batching  Padding          → 引入问题：补 PAD 浪费计算
  Continuous Batching 调度器   Cont. Batching   → 解决：短请求完成即释放槽位
  PagedAttention：分页内存管理   PagedAttention   → 解决：不预分配 max_len，按需分配 Block
  FlashAttention：SRAM-aware 注意力计算   varlen 接口      → 解决：拼接输入取代 padding，零浪费

「调度延迟」这个问题：
  Continuous Batching 调度器   Scheduler        → 毫秒级：请求级调度，减少 GPU 空转
  Chunked Prefill：切片长 Prompt  Chunked Prefill  → 毫秒级：防止长 prefill 独占 GPU
  CUDA Graph：录制重放，跳过调度层   CUDA Graph       → 微秒级：跳过 Python/driver 调度开销
```
