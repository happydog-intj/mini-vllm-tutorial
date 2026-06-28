# Step 14: HTTP Serve — OpenAI 兼容推理服务

## 为什么需要这一步

前面的步骤已经实现了一个能正确推理的引擎（`RealModelEngine`）。但引擎是 Python 对象，只能被同一个进程调用——这意味着写好的推理能力还被"锁"在代码里，无法被其他程序、其他语言、其他机器使用。

要让推理引擎真正可用，需要解决三个问题：

1. **通信协议**：调用方怎么把请求发过来、怎么拿到结果？
2. **接口规范**：API 长什么样，调用方要不要专门适配？
3. **并发模型**：HTTP 请求是高并发 IO，GPU 推理是独占计算，两者怎么协调？

这一步通过 FastAPI 把引擎包装成 HTTP 服务，并采用 OpenAI Chat Completions 格式，使得任何已经对接过 OpenAI API 的客户端——无论是 Python 的 `openai` 库、前端的 chatUI，还是 curl——都可以直接使用，零适配成本。

## 为什么要兼容 OpenAI API 格式

OpenAI 的 Chat Completions API（`POST /v1/chat/completions`）已经成为大模型服务的事实标准接口。这个格式的最大价值不在于 OpenAI 本身，而在于**围绕它建立的庞大生态**：

```
已有的工具/框架                  通过兼容接口直接使用本地引擎
─────────────────────────────────────────────────────────
openai Python SDK  ────────────────────────────────────►
LangChain / LlamaIndex ────────────────────────────────►
各种 Chat UI (Open WebUI 等) ──────────────────────────►
curl / httpie 测试脚本  ────────────────────────────────►
                                   ▼
                         mini-vllm-tutorial server
                         (本地运行，无需网络，无需付费)
```

采用兼容格式，意味着不需要为每个工具写适配层，也不需要教用户学新的 API。

## 核心协议：Server-Sent Events（SSE）

语言模型是逐个 token 生成的——每生成一个词，理论上就可以发给用户，而不必等全部生成完再一起发。这种"边生成边发送"的体验需要一种协议来支撑。

SSE（Server-Sent Events）是 HTTP 协议原生支持的服务端推送机制：

```
客户端                              服务端
  │                                   │
  │── POST /v1/chat/completions ──────►│
  │   {"stream": true}                │
  │                                   │ 建立长连接，不断写入数据
  │◄── HTTP 200 (保持连接) ────────────│
  │◄── data: {"choices":[{"delta":    │
  │           {"content":"你"}}]}\n\n  │
  │◄── data: {"choices":[{"delta":    │
  │           {"content":"好"}}]}\n\n  │
  │◄── data: [DONE]\n\n ───────────────│
  │                                   │ 连接关闭
```

SSE 的格式非常简单：每条消息以 `data: ` 开头，以两个换行符 `\n\n` 结束。浏览器和 HTTP 客户端原生支持，不需要 WebSocket 或特殊库。

与非流式对比：

| 模式 | 体验 | 实现方式 | 适用场景 |
|------|------|---------|---------|
| `stream: false` | 等待全部生成完才返回 | 普通 JSON 响应 | 批处理、程序调用 |
| `stream: true` | 逐字出现，响应更及时 | SSE 长连接 | 对话界面、交互场景 |

**注意**：本实现中，流式模式的实际效果是"生成完毕后逐字符回放"——因为底层引擎是同步的，生成和输出仍是串行的。真正的逐 token 实时推送需要引擎支持 `AsyncGenerator`，这是更进阶的话题。

## 为什么用 asyncio 而不是多线程

FastAPI 基于 asyncio，天然适合处理大量并发 HTTP 请求。这里有一个常见的问题：

> 直接在 async 函数里调用同步的推理代码会发生什么？

答案是：**整个事件循环会被阻塞**。asyncio 是单线程的，如果在 async 函数里直接调用耗时的同步代码，其他所有请求都得等着。

解决方案是 `run_in_executor`，它把同步任务丢进线程池，让事件循环可以继续处理其他 IO：

```python
# ❌ 错误：阻塞事件循环
output = engine.generate(prompt)

# ✅ 正确：在线程池里运行，不阻塞
loop = asyncio.get_event_loop()
output = await loop.run_in_executor(
    None, lambda: engine.generate(prompt, max_new_tokens=request.max_tokens)
)
```

```
asyncio 事件循环（单线程）
┌─────────────────────────────────────────────┐
│                                             │
│  请求A ──► run_in_executor ──► 线程池       │
│                │                  │         │
│  请求B ──► 接受连接               │ GPU推理  │
│                │                  │         │
│  请求C ──► 接受连接               │         │
│                │                  ▼         │
│  请求A ◄─ await 收到结果 ◄────────┘         │
└─────────────────────────────────────────────┘
```

本质上，推理是 GPU 密集型操作，HTTP 收发是 IO 密集型操作。asyncio 负责高效管理 IO，线程池负责隔离 GPU 计算，两者分工明确。

## ChatML 格式：messages 怎么变成 prompt

OpenAI API 接受结构化的 `messages` 数组，但模型实际接受的是一段文本。`messages_to_prompt` 函数负责这个转换，使用 Qwen3 的 ChatML 模板：

```python
# 输入（API 格式）
messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user",   "content": "你好"}
]

# 输出（模型实际看到的文本）
"""
<|im_start|>system
你是一个助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
"""
```

最后一行 `<|im_start|>assistant\n` 是关键：它告诉模型"现在该你说话了"，模型会从这个位置开始续写。不同模型有不同的对话模板（Llama 用 `[INST]`，ChatML 用 `<|im_start|>`），这里固定使用 Qwen3 的格式。

## 代码结构

```
server.py
│
├── ChatCompletionRequest          # 请求体结构（Pydantic 验证）
│   ├── model: str
│   ├── messages: List[Message]
│   ├── max_tokens: int
│   ├── temperature: float
│   └── stream: bool
│
├── init_engine(model_path)        # 启动时加载模型（阻塞，只执行一次）
│
├── messages_to_prompt(messages)   # ChatML 格式转换
│
├── make_chunk(content, model)     # 构造 SSE 数据块
│
├── POST /v1/chat/completions      # 主接口
│   ├── stream=False → JSONResponse
│   └── stream=True  → StreamingResponse (SSE)
│
└── GET  /health                   # 健康检查
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `server.py` | FastAPI 服务，实现 `/v1/chat/completions` 和 `/health` |
| `run.py` | 语法检查 + 依赖可用性验证 |

## 启动服务

```bash
# 安装依赖
pip install fastapi uvicorn

# 启动（需要先确保 step15_real_model 的模型路径可用）
python server.py --model ~/huggingface/Qwen3-0.6B --port 8000
```

启动成功后会看到：
```
加载模型: ~/huggingface/Qwen3-0.6B
服务就绪 ✅

服务启动: http://0.0.0.0:8000
```

## 测试

**健康检查：**
```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

**非流式请求：**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}], "max_tokens": 50}'
```

**流式请求：**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}], "max_tokens": 50, "stream": true}'
```

**使用 OpenAI Python SDK（直接指向本地服务）：**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

## 验证（无需 GPU）

```bash
python run.py
```

输出：
```
HTTP Serve：OpenAI 兼容推理服务: HTTP 服务封装（OpenAI 兼容接口）
==================================================
server.py 语法检查通过 ✅
FastAPI x.x.x / uvicorn 依赖可用 ✅

✅ step20_serve 通过
```

## 与真实 vLLM 的差距

本实现是一个结构完整但功能简化的版本，与生产级 vLLM 的主要差距：

| 特性 | 本实现 | vLLM |
|------|--------|------|
| 并发请求 | 串行执行（线程池单 worker） | 真正并发，continuous batching |
| 流式推送 | 生成完后回放字符 | 每生成一个 token 立刻推送 |
| 请求队列 | 无，直接阻塞 | 带优先级的调度器 |
| 多模型 | 单模型 | 多模型、多 LoRA |
| OpenAI 完整兼容 | messages + stream 基础功能 | function calling、logprobs、tools 等 |

最核心的差距在于：vLLM 的推理引擎本身就是异步的——每生成一个 token 就可以立刻通过 SSE 推出去，不需要等全部完成。这需要引擎层面支持 `AsyncGenerator`，是 step 之后可以继续探索的方向。

## 下一步

到这里，我们已经有了一个完整的端到端推理服务：从模型权重加载，到 KV Cache 管理，到 HTTP API 对外服务。

但还有一个关键能力缺失：**同时处理多个请求时的效率**。当前实现每次只能处理一个请求，GPU 的大部分算力在等待 IO 时处于空闲状态。下一步要解决的问题是：如何把多个请求的 token 生成合并到同一次 GPU 前向计算中——这就是 vLLM 最核心的优化：**Continuous Batching**（连续批处理）。
