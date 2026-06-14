# Step 13: HTTP Serve — OpenAI 兼容推理服务

## 本节目标

将推理引擎封装为 OpenAI 兼容的 HTTP API，支持流式和非流式两种响应模式。

## 核心概念

### OpenAI Chat Completions API

标准接口格式（`/v1/chat/completions`）：

```json
POST /v1/chat/completions
{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "你好"}],
  "max_tokens": 512,
  "stream": false
}
```

响应：
```json
{
  "choices": [{"message": {"role": "assistant", "content": "你好！..."}}]
}
```

### 流式响应（SSE）

`stream: true` 时，服务器通过 Server-Sent Events 逐字返回：

```
data: {"choices": [{"delta": {"content": "你"}}]}
data: {"choices": [{"delta": {"content": "好"}}]}
data: [DONE]
```

### ChatML 格式

Qwen3 使用 ChatML 模板构造 prompt：

```
<|im_start|>system
你是一个助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```

### 异步推理

用 `asyncio.run_in_executor` 将同步的模型推理放入线程池，避免阻塞 FastAPI event loop：

```python
output = await loop.run_in_executor(None, lambda: engine.generate(prompt))
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `server.py` | FastAPI 服务，实现 `/v1/chat/completions` 和 `/health` |
| `run.py` | 语法检查 + 依赖验证 |

## 启动服务

```bash
pip install fastapi uvicorn

python server.py --model ~/huggingface/Qwen3-0.6B --port 8000
```

## 测试

```bash
curl http://localhost:8000/health

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}], "max_tokens": 50}'
```

## 与 vLLM 的差异

真实 vLLM 还包括：
- 请求队列与调度器（continuous batching）
- 异步 token streaming（逐 token 推送，而非生成完再流）
- 多模型、多 LoRA 支持
- OpenAI 完整兼容（function calling、logprobs 等）
