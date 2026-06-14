"""
step13: OpenAI 兼容 HTTP 推理服务
"""
import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from typing import List, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3"
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


app = FastAPI(title="mini-vllm-tutorial Server")
engine = None


def init_engine(model_path: str):
    global engine
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step08_real_model'))
    from engine import RealModelEngine
    print(f"加载模型: {model_path}")
    engine = RealModelEngine(model_path=model_path)
    print("服务就绪 ✅")


def messages_to_prompt(messages: List[Message]) -> str:
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def make_chunk(content: str, model: str, finish_reason: Optional[str] = None) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content} if content else {},
                      "finish_reason": finish_reason}]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if engine is None:
        return JSONResponse({"error": "模型未加载"}, status_code=503)

    prompt = messages_to_prompt(request.messages)

    if request.stream:
        async def generate_stream():
            yield make_chunk("", request.model)
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None, lambda: engine.generate(prompt, max_new_tokens=request.max_tokens)
            )
            for char in output:
                yield make_chunk(char, request.model)
                await asyncio.sleep(0)
            yield make_chunk("", request.model, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    else:
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None, lambda: engine.generate(prompt, max_new_tokens=request.max_tokens)
        )
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion",
                "created": int(time.time()), "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": output},
                              "finish_reason": "stop"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    init_engine(args.model)
    print(f"\n服务启动: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
