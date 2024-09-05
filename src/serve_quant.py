import argparse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import gc
import torch
import time
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Global variables
engine = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False

@app.on_event("startup")
async def startup_event():
    global engine
    parser = argparse.ArgumentParser(description="vLLM Server with LoRA support")
    parser.add_argument("model", type=str, help="The name or path of the HuggingFace model to use")
    parser.add_argument("--enable-lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--lora-modules", type=str, help="LoRA modules to use")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization method")
    parser.add_argument("--gpu-utilization", type=float, default=0.9, help="Target GPU memory utilization (0.0 to 1.0)")
    args = parser.parse_args()

    engine_args = AsyncEngineArgs(
        model=args.model,
        quantization=args.quantization,
        enable_lora=args.enable_lora,
        gpu_memory_utilization=args.gpu_utilization,
    )

    if args.enable_lora:
        lora_modules = {}
        for module in args.lora_modules.split(','):
            name, path = module.split('=')
            lora_modules[name] = path
        engine_args.lora_modules = lora_modules

    if args.quantization == "awq":
        engine_args.dtype = 'float16'

    engine = AsyncLLMEngine.from_engine_args(engine_args)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global engine

    prompt = ""
    for message in request.messages:
        prompt += f"{message.role}: {message.content}\n"
    prompt += "assistant:"

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    results_generator = engine.generate(prompt, sampling_params, random_uuid())

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output:
        response = {
            "id": "chatcmpl-" + random_uuid(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_output.outputs[0].text.strip()
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(final_output.outputs[0].text.split()),
                "total_tokens": len(prompt.split()) + len(final_output.outputs[0].text.split())
            }
        }
        return JSONResponse(content=response)
    else:
        return JSONResponse(content={"error": "Generation failed"}, status_code=500)

@app.on_event("shutdown")
async def shutdown_event():
    global engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)