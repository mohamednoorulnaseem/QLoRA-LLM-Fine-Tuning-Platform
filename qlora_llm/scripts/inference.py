"""
QLoRA LLM Fine-Tuning Platform — Inference Server
===================================================
Author : Mohamed Noorul Naseem M
Purpose: Run inference with fine-tuned QLoRA adapters in three modes:
         1. Interactive chat (terminal REPL)
         2. Single prompt (one-shot generation)
         3. FastAPI REST API server
Hardware: Optimized for NVIDIA RTX 4060 8GB VRAM

Usage:
    # Interactive chat
    python scripts/inference.py \\
        --base_model mistralai/Mistral-7B-Instruct-v0.2 \\
        --adapter_path models/my-finetuned-model/final_adapter \\
        --mode chat

    # Single prompt
    python scripts/inference.py \\
        --adapter_path models/my-finetuned-model/final_adapter \\
        --mode single \\
        --prompt "Explain transformers in simple terms"

    # REST API server
    python scripts/inference.py \\
        --adapter_path models/my-finetuned-model/final_adapter \\
        --mode api \\
        --port 8000
"""

import argparse
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_PORT = 8000


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_model_for_inference(
    base_model: str,
    adapter_path: str = None,
    load_in_4bit: bool = True,
):
    """
    Load a base model (optionally quantized) and merge a LoRA adapter.

    Args:
        base_model:   HuggingFace model ID or local path.
        adapter_path: Path to the LoRA adapter directory.
                      If None, loads the base model without any adapter.
        load_in_4bit: Whether to load in 4-bit mode (saves VRAM).

    Returns:
        (model, tokenizer) tuple ready for generation.
    """
    print(f"\n🔧 Loading model for inference...")
    print(f"   Base model   : {base_model}")
    print(f"   Adapter      : {adapter_path or 'None (base only)'}")
    print(f"   4-bit quant  : {load_in_4bit}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Quantization config ---
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # --- Load base model ---
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    # --- Merge LoRA adapter ---
    if adapter_path and os.path.exists(adapter_path):
        print(f"   Merging LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("   ✅ Adapter merged successfully")
    elif adapter_path:
        print(f"   ⚠️  Adapter path not found: {adapter_path}")
        print("   Running with base model only.")

    model.eval()

    # --- Memory info ---
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"\n📊 GPU Memory:")
        print(f"   Allocated : {allocated:.2f} GB")
        print(f"   Reserved  : {reserved:.2f} GB")

    print("✅ Model ready for inference\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    do_sample: bool = True,
    stream: bool = False,
) -> dict:
    """
    Generate a response for a given prompt.

    Args:
        model:              The loaded model.
        tokenizer:          The corresponding tokenizer.
        prompt:             User prompt string.
        max_new_tokens:     Maximum tokens to generate.
        temperature:        Sampling temperature (0.0 = greedy).
        top_p:              Nucleus sampling threshold.
        repetition_penalty: Penalty for repeated tokens.
        do_sample:          Whether to use sampling (False = greedy decoding).
        stream:             Whether to stream tokens to stdout.

    Returns:
        Dictionary with ``response``, ``tokens_generated``, ``time_seconds``,
        and ``tokens_per_second``.
    """
    # Format prompt in the instruction template
    formatted_prompt = (
        f"### Instruction:\n{prompt}\n\n### Response:\n"
    )

    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    # Generation config
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature if do_sample else 1.0,
        "top_p": top_p if do_sample else 1.0,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Streamer for live output
    streamer = None
    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - start_time

    # Decode only the generated tokens (skip the input prompt)
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tokens_generated = len(generated_ids)

    result = {
        "response": response,
        "tokens_generated": tokens_generated,
        "time_seconds": round(elapsed, 3),
        "tokens_per_second": round(tokens_generated / elapsed, 1) if elapsed > 0 else 0,
    }

    return result


# ---------------------------------------------------------------------------
# Mode: Interactive Chat
# ---------------------------------------------------------------------------
def run_chat_mode(model, tokenizer, **gen_kwargs):
    """Start an interactive chat REPL in the terminal."""
    print("=" * 60)
    print("💬 QLoRA Chat Mode")
    print("=" * 60)
    print("Type your message and press Enter. Commands:")
    print("  /quit or /exit  — Exit the chat")
    print("  /clear          — Clear the screen")
    print("  /config         — Show current generation settings")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit"):
            print("👋 Goodbye!")
            break
        if user_input.lower() == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue
        if user_input.lower() == "/config":
            print(f"\n⚙️  Generation Config:")
            for k, v in gen_kwargs.items():
                print(f"   {k}: {v}")
            print()
            continue

        print("\nAssistant: ", end="", flush=True)
        result = generate_response(
            model, tokenizer, user_input,
            stream=True, **gen_kwargs,
        )
        print(f"\n   [{result['tokens_generated']} tokens, "
              f"{result['time_seconds']}s, "
              f"{result['tokens_per_second']} tok/s]\n")


# ---------------------------------------------------------------------------
# Mode: Single Prompt
# ---------------------------------------------------------------------------
def run_single_mode(model, tokenizer, prompt: str, **gen_kwargs):
    """Generate a single response and exit."""
    print(f"\n📝 Prompt: {prompt}\n")
    print("=" * 60)

    result = generate_response(
        model, tokenizer, prompt,
        stream=True, **gen_kwargs,
    )

    print("\n" + "=" * 60)
    print(f"📊 Stats: {result['tokens_generated']} tokens | "
          f"{result['time_seconds']}s | "
          f"{result['tokens_per_second']} tok/s")


# ---------------------------------------------------------------------------
# Mode: FastAPI REST API
# ---------------------------------------------------------------------------
def run_api_mode(model, tokenizer, port: int = DEFAULT_PORT, host: str = "0.0.0.0"):
    """Start a FastAPI REST API server for model inference."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
        import uvicorn
    except ImportError:
        print("❌ FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    # --- Pydantic models ---
    class GenerateRequest(BaseModel):
        prompt: str = Field(..., description="The input prompt for generation")
        max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
        temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
        top_p: float = Field(DEFAULT_TOP_P, ge=0.0, le=1.0)
        repetition_penalty: float = Field(DEFAULT_REPETITION_PENALTY, ge=1.0, le=2.0)
        do_sample: bool = Field(True)

    class GenerateResponse(BaseModel):
        prompt: str
        response: str
        tokens_generated: int
        time_seconds: float
        tokens_per_second: float

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        gpu_available: bool
        gpu_name: Optional[str] = None
        gpu_memory_allocated_gb: Optional[float] = None

    # --- App lifecycle ---
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"\n🚀 API server starting on http://{host}:{port}")
        print(f"   Docs: http://localhost:{port}/docs")
        yield
        print("\n🛑 API server shutting down")

    # --- FastAPI app ---
    app = FastAPI(
        title="QLoRA LLM Inference API",
        description="REST API for fine-tuned LLM inference using QLoRA adapters",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Endpoints ---
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check if the API and model are operational."""
        gpu_name = None
        gpu_mem = None
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)

        return HealthResponse(
            status="healthy",
            model_loaded=True,
            gpu_available=torch.cuda.is_available(),
            gpu_name=gpu_name,
            gpu_memory_allocated_gb=gpu_mem,
        )

    @app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
    async def generate(request: GenerateRequest):
        """Generate a response from the fine-tuned model."""
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        try:
            result = generate_response(
                model,
                tokenizer,
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                stream=False,
            )

            return GenerateResponse(
                prompt=request.prompt,
                response=result["response"],
                tokens_generated=result["tokens_generated"],
                time_seconds=result["time_seconds"],
                tokens_per_second=result["tokens_per_second"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    @app.post("/batch_generate", tags=["Inference"])
    async def batch_generate(prompts: list[str]):
        """Generate responses for multiple prompts."""
        if not prompts:
            raise HTTPException(status_code=400, detail="Prompts list cannot be empty")
        if len(prompts) > 10:
            raise HTTPException(status_code=400, detail="Max 10 prompts per batch")

        results = []
        for prompt in prompts:
            result = generate_response(
                model, tokenizer, prompt,
                stream=False,
            )
            results.append({
                "prompt": prompt,
                "response": result["response"],
                "tokens_generated": result["tokens_generated"],
                "time_seconds": result["time_seconds"],
                "tokens_per_second": result["tokens_per_second"],
            })

        return {"results": results, "total_prompts": len(results)}

    # --- Run server ---
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="QLoRA Inference — Chat, Single Prompt, or API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat
  python scripts/inference.py \\
      --base_model mistralai/Mistral-7B-Instruct-v0.2 \\
      --adapter_path models/my-model/final_adapter \\
      --mode chat

  # Single prompt
  python scripts/inference.py \\
      --adapter_path models/my-model/final_adapter \\
      --mode single \\
      --prompt "What is QLoRA?"

  # REST API server
  python scripts/inference.py \\
      --adapter_path models/my-model/final_adapter \\
      --mode api --port 8000
        """,
    )

    # Model
    parser.add_argument("--base_model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization (uses more VRAM)")

    # Mode
    parser.add_argument("--mode", choices=["chat", "single", "api"],
                        default="chat", help="Inference mode (default: chat)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt for single mode")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"API server port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="API server host (default: 0.0.0.0)")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help=f"Max tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P,
                        help=f"Top-p nucleus sampling (default: {DEFAULT_TOP_P})")
    parser.add_argument("--repetition_penalty", type=float,
                        default=DEFAULT_REPETITION_PENALTY,
                        help=f"Repetition penalty (default: {DEFAULT_REPETITION_PENALTY})")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of sampling")

    args = parser.parse_args()

    # --- Load model ---
    model, tokenizer = load_model_for_inference(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
    )

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": not args.greedy,
    }

    # --- Run selected mode ---
    if args.mode == "chat":
        run_chat_mode(model, tokenizer, **gen_kwargs)

    elif args.mode == "single":
        if not args.prompt:
            print("❌ --prompt is required for single mode")
            sys.exit(1)
        run_single_mode(model, tokenizer, args.prompt, **gen_kwargs)

    elif args.mode == "api":
        run_api_mode(model, tokenizer, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
