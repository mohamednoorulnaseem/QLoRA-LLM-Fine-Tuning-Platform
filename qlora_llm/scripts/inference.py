#!/usr/bin/env python3
"""
inference.py — Model Serving: Interactive Chat, Single Prompt & FastAPI REST API
=================================================================================

Three modes:
  1. chat   — Interactive CLI with token-by-token streaming
  2. single — One-shot generation on a single prompt
  3. api    — FastAPI REST server with /generate and /health endpoints

Loads the base model in 4-bit, merges LoRA adapter via merge_and_unload(),
and serves the result for zero-overhead inference.

Usage:
  # Interactive chat
  python scripts/inference.py --adapter_path models/my-model/final_adapter --mode chat

  # Single prompt
  python scripts/inference.py --adapter_path models/my-model/final_adapter \\
    --mode single --prompt "Explain QLoRA"

  # REST API server
  python scripts/inference.py --adapter_path models/my-model/final_adapter \\
    --mode api --port 8000

Author : Mohamed Noorul Naseem M
Hardware: Lenovo LOQ | RTX 4060 8GB | i7-13650HX | 24GB RAM
"""

import argparse
import os
import sys
import time
from typing import Optional

import torch
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

console = Console()

# ──────────────────────────── Defaults ────────────────────────────
DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_ADAPTER = os.path.join("models", "qlora-mistral-7b", "final_adapter")
DEFAULT_PORT = 8000

# Generation defaults
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1


# ──────────────────────────── Model Loading ───────────────────────


def load_model(base_model: str, adapter_path: Optional[str] = None, merge: bool = True):
    """
    Load the base model in 4-bit quantization and optionally merge a LoRA adapter.

    Args:
        base_model: HuggingFace model ID or local path.
        adapter_path: Path to LoRA adapter directory. If None, loads base model only.
        merge: If True, merge adapter weights into base model for faster inference.

    Returns:
        Tuple of (model, tokenizer).
    """
    console.print(f"\n⏳ Loading base model: [bold]{base_model}[/]")

    # Quantization config for inference (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path and os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")) else base_model,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    mem_used = torch.cuda.memory_allocated(0) / 1e9
    console.print(f"   ✅ Base model loaded — VRAM: [bold]{mem_used:.2f} GB[/]")

    # Load and merge LoRA adapter
    if adapter_path and os.path.exists(adapter_path):
        console.print(f"🔗 Loading LoRA adapter from: [cyan]{adapter_path}[/]")
        model = PeftModel.from_pretrained(model, adapter_path)

        if merge:
            console.print("   Merging adapter weights into base model...")
            model = model.merge_and_unload()
            console.print("   ✅ Adapter merged — zero-overhead inference ready")
        else:
            console.print("   ✅ Adapter loaded (not merged)")
    elif adapter_path:
        console.print(f"[yellow]⚠ Adapter path not found: {adapter_path} — using base model only[/]")

    model.eval()
    mem_final = torch.cuda.memory_allocated(0) / 1e9
    console.print(f"   📊 Final VRAM usage: [bold]{mem_final:.2f} GB[/]\n")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    stream: bool = False,
) -> str:
    """
    Generate a response from the model.

    Args:
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
        prompt: User input prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.1=deterministic, 0.9=creative).
        top_p: Nucleus sampling probability.
        repetition_penalty: Penalty for repeated tokens.
        stream: If True, stream tokens to stdout.

    Returns:
        Generated response string.
    """
    # Format as Mistral instruct
    formatted = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Setup streamer if requested
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            streamer=streamer,
        )

    # Decode response (skip the prompt tokens)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    return response


# ──────────────────────────── Modes ───────────────────────────────


def run_chat(model, tokenizer, args):
    """Interactive chat mode with streaming output."""
    console.print(Panel(
        "[bold green]💬 Interactive Chat Mode[/]\n"
        "Type your message and press Enter.\n"
        "Commands: [cyan]/quit[/] to exit, [cyan]/clear[/] to clear screen.",
        border_style="green",
    ))

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n\n[dim]Goodbye! 👋[/]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("\n[dim]Goodbye! 👋[/]")
            break
        if user_input.lower() == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        console.print("\n🤖 [bold cyan]Assistant:[/] ", end="")
        start = time.time()
        response = generate_response(
            model, tokenizer, user_input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stream=True,
        )
        elapsed = time.time() - start
        tokens = len(tokenizer.encode(response))
        console.print(f"\n[dim]({tokens} tokens, {elapsed:.1f}s, {tokens / elapsed:.1f} tok/s)[/]")


def run_single(model, tokenizer, args):
    """Single prompt mode — generate one response and exit."""
    if not args.prompt:
        console.print("[bold red]❌ --prompt is required in single mode[/]")
        sys.exit(1)

    console.print(f"\n📝 Prompt: [bold]{args.prompt}[/]\n")
    console.print("🤖 [bold cyan]Response:[/]\n")

    start = time.time()
    response = generate_response(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stream=True,
    )
    elapsed = time.time() - start
    tokens = len(tokenizer.encode(response))

    console.print(f"\n\n[dim]({tokens} tokens, {elapsed:.1f}s, {tokens / elapsed:.1f} tok/s)[/]")


def run_api(model, tokenizer, args):
    """FastAPI REST API server with /generate and /health endpoints."""
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError:
        console.print("[bold red]❌ FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn[/]")
        sys.exit(1)

    app = FastAPI(
        title="QLoRA LLM Inference API",
        description="Fine-tuned LLM inference endpoint powered by QLoRA",
        version="1.0.0",
    )

    class GenerateRequest(BaseModel):
        prompt: str = Field(..., description="Input prompt for the model")
        max_new_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1, le=2048, description="Maximum tokens to generate")
        temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0, description="Sampling temperature")
        top_p: float = Field(DEFAULT_TOP_P, ge=0.0, le=1.0, description="Nucleus sampling probability")

    class GenerateResponse(BaseModel):
        response: str
        tokens_generated: int
        inference_time_seconds: float
        tokens_per_second: float

    class HealthResponse(BaseModel):
        status: str
        gpu: str
        vram_used_gb: float
        vram_total_gb: float

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint — returns GPU info and status."""
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
        vram_used = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        vram_total = torch.cuda.get_device_properties(0).total_mem / 1e9 if torch.cuda.is_available() else 0
        return HealthResponse(
            status="ok",
            gpu=gpu_name,
            vram_used_gb=round(vram_used, 2),
            vram_total_gb=round(vram_total, 2),
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Generate text from a prompt using the fine-tuned model."""
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        start = time.time()
        try:
            response_text = generate_response(
                model, tokenizer, request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

        elapsed = time.time() - start
        tokens = len(tokenizer.encode(response_text))

        return GenerateResponse(
            response=response_text,
            tokens_generated=tokens,
            inference_time_seconds=round(elapsed, 3),
            tokens_per_second=round(tokens / elapsed, 1) if elapsed > 0 else 0,
        )

    console.print(Panel(
        f"[bold green]🚀 FastAPI Server Starting[/]\n"
        f"Host: [white]0.0.0.0:{args.port}[/]\n"
        f"Docs: [cyan]http://localhost:{args.port}/docs[/]",
        border_style="green",
    ))

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


# ──────────────────────────── CLI ─────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA Inference Server — Chat, Single Prompt, or REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL,
                        help=f"Base model ID (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER,
                        help=f"LoRA adapter path (default: {DEFAULT_ADAPTER})")
    parser.add_argument("--mode", choices=["chat", "single", "api"], default="chat",
                        help="Inference mode (default: chat)")

    # Generation options
    parser.add_argument("--prompt", help="Prompt for single mode")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max new tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Temperature (default: {DEFAULT_TEMPERATURE})")

    # API options
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"API port (default: {DEFAULT_PORT})")

    # Model options
    parser.add_argument("--no_merge", action="store_true",
                        help="Don't merge adapter weights (keeps PEFT model)")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(
        args.base_model,
        adapter_path=args.adapter_path,
        merge=not args.no_merge,
    )

    # Run selected mode
    if args.mode == "chat":
        run_chat(model, tokenizer, args)
    elif args.mode == "single":
        run_single(model, tokenizer, args)
    elif args.mode == "api":
        run_api(model, tokenizer, args)


if __name__ == "__main__":
    main()
