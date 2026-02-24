#!/usr/bin/env python3
"""
evaluate.py — Evaluation Suite for QLoRA Fine-Tuned Models
============================================================

Evaluation metrics:
  • Perplexity       — exp(mean NLL) on held-out eval data (lower = better)
  • ROUGE scores     — ROUGE-1, ROUGE-2, ROUGE-L F1 for generation quality
  • Speed benchmark  — Tokens/second averaged over 3 runs with GPU warmup
  • Benchmark prompts — JSON-driven qualitative eval with keyword matching
  • VRAM reporting   — Current GPU memory usage stats

Usage:
  python scripts/evaluate.py \\
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \\
    --adapter_path models/my-model/final_adapter \\
    --eval_data data/train.jsonl

Author : Mohamed Noorul Naseem M
Hardware: Lenovo LOQ | RTX 4060 8GB | i7-13650HX | 24GB RAM
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

console = Console()

# ──────────────────────────── Defaults ────────────────────────────
DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_ADAPTER = os.path.join("models", "qlora-mistral-7b", "final_adapter")
DEFAULT_EVAL_DATA = os.path.join("data", "train.jsonl")
BENCHMARK_PROMPTS_PATH = os.path.join("configs", "benchmark_prompts.json")


# ──────────────────────────── Model Loading ───────────────────────


def load_model_for_eval(base_model: str, adapter_path: Optional[str] = None):
    """
    Load the model in 4-bit for evaluation.

    Args:
        base_model: HuggingFace model ID or local path.
        adapter_path: Path to LoRA adapter directory.

    Returns:
        Tuple of (model, tokenizer).
    """
    console.print(f"\n⏳ Loading model for evaluation: [bold]{base_model}[/]")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path and os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")) else base_model,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if adapter_path and os.path.exists(adapter_path):
        console.print(f"🔗 Loading LoRA adapter: [cyan]{adapter_path}[/]")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        console.print("   ✅ Adapter merged for evaluation")
    elif adapter_path:
        console.print(f"[yellow]⚠ Adapter not found: {adapter_path} — evaluating base model[/]")

    model.eval()
    mem = torch.cuda.memory_allocated(0) / 1e9
    console.print(f"   📊 VRAM usage: [bold]{mem:.2f} GB[/]\n")

    return model, tokenizer


# ──────────────────────────── Perplexity ──────────────────────────


def compute_perplexity(
    model,
    tokenizer,
    eval_data_path: str,
    max_samples: int = 100,
    max_length: int = 2048,
) -> float:
    """
    Compute perplexity on evaluation data.

    Perplexity = exp(mean negative log-likelihood) — lower is better.

    Args:
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
        eval_data_path: Path to JSONL evaluation data.
        max_samples: Maximum number of samples to evaluate.
        max_length: Maximum token length per sample.

    Returns:
        Perplexity score (float).
    """
    console.print("[bold]📊 Computing Perplexity...[/]")

    if not os.path.exists(eval_data_path):
        console.print(f"[bold red]❌ Eval data not found: {eval_data_path}[/]")
        return float("inf")

    # Load eval samples
    samples = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    samples = samples[:max_samples]
    console.print(f"   Evaluating on {len(samples)} samples...")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for sample in tqdm(samples, desc="   Perplexity"):
            # Format as Mistral instruct
            text = f"<s>[INST] {sample.get('instruction', '')}\n\n{sample.get('input', '')} [/INST] {sample.get('output', '')}</s>"

            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            input_ids = encodings["input_ids"]
            target_ids = input_ids.clone()

            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss

            num_tokens = input_ids.size(1)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    color = "green" if perplexity < 5.0 else "yellow" if perplexity < 10.0 else "red"
    console.print(f"   Perplexity: [{color}]{perplexity:.4f}[/] (target: < 5.0)\n")

    return float(perplexity)


# ──────────────────────────── ROUGE ───────────────────────────────


def compute_rouge(
    model,
    tokenizer,
    eval_data_path: str,
    max_samples: int = 50,
    max_new_tokens: int = 256,
) -> dict:
    """
    Compute ROUGE scores comparing model generations to reference outputs.

    Args:
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
        eval_data_path: Path to JSONL evaluation data.
        max_samples: Maximum number of samples.
        max_new_tokens: Max tokens to generate per sample.

    Returns:
        Dict with rouge1, rouge2, rougeL F1 scores.
    """
    console.print("[bold]📊 Computing ROUGE scores...[/]")

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        console.print("[yellow]⚠ rouge-score not installed. Run: pip install rouge-score[/]")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    if not os.path.exists(eval_data_path):
        console.print(f"[bold red]❌ Eval data not found: {eval_data_path}[/]")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # Load eval samples
    samples = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    samples = samples[:max_samples]
    console.print(f"   Evaluating on {len(samples)} samples...")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    with torch.no_grad():
        for sample in tqdm(samples, desc="   ROUGE"):
            prompt = f"<s>[INST] {sample.get('instruction', '')}\n\n{sample.get('input', '')} [/INST]"
            reference = sample.get("output", "")

            if not reference:
                continue

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
            )

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            scores = scorer.score(reference, generated)
            for key in all_scores:
                all_scores[key].append(scores[key].fmeasure)

    # Average scores
    results = {}
    for key in all_scores:
        results[key] = float(np.mean(all_scores[key])) if all_scores[key] else 0.0

    table = Table(title="ROUGE Scores", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("F1 Score", style="bold")
    for key, val in results.items():
        color = "green" if val > 0.3 else "yellow" if val > 0.1 else "red"
        table.add_row(key.upper(), f"[{color}]{val:.4f}[/]")
    console.print(table)

    return results


# ──────────────────────────── Speed Benchmark ─────────────────────


def speed_benchmark(
    model,
    tokenizer,
    prompt: str = "Explain the concept of machine learning in detail.",
    max_new_tokens: int = 256,
    num_runs: int = 3,
) -> dict:
    """
    Benchmark inference speed (tokens/second) averaged over multiple runs.
    Includes a GPU warmup pass.

    Args:
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
        prompt: Prompt to use for benchmarking.
        max_new_tokens: Tokens to generate per run.
        num_runs: Number of benchmark runs.

    Returns:
        Dict with avg_tokens_per_sec, total_tokens, avg_time_seconds.
    """
    console.print("[bold]⚡ Running Speed Benchmark...[/]")

    formatted = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Warmup pass (not counted)
    console.print("   🔥 Warmup pass...")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=20, do_sample=False)

    times = []
    token_counts = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start

        gen_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
        times.append(elapsed)
        token_counts.append(gen_tokens)

        console.print(f"   Run {i + 1}/{num_runs}: {gen_tokens} tokens in {elapsed:.2f}s ({gen_tokens / elapsed:.1f} tok/s)")

    avg_time = np.mean(times)
    avg_tokens = np.mean(token_counts)
    avg_speed = avg_tokens / avg_time

    result = {
        "avg_tokens_per_sec": float(avg_speed),
        "total_tokens": int(sum(token_counts)),
        "avg_time_seconds": float(avg_time),
        "num_runs": num_runs,
    }

    color = "green" if avg_speed >= 15 else "yellow" if avg_speed >= 10 else "red"
    console.print(f"\n   Average speed: [{color}]{avg_speed:.1f} tok/s[/] (target: ≥ 15 tok/s)\n")

    return result


# ──────────────────────────── Benchmark Prompts ───────────────────


def run_benchmark_prompts(
    model,
    tokenizer,
    benchmark_path: str = BENCHMARK_PROMPTS_PATH,
    max_new_tokens: int = 256,
) -> dict:
    """
    Run qualitative evaluation using JSON-configured benchmark prompts.
    Checks for expected keyword presence in generated outputs.

    Args:
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
        benchmark_path: Path to benchmark_prompts.json.
        max_new_tokens: Tokens to generate per prompt.

    Returns:
        Dict with per-prompt results and overall pass rate.
    """
    console.print("[bold]🎯 Running Benchmark Prompts...[/]")

    if not os.path.exists(benchmark_path):
        console.print(f"[yellow]⚠ Benchmark file not found: {benchmark_path}[/]")
        return {"pass_rate": 0.0, "results": []}

    with open(benchmark_path, "r") as f:
        config = json.load(f)

    prompts = config.get("prompts", [])
    defaults = config.get("model_defaults", {})
    max_new_tokens = defaults.get("max_new_tokens", max_new_tokens)

    results = []
    passed = 0

    table = Table(title="Benchmark Results", show_header=True)
    table.add_column("ID", style="cyan", width=15)
    table.add_column("Category", width=20)
    table.add_column("Keywords Found", width=15)
    table.add_column("Pass", width=6)

    for prompt_config in tqdm(prompts, desc="   Benchmarks"):
        prompt_text = prompt_config["prompt"]
        expected_keywords = prompt_config.get("expected_keywords", [])
        prompt_id = prompt_config.get("id", "unknown")
        category = prompt_config.get("category", "unknown")

        # Generate response
        formatted = f"<s>[INST] {prompt_text} [/INST]"
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=defaults.get("temperature", 0.7),
                top_p=defaults.get("top_p", 0.9),
                do_sample=True,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        # Check keyword matches
        found_keywords = [kw for kw in expected_keywords if kw.lower() in generated.lower()]
        match_ratio = len(found_keywords) / len(expected_keywords) if expected_keywords else 1.0
        is_pass = match_ratio >= 0.5  # Pass if ≥50% keywords found

        if is_pass:
            passed += 1

        result = {
            "id": prompt_id,
            "category": category,
            "prompt": prompt_text,
            "generated": generated[:200] + "..." if len(generated) > 200 else generated,
            "expected_keywords": expected_keywords,
            "found_keywords": found_keywords,
            "match_ratio": match_ratio,
            "passed": is_pass,
        }
        results.append(result)

        color = "green" if is_pass else "red"
        table.add_row(
            prompt_id,
            category,
            f"{len(found_keywords)}/{len(expected_keywords)}",
            f"[{color}]{'✅' if is_pass else '❌'}[/]",
        )

    console.print(table)

    pass_rate = passed / len(prompts) if prompts else 0.0
    color = "green" if pass_rate >= 0.7 else "yellow" if pass_rate >= 0.5 else "red"
    console.print(f"\n   Overall pass rate: [{color}]{pass_rate:.0%}[/] ({passed}/{len(prompts)})\n")

    return {"pass_rate": pass_rate, "passed": passed, "total": len(prompts), "results": results}


# ──────────────────────────── VRAM Report ─────────────────────────


def vram_report() -> dict:
    """Report current GPU VRAM usage."""
    if not torch.cuda.is_available():
        return {"available": False}

    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    free = total - allocated

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(free, 2),
        "utilization_pct": round(allocated / total * 100, 1),
    }

    table = Table(title="📊 VRAM Report", show_header=False)
    table.add_column("", style="cyan")
    table.add_column("", style="bold")
    table.add_row("GPU", report["gpu"])
    table.add_row("Allocated", f"{report['allocated_gb']:.2f} GB")
    table.add_row("Reserved", f"{report['reserved_gb']:.2f} GB")
    table.add_row("Total", f"{report['total_gb']:.2f} GB")
    table.add_row("Free", f"{report['free_gb']:.2f} GB")
    table.add_row("Utilization", f"{report['utilization_pct']:.1f}%")
    console.print(table)

    return report


# ──────────────────────────── Full Evaluation ─────────────────────


def run_full_evaluation(model, tokenizer, args):
    """Run the complete evaluation suite and save results."""
    console.print(Panel(
        "[bold cyan]📊 Full Evaluation Suite[/]\n"
        f"Model: [white]{args.base_model}[/]\n"
        f"Adapter: [white]{args.adapter_path}[/]\n"
        f"Eval Data: [white]{args.eval_data}[/]",
        title="Evaluation Configuration",
        border_style="cyan",
    ))

    all_results = {}

    # 1. VRAM Report
    console.print("\n" + "=" * 50)
    all_results["vram"] = vram_report()

    # 2. Perplexity
    console.print("=" * 50)
    all_results["perplexity"] = compute_perplexity(
        model, tokenizer, args.eval_data,
        max_samples=args.max_samples,
    )

    # 3. Speed Benchmark
    console.print("=" * 50)
    all_results["speed"] = speed_benchmark(
        model, tokenizer,
        num_runs=args.speed_runs,
    )

    # 4. ROUGE (optional — can be slow)
    if args.compute_rouge:
        console.print("=" * 50)
        all_results["rouge"] = compute_rouge(
            model, tokenizer, args.eval_data,
            max_samples=min(args.max_samples, 50),
        )

    # 5. Benchmark Prompts (if file exists)
    if os.path.exists(args.benchmark_path):
        console.print("=" * 50)
        all_results["benchmark_prompts"] = run_benchmark_prompts(
            model, tokenizer,
            benchmark_path=args.benchmark_path,
        )

    # ── Summary ──
    console.print("\n" + "=" * 50)
    summary = Table(title="📋 Evaluation Summary", show_header=True)
    summary.add_column("Metric", style="cyan", width=25)
    summary.add_column("Value", style="bold", width=20)
    summary.add_column("Target", width=15)
    summary.add_column("Status", width=6)

    # Perplexity
    ppl = all_results.get("perplexity", float("inf"))
    ppl_pass = "✅" if ppl < 5.0 else "❌"
    summary.add_row("Perplexity", f"{ppl:.4f}", "< 5.0", ppl_pass)

    # Speed
    speed = all_results.get("speed", {}).get("avg_tokens_per_sec", 0)
    speed_pass = "✅" if speed >= 15 else "❌"
    summary.add_row("Tokens/sec", f"{speed:.1f}", "≥ 15", speed_pass)

    # ROUGE
    if "rouge" in all_results:
        r1 = all_results["rouge"].get("rouge1", 0)
        summary.add_row("ROUGE-1 F1", f"{r1:.4f}", "> 0.3", "✅" if r1 > 0.3 else "❌")

    # Benchmark pass rate
    if "benchmark_prompts" in all_results:
        pr = all_results["benchmark_prompts"].get("pass_rate", 0)
        summary.add_row("Benchmark Pass", f"{pr:.0%}", "≥ 70%", "✅" if pr >= 0.7 else "❌")

    # VRAM
    vram_used = all_results.get("vram", {}).get("allocated_gb", 0)
    summary.add_row("VRAM Usage", f"{vram_used:.2f} GB", "≤ 7.5 GB", "✅" if vram_used <= 7.5 else "❌")

    console.print(summary)

    # Save results to JSON
    output_file = os.path.join(os.path.dirname(args.eval_data) or ".", "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n💾 Results saved → [cyan]{output_file}[/]\n")

    return all_results


# ──────────────────────────── CLI ─────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA Evaluation Suite — Perplexity, ROUGE, Speed, Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL,
                        help=f"Base model (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER,
                        help=f"LoRA adapter (default: {DEFAULT_ADAPTER})")
    parser.add_argument("--eval_data", default=DEFAULT_EVAL_DATA,
                        help=f"Eval JSONL (default: {DEFAULT_EVAL_DATA})")
    parser.add_argument("--benchmark_path", default=BENCHMARK_PROMPTS_PATH,
                        help=f"Benchmark prompts JSON (default: {BENCHMARK_PROMPTS_PATH})")

    # Evaluation options
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Max samples for perplexity/ROUGE (default: 100)")
    parser.add_argument("--speed_runs", type=int, default=3,
                        help="Number of speed benchmark runs (default: 3)")
    parser.add_argument("--compute_rouge", action="store_true",
                        help="Include ROUGE evaluation (slower)")

    # Quick mode options
    parser.add_argument("--perplexity_only", action="store_true",
                        help="Only compute perplexity")
    parser.add_argument("--speed_only", action="store_true",
                        help="Only run speed benchmark")
    parser.add_argument("--benchmark_only", action="store_true",
                        help="Only run benchmark prompts")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_for_eval(args.base_model, args.adapter_path)

    # Quick mode or full evaluation
    if args.perplexity_only:
        compute_perplexity(model, tokenizer, args.eval_data, max_samples=args.max_samples)
    elif args.speed_only:
        speed_benchmark(model, tokenizer, num_runs=args.speed_runs)
    elif args.benchmark_only:
        run_benchmark_prompts(model, tokenizer, benchmark_path=args.benchmark_path)
    else:
        run_full_evaluation(model, tokenizer, args)


if __name__ == "__main__":
    main()
