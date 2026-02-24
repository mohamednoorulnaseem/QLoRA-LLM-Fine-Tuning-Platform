"""
QLoRA LLM Fine-Tuning Platform — Evaluation Suite
===================================================
Author : Mohamed Noorul Naseem M
Purpose: Evaluate fine-tuned QLoRA models with multiple metrics:
         1. Perplexity — language modeling quality
         2. ROUGE scores — response similarity to references
         3. Inference speed — tokens/sec throughput
         4. Benchmark prompts — keyword-match scoring
Hardware: Optimized for NVIDIA RTX 4060 8GB VRAM

Usage:
    python scripts/evaluate.py \\
        --base_model mistralai/Mistral-7B-Instruct-v0.2 \\
        --adapter_path models/my-finetuned-model/final_adapter \\
        --eval_data data/train.jsonl

    # Benchmark prompts only
    python scripts/evaluate.py \\
        --adapter_path models/my-model/final_adapter \\
        --benchmark_only
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_EVAL_DATA = str(PROJECT_ROOT / "data" / "train.jsonl")
DEFAULT_BENCHMARK = str(PROJECT_ROOT / "configs" / "benchmark_prompts.json")
DEFAULT_MAX_SAMPLES = 100  # Max samples for perplexity evaluation


# ---------------------------------------------------------------------------
# Model Loading (identical to inference.py for consistency)
# ---------------------------------------------------------------------------
def load_model(base_model: str, adapter_path: str = None):
    """Load a quantized model with optional LoRA adapter for evaluation."""
    print(f"\n🔧 Loading model: {base_model}")
    if adapter_path:
        print(f"   Adapter: {adapter_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        print("   ✅ LoRA adapter merged")

    model.eval()
    print("   ✅ Model loaded & ready for evaluation\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Metric 1: Perplexity
# ---------------------------------------------------------------------------
def evaluate_perplexity(
    model,
    tokenizer,
    eval_data_path: str,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    max_length: int = 1024,
) -> dict:
    """
    Compute perplexity on an evaluation JSONL dataset.

    Perplexity = exp(average cross-entropy loss).
    Lower is better. A well fine-tuned model should have perplexity < 5.0.

    Args:
        model:          The loaded model.
        tokenizer:      The corresponding tokenizer.
        eval_data_path: Path to evaluation JSONL file.
        max_samples:    Maximum number of samples to evaluate.
        max_length:     Maximum token length per sample.

    Returns:
        Dictionary with perplexity stats.
    """
    print("=" * 60)
    print("📊 Evaluating: PERPLEXITY")
    print("=" * 60)

    # Load evaluation data
    samples = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                # Format as instruction-response pair
                instruction = record.get("instruction", "")
                user_input = record.get("input", "")
                output = record.get("output", "")

                if user_input.strip():
                    text = (
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Input:\n{user_input}\n\n"
                        f"### Response:\n{output}"
                    )
                else:
                    text = (
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Response:\n{output}"
                    )
                samples.append(text)

            if len(samples) >= max_samples:
                break

    print(f"   Evaluating {len(samples)} samples...")

    total_loss = 0.0
    total_tokens = 0
    per_sample_ppl = []

    with torch.no_grad():
        for i, text in enumerate(samples):
            # Tokenize
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            input_ids = encodings["input_ids"]
            seq_len = input_ids.shape[1]

            if seq_len < 2:
                continue

            # Forward pass
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()

            total_loss += loss * (seq_len - 1)  # Weight by token count
            total_tokens += seq_len - 1
            per_sample_ppl.append(math.exp(loss))

            if (i + 1) % 20 == 0:
                running_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
                print(f"   [{i + 1}/{len(samples)}] Running perplexity: {running_ppl:.4f}")

    # Final computation
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)
    min_ppl = min(per_sample_ppl) if per_sample_ppl else float("inf")
    max_ppl = max(per_sample_ppl) if per_sample_ppl else float("inf")
    median_ppl = sorted(per_sample_ppl)[len(per_sample_ppl) // 2] if per_sample_ppl else float("inf")

    result = {
        "perplexity": round(perplexity, 4),
        "min_perplexity": round(min_ppl, 4),
        "max_perplexity": round(max_ppl, 4),
        "median_perplexity": round(median_ppl, 4),
        "total_tokens_evaluated": total_tokens,
        "samples_evaluated": len(per_sample_ppl),
    }

    print(f"\n   📈 Results:")
    print(f"   Perplexity (avg) : {result['perplexity']}")
    print(f"   Perplexity (min) : {result['min_perplexity']}")
    print(f"   Perplexity (max) : {result['max_perplexity']}")
    print(f"   Perplexity (med) : {result['median_perplexity']}")
    print(f"   Tokens evaluated : {result['total_tokens_evaluated']:,}")

    # Quality assessment
    if perplexity < 3.0:
        print("   ✅ Excellent — very low perplexity")
    elif perplexity < 5.0:
        print("   ✅ Good — model is well fine-tuned")
    elif perplexity < 10.0:
        print("   ⚠️  Fair — consider more training or data")
    else:
        print("   ❌ High — model needs improvement")

    return result


# ---------------------------------------------------------------------------
# Metric 2: ROUGE Scores
# ---------------------------------------------------------------------------
def evaluate_rouge(
    model,
    tokenizer,
    eval_data_path: str,
    max_samples: int = 50,
    max_new_tokens: int = 200,
) -> dict:
    """
    Compute ROUGE scores by comparing generated outputs to reference outputs.

    ROUGE-1, ROUGE-2, ROUGE-L measure n-gram overlap with reference responses.
    Higher is better (0.0 to 1.0).

    Args:
        model:          The loaded model.
        tokenizer:      The corresponding tokenizer.
        eval_data_path: Path to evaluation JSONL with reference outputs.
        max_samples:    Maximum samples to evaluate (ROUGE is slower).
        max_new_tokens: Max generation length per sample.

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    """
    print("\n" + "=" * 60)
    print("📊 Evaluating: ROUGE SCORES")
    print("=" * 60)

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("   ⚠️  rouge-score not installed. Run: pip install rouge-score")
        return {"error": "rouge-score not installed"}

    # Load evaluation data
    samples = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                samples.append(record)
            if len(samples) >= max_samples:
                break

    print(f"   Evaluating {len(samples)} samples...")

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    with torch.no_grad():
        for i, sample in enumerate(samples):
            instruction = sample.get("instruction", "You are a helpful AI assistant.")
            user_input = sample.get("input", "")
            reference = sample.get("output", "")

            # Build prompt (without reference)
            if user_input.strip():
                prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{user_input}\n\n"
                    f"### Response:\n"
                )
            else:
                prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Response:\n"
                )

            # Generate
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(model.device)

            input_length = inputs["input_ids"].shape[1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,        # Low temp for consistent evaluation
                do_sample=False,         # Greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            generated = tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
            ).strip()

            # Score
            scores = scorer.score(reference, generated)
            all_scores["rouge1"].append(scores["rouge1"].fmeasure)
            all_scores["rouge2"].append(scores["rouge2"].fmeasure)
            all_scores["rougeL"].append(scores["rougeL"].fmeasure)

            if (i + 1) % 10 == 0:
                avg_r1 = sum(all_scores["rouge1"]) / len(all_scores["rouge1"])
                print(f"   [{i + 1}/{len(samples)}] Running ROUGE-1: {avg_r1:.4f}")

    # Aggregate
    result = {}
    for metric_name, scores_list in all_scores.items():
        if scores_list:
            avg = sum(scores_list) / len(scores_list)
            result[f"{metric_name}_f1"] = round(avg, 4)
        else:
            result[f"{metric_name}_f1"] = 0.0

    result["samples_evaluated"] = len(samples)

    print(f"\n   📈 Results:")
    print(f"   ROUGE-1 (F1) : {result['rouge1_f1']:.4f}")
    print(f"   ROUGE-2 (F1) : {result['rouge2_f1']:.4f}")
    print(f"   ROUGE-L (F1) : {result['rougeL_f1']:.4f}")

    return result


# ---------------------------------------------------------------------------
# Metric 3: Inference Speed
# ---------------------------------------------------------------------------
def evaluate_speed(
    model,
    tokenizer,
    num_runs: int = 5,
    prompt: str = "Explain the concept of machine learning in detail.",
    max_new_tokens: int = 200,
) -> dict:
    """
    Measure inference speed (tokens per second) over multiple runs.

    Target for RTX 4060: ≥ 15 tokens/sec.

    Args:
        model:          The loaded model.
        tokenizer:      The corresponding tokenizer.
        num_runs:       Number of generation runs to average.
        prompt:         Prompt to use for benchmarking.
        max_new_tokens: Tokens to generate per run.

    Returns:
        Dictionary with speed metrics.
    """
    print("\n" + "=" * 60)
    print("📊 Evaluating: INFERENCE SPEED")
    print("=" * 60)

    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    # Warmup run (first run is always slower due to CUDA kernel compilation)
    print("   Warmup run...")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Timed runs
    times = []
    tokens_list = []

    for run in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

        tokens_generated = outputs[0].shape[0] - input_length
        tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

        times.append(elapsed)
        tokens_list.append(tokens_generated)

        print(f"   Run {run + 1}/{num_runs}: {tokens_generated} tokens, "
              f"{elapsed:.2f}s, {tokens_per_sec:.1f} tok/s")

    # Aggregate
    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_list) / len(tokens_list)
    avg_tps = avg_tokens / avg_time if avg_time > 0 else 0
    max_tps = max(t / e for t, e in zip(tokens_list, times) if e > 0)
    min_tps = min(t / e for t, e in zip(tokens_list, times) if e > 0)

    result = {
        "avg_tokens_per_second": round(avg_tps, 1),
        "max_tokens_per_second": round(max_tps, 1),
        "min_tokens_per_second": round(min_tps, 1),
        "avg_generation_time_seconds": round(avg_time, 3),
        "avg_tokens_generated": round(avg_tokens, 0),
        "num_runs": num_runs,
    }

    print(f"\n   📈 Results:")
    print(f"   Avg speed   : {result['avg_tokens_per_second']} tok/s")
    print(f"   Max speed   : {result['max_tokens_per_second']} tok/s")
    print(f"   Min speed   : {result['min_tokens_per_second']} tok/s")
    print(f"   Avg time    : {result['avg_generation_time_seconds']}s")

    if avg_tps >= 15:
        print("   ✅ Target met (≥ 15 tok/s)")
    else:
        print("   ⚠️  Below target (< 15 tok/s)")

    return result


# ---------------------------------------------------------------------------
# Metric 4: Benchmark Prompts (Keyword Matching)
# ---------------------------------------------------------------------------
def evaluate_benchmark(
    model,
    tokenizer,
    benchmark_path: str = DEFAULT_BENCHMARK,
) -> dict:
    """
    Run benchmark prompts and score based on expected keyword presence.

    Each prompt in ``benchmark_prompts.json`` has a list of expected keywords.
    The score for each prompt is the fraction of expected keywords found
    in the generated response.

    Args:
        model:          The loaded model.
        tokenizer:      The corresponding tokenizer.
        benchmark_path: Path to benchmark prompts JSON config.

    Returns:
        Dictionary with per-prompt and aggregate scores.
    """
    print("\n" + "=" * 60)
    print("📊 Evaluating: BENCHMARK PROMPTS")
    print("=" * 60)

    if not os.path.exists(benchmark_path):
        print(f"   ⚠️  Benchmark file not found: {benchmark_path}")
        return {"error": "File not found"}

    with open(benchmark_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    prompts = config.get("prompts", [])
    eval_settings = config.get("evaluation_settings", {})
    temperature = eval_settings.get("temperature", 0.1)
    do_sample = eval_settings.get("do_sample", False)

    print(f"   Running {len(prompts)} benchmark prompts...")

    results_per_prompt = []
    total_score = 0.0

    with torch.no_grad():
        for i, prompt_config in enumerate(prompts):
            prompt_id = prompt_config["id"]
            category = prompt_config["category"]
            prompt = prompt_config["prompt"]
            expected_keywords = prompt_config.get("expected_keywords", [])
            max_tokens = prompt_config.get("max_new_tokens", 256)

            # Generate
            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(model.device)

            input_length = inputs["input_ids"].shape[1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
            ).strip()

            # Score by keyword presence
            response_lower = response.lower()
            found_keywords = [
                kw for kw in expected_keywords
                if kw.lower() in response_lower
            ]
            keyword_score = (
                len(found_keywords) / len(expected_keywords)
                if expected_keywords
                else 1.0
            )

            total_score += keyword_score

            prompt_result = {
                "id": prompt_id,
                "category": category,
                "prompt": prompt,
                "response_preview": response[:200] + "..." if len(response) > 200 else response,
                "expected_keywords": expected_keywords,
                "found_keywords": found_keywords,
                "missing_keywords": [kw for kw in expected_keywords if kw not in found_keywords],
                "keyword_score": round(keyword_score, 3),
            }
            results_per_prompt.append(prompt_result)

            status = "✅" if keyword_score >= 0.6 else "⚠️" if keyword_score >= 0.3 else "❌"
            print(f"   {status} [{prompt_id}] Score: {keyword_score:.0%} "
                  f"({len(found_keywords)}/{len(expected_keywords)} keywords)")

    # Aggregate
    avg_score = total_score / len(prompts) if prompts else 0
    pass_count = sum(1 for r in results_per_prompt if r["keyword_score"] >= 0.6)

    result = {
        "avg_keyword_score": round(avg_score, 4),
        "pass_rate": round(pass_count / len(prompts), 4) if prompts else 0,
        "passed": pass_count,
        "total": len(prompts),
        "per_prompt": results_per_prompt,
    }

    # Per-category summary
    categories = set(r["category"] for r in results_per_prompt)
    category_scores = {}
    for cat in categories:
        cat_results = [r for r in results_per_prompt if r["category"] == cat]
        cat_avg = sum(r["keyword_score"] for r in cat_results) / len(cat_results)
        category_scores[cat] = round(cat_avg, 4)

    result["category_scores"] = category_scores

    print(f"\n   📈 Aggregate Results:")
    print(f"   Avg keyword score : {result['avg_keyword_score']:.1%}")
    print(f"   Pass rate (≥60%)  : {result['pass_rate']:.1%} ({pass_count}/{len(prompts)})")
    for cat, score in category_scores.items():
        print(f"   [{cat}] : {score:.1%}")

    return result


# ---------------------------------------------------------------------------
# Full Evaluation Pipeline
# ---------------------------------------------------------------------------
def run_full_evaluation(
    base_model: str,
    adapter_path: str = None,
    eval_data_path: str = DEFAULT_EVAL_DATA,
    benchmark_path: str = DEFAULT_BENCHMARK,
    max_perplexity_samples: int = DEFAULT_MAX_SAMPLES,
    max_rouge_samples: int = 50,
    speed_runs: int = 5,
    output_path: str = None,
) -> dict:
    """
    Run the complete evaluation suite: perplexity, ROUGE, speed, benchmarks.

    Args:
        base_model:              HuggingFace model ID.
        adapter_path:            Path to LoRA adapter (or None for base model).
        eval_data_path:          Path to evaluation JSONL.
        benchmark_path:          Path to benchmark prompts JSON.
        max_perplexity_samples:  Max samples for perplexity.
        max_rouge_samples:       Max samples for ROUGE.
        speed_runs:              Number of speed benchmark runs.
        output_path:             Where to save the evaluation report JSON.

    Returns:
        Complete evaluation report dictionary.
    """
    print("\n" + "=" * 60)
    print("🎯 QLoRA Model Evaluation Suite")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(base_model, adapter_path)

    report = {
        "model": base_model,
        "adapter": adapter_path,
        "eval_data": eval_data_path,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    # 1. Perplexity
    if eval_data_path and os.path.exists(eval_data_path):
        report["perplexity"] = evaluate_perplexity(
            model, tokenizer, eval_data_path,
            max_samples=max_perplexity_samples,
        )
    else:
        print(f"\n⚠️  Eval data not found: {eval_data_path}")
        print("   Skipping perplexity & ROUGE evaluation.")
        report["perplexity"] = {"skipped": True}

    # 2. ROUGE
    if eval_data_path and os.path.exists(eval_data_path):
        report["rouge"] = evaluate_rouge(
            model, tokenizer, eval_data_path,
            max_samples=max_rouge_samples,
        )
    else:
        report["rouge"] = {"skipped": True}

    # 3. Speed
    report["speed"] = evaluate_speed(
        model, tokenizer,
        num_runs=speed_runs,
    )

    # 4. Benchmark prompts
    report["benchmark"] = evaluate_benchmark(
        model, tokenizer,
        benchmark_path=benchmark_path,
    )

    # --- Save report ---
    if output_path is None:
        output_path = str(PROJECT_ROOT / "models" / "evaluation_report.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Remove per-prompt details for cleaner top-level report
        clean_report = {k: v for k, v in report.items()}
        json.dump(clean_report, f, indent=2, default=str)
    print(f"\n💾 Evaluation report saved → {output_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("📋 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"   Model     : {base_model}")
    print(f"   Adapter   : {adapter_path or 'None'}")
    print(f"   GPU       : {report['gpu']}")
    print()

    if "perplexity" in report and not report["perplexity"].get("skipped"):
        ppl = report["perplexity"].get("perplexity", "N/A")
        print(f"   Perplexity      : {ppl}")

    if "rouge" in report and not report["rouge"].get("skipped"):
        r1 = report["rouge"].get("rouge1_f1", "N/A")
        rL = report["rouge"].get("rougeL_f1", "N/A")
        print(f"   ROUGE-1 (F1)    : {r1}")
        print(f"   ROUGE-L (F1)    : {rL}")

    if "speed" in report:
        tps = report["speed"].get("avg_tokens_per_second", "N/A")
        print(f"   Speed           : {tps} tok/s")

    if "benchmark" in report and not report["benchmark"].get("error"):
        score = report["benchmark"].get("avg_keyword_score", "N/A")
        rate = report["benchmark"].get("pass_rate", "N/A")
        print(f"   Benchmark score : {score}")
        print(f"   Benchmark pass  : {rate}")

    print("=" * 60)
    print("✅ Evaluation complete!")

    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="QLoRA Model Evaluation Suite — Perplexity, ROUGE, Speed, Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation
  python scripts/evaluate.py \\
      --base_model mistralai/Mistral-7B-Instruct-v0.2 \\
      --adapter_path models/my-model/final_adapter \\
      --eval_data data/train.jsonl

  # Benchmark prompts only
  python scripts/evaluate.py \\
      --adapter_path models/my-model/final_adapter \\
      --benchmark_only

  # Custom output
  python scripts/evaluate.py \\
      --output_path results/eval_report.json
        """,
    )

    parser.add_argument("--base_model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--eval_data", type=str, default=DEFAULT_EVAL_DATA,
                        help=f"Evaluation JSONL path (default: {DEFAULT_EVAL_DATA})")
    parser.add_argument("--benchmark_path", type=str, default=DEFAULT_BENCHMARK,
                        help="Path to benchmark prompts JSON")
    parser.add_argument("--max_perplexity_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help=f"Max samples for perplexity (default: {DEFAULT_MAX_SAMPLES})")
    parser.add_argument("--max_rouge_samples", type=int, default=50,
                        help="Max samples for ROUGE (default: 50)")
    parser.add_argument("--speed_runs", type=int, default=5,
                        help="Number of speed benchmark runs (default: 5)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to save evaluation report JSON")
    parser.add_argument("--benchmark_only", action="store_true",
                        help="Run only benchmark prompts (skip perplexity/ROUGE/speed)")

    args = parser.parse_args()

    if args.benchmark_only:
        # Quick benchmark-only mode
        model, tokenizer = load_model(args.base_model, args.adapter_path)
        result = evaluate_benchmark(model, tokenizer, args.benchmark_path)
        print(json.dumps(result, indent=2, default=str))
    else:
        run_full_evaluation(
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            eval_data_path=args.eval_data,
            benchmark_path=args.benchmark_path,
            max_perplexity_samples=args.max_perplexity_samples,
            max_rouge_samples=args.max_rouge_samples,
            speed_runs=args.speed_runs,
            output_path=args.output_path,
        )


if __name__ == "__main__":
    main()
