#!/usr/bin/env python3
"""
prepare_data.py — Data Preparation Utilities for QLoRA Fine-Tuning
==================================================================

Handles:
  • CSV  →  JSONL conversion
  • HuggingFace dataset  →  JSONL conversion
  • Sample JSONL generation (for quick testing)
  • Dataset validation (empty outputs, JSON syntax, token-length checks)

Output format (each line in train.jsonl):
  {"instruction": "...", "input": "...", "output": "..."}

Usage:
  python scripts/prepare_data.py                        # Generate sample data
  python scripts/prepare_data.py --validate data/train.jsonl
  python scripts/prepare_data.py --source csv --csv_path my.csv
  python scripts/prepare_data.py --source hf  --hf_name tatsu-lab/alpaca

Author : Mohamed Noorul Naseem M
Hardware: Lenovo LOQ | RTX 4060 8GB | i7-13650HX | 24GB RAM
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

# ──────────────────────────── Constants ────────────────────────────
DEFAULT_OUTPUT = os.path.join("data", "train.jsonl")
MAX_TOKEN_LENGTH = 2048  # Warn if instruction+output exceeds this
RANDOM_SEED = 42

# ──────────────────────────── Sample Data ─────────────────────────

SAMPLE_DATA = [
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is machine learning?",
        "output": (
            "Machine learning is a branch of artificial intelligence that enables "
            "computers to learn patterns from data without being explicitly programmed. "
            "Instead of following hard-coded rules, ML algorithms build mathematical "
            "models from training data and use them to make predictions or decisions."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "Explain the transformer architecture in simple terms.",
        "output": (
            "A transformer is a deep learning architecture that processes sequences "
            "of data using self-attention mechanisms. Unlike older models (RNNs), "
            "transformers look at all parts of the input simultaneously, which makes "
            "them faster to train and better at capturing long-range relationships. "
            "Key innovations include multi-head attention, positional encoding, and "
            "the encoder-decoder structure (though many modern LLMs use decoder-only)."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is the difference between fine-tuning and training from scratch?",
        "output": (
            "Training from scratch means initializing a model with random weights and "
            "training it on a large dataset — this requires massive compute (8+ GPUs, "
            "weeks of training). Fine-tuning starts from a pre-trained model that "
            "already understands language and adapts it to a specific domain or task "
            "using a much smaller dataset. With techniques like QLoRA, fine-tuning a "
            "7B parameter model is possible on a single consumer GPU with 8GB VRAM."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is QLoRA?",
        "output": (
            "QLoRA (Quantized Low-Rank Adaptation) is a fine-tuning technique that "
            "combines 4-bit quantization with LoRA adapters. The base model is loaded "
            "in 4-bit NormalFloat (NF4) precision, drastically reducing VRAM usage, "
            "while small trainable LoRA matrices are injected into key layers. This "
            "allows fine-tuning of 7B+ parameter models on consumer GPUs like the "
            "RTX 4060 (8GB VRAM) without sacrificing model quality."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "How does LoRA work?",
        "output": (
            "LoRA (Low-Rank Adaptation) works by freezing the original model weights "
            "and injecting small trainable matrices into specific layers. For a weight "
            "matrix W of size d×d, LoRA adds two small matrices: B (d×r) and A (r×d), "
            "where r is the rank (typically 8-64). The update is W' = W + BA. Since "
            "r << d, only a tiny fraction of parameters are trained — typically less "
            "than 0.1% of the total model parameters."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is perplexity and why does it matter?",
        "output": (
            "Perplexity measures how 'surprised' a language model is by test data. "
            "Mathematically, it's the exponentiation of the average negative log-likelihood "
            "across all tokens. A lower perplexity means the model predicts the test data "
            "more accurately. For fine-tuned models, a perplexity below 5.0 on a "
            "domain-specific evaluation set generally indicates good adaptation."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is gradient checkpointing?",
        "output": (
            "Gradient checkpointing is a memory optimization technique that trades "
            "compute time for VRAM savings. Instead of storing all intermediate "
            "activations during the forward pass (needed for backpropagation), it "
            "only saves checkpoints at certain layers and recomputes the rest during "
            "the backward pass. This can reduce VRAM usage by ~30% at the cost of "
            "~20% slower training — a worthwhile tradeoff on memory-constrained GPUs."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What are the advantages of using Mistral-7B for fine-tuning?",
        "output": (
            "Mistral-7B offers several advantages: (1) Sliding Window Attention for "
            "efficient handling of long contexts up to 32K tokens, (2) Grouped-Query "
            "Attention (GQA) for faster inference with 8 KV heads, (3) Strong baseline "
            "performance that outperforms LLaMA-2-13B despite being half the size, "
            "(4) Apache 2.0 license allowing commercial use, and (5) Well-documented "
            "instruct format ([INST] tags) making fine-tuning straightforward."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "Explain the concept of VRAM and why it's important for LLM training.",
        "output": (
            "VRAM (Video Random Access Memory) is the GPU's dedicated memory used to "
            "store model weights, activations, gradients, and optimizer states during "
            "training. For LLM training, VRAM is usually the bottleneck — a 7B model "
            "in full precision (fp32) requires ~28GB just for weights. Techniques like "
            "4-bit quantization (NF4), gradient checkpointing, and paged optimizers "
            "compress this to fit within 8GB on cards like the RTX 4060."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is the role of a tokenizer in NLP?",
        "output": (
            "A tokenizer converts raw text into numerical tokens that a model can "
            "process. It handles vocabulary mapping, subword splitting (BPE or "
            "SentencePiece), and special token insertion. For Mistral-7B, the "
            "tokenizer uses SentencePiece BPE with a vocabulary of ~32,000 tokens. "
            "Proper tokenization is critical — mismatched tokenizers between training "
            "and inference will produce garbage outputs."
        ),
    },
]


# ──────────────────────────── Core Functions ──────────────────────


def generate_sample_data(output_path: str = DEFAULT_OUTPUT, num_samples: int = 10) -> str:
    """
    Generate a sample JSONL file with pre-written QA pairs for quick testing.

    Args:
        output_path: Path to write the JSONL file.
        num_samples: Number of samples to generate (cycles through SAMPLE_DATA).

    Returns:
        Absolute path to the generated file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = []
    for i in range(num_samples):
        samples.append(SAMPLE_DATA[i % len(SAMPLE_DATA)])

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    abs_path = os.path.abspath(output_path)
    console.print(f"\n✅ [bold green]Generated {num_samples} sample(s)[/] → [cyan]{abs_path}[/]\n")
    return abs_path


def csv_to_jsonl(
    csv_path: str,
    output_path: str = DEFAULT_OUTPUT,
    instruction_col: str = "instruction",
    input_col: str = "input",
    output_col: str = "output",
    system_prompt: Optional[str] = None,
) -> str:
    """
    Convert a CSV file to JSONL format for QLoRA training.

    Args:
        csv_path: Path to input CSV.
        output_path: Path for output JSONL.
        instruction_col: Column name containing system/instruction text.
        input_col: Column name containing user input/question.
        output_col: Column name containing expected output/answer.
        system_prompt: If provided, overrides the instruction_col value for all rows.

    Returns:
        Absolute path to the generated file.
    """
    if not os.path.exists(csv_path):
        console.print(f"[bold red]❌ CSV file not found: {csv_path}[/]")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    console.print(f"📂 Loaded CSV with [bold]{len(df)}[/] rows and columns: {list(df.columns)}")

    # Validate required columns exist
    required = [input_col, output_col]
    if not system_prompt:
        required.append(instruction_col)

    missing = [col for col in required if col not in df.columns]
    if missing:
        console.print(f"[bold red]❌ Missing columns: {missing}[/]")
        console.print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    count = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting CSV → JSONL"):
            user_input = str(row[input_col]).strip()
            output_text = str(row[output_col]).strip()

            if not output_text or output_text.lower() == "nan":
                skipped += 1
                continue

            entry = {
                "instruction": system_prompt or str(row.get(instruction_col, "You are a helpful assistant.")).strip(),
                "input": user_input,
                "output": output_text,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    abs_path = os.path.abspath(output_path)
    console.print(f"\n✅ [bold green]Converted {count} rows[/] (skipped {skipped} empty) → [cyan]{abs_path}[/]\n")
    return abs_path


def hf_dataset_to_jsonl(
    dataset_name: str,
    output_path: str = DEFAULT_OUTPUT,
    split: str = "train",
    max_samples: Optional[int] = None,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
) -> str:
    """
    Download a HuggingFace dataset and convert it to JSONL format.

    Args:
        dataset_name: HuggingFace dataset ID (e.g., 'tatsu-lab/alpaca').
        output_path: Path for output JSONL.
        split: Dataset split to use ('train', 'test', etc.).
        max_samples: Maximum samples to include (None = all).
        instruction_field: Field name for instruction.
        input_field: Field name for input.
        output_field: Field name for output.

    Returns:
        Absolute path to the generated file.
    """
    from datasets import load_dataset

    console.print(f"📡 Downloading [bold]{dataset_name}[/] (split={split})...")
    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    console.print(f"   Loaded {len(dataset)} samples. Columns: {dataset.column_names}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(dataset, desc="Converting HF → JSONL"):
            entry = {
                "instruction": str(sample.get(instruction_field, "You are a helpful assistant.")).strip(),
                "input": str(sample.get(input_field, "")).strip(),
                "output": str(sample.get(output_field, "")).strip(),
            }
            if entry["output"]:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    abs_path = os.path.abspath(output_path)
    console.print(f"\n✅ [bold green]Converted {count} samples[/] → [cyan]{abs_path}[/]\n")
    return abs_path


def validate_dataset(filepath: str, max_token_length: int = MAX_TOKEN_LENGTH) -> dict:
    """
    Validate a JSONL dataset for common issues before training.

    Checks:
      • JSON syntax errors
      • Missing required fields (instruction, input, output)
      • Empty output fields
      • Estimated token length exceeding max_token_length

    Args:
        filepath: Path to the JSONL file.
        max_token_length: Warn if approximate token count exceeds this.

    Returns:
        Dictionary with validation statistics.
    """
    if not os.path.exists(filepath):
        console.print(f"[bold red]❌ File not found: {filepath}[/]")
        return {"valid": False, "error": "File not found"}

    stats = {
        "total_lines": 0,
        "valid_lines": 0,
        "json_errors": 0,
        "missing_fields": 0,
        "empty_outputs": 0,
        "long_sequences": 0,
        "output_lengths": [],
    }

    required_fields = {"instruction", "input", "output"}

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            stats["total_lines"] += 1
            line = line.strip()
            if not line:
                continue

            # Check JSON syntax
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                console.print(f"  [red]⚠ Line {i}: Invalid JSON[/]")
                continue

            # Check required fields
            if not required_fields.issubset(entry.keys()):
                stats["missing_fields"] += 1
                missing = required_fields - set(entry.keys())
                console.print(f"  [yellow]⚠ Line {i}: Missing fields: {missing}[/]")
                continue

            # Check empty output
            if not entry["output"].strip():
                stats["empty_outputs"] += 1
                console.print(f"  [yellow]⚠ Line {i}: Empty output field[/]")
                continue

            # Estimate token length (~4 chars per token)
            total_chars = len(entry["instruction"]) + len(entry["input"]) + len(entry["output"])
            approx_tokens = total_chars // 4
            if approx_tokens > max_token_length:
                stats["long_sequences"] += 1

            stats["output_lengths"].append(len(entry["output"].split()))
            stats["valid_lines"] += 1

    # Display results
    table = Table(title=f"📊 Dataset Validation — {filepath}", show_header=True)
    table.add_column("Check", style="cyan", width=25)
    table.add_column("Result", style="bold")

    table.add_row("Total lines", str(stats["total_lines"]))
    table.add_row("Valid lines", f"[green]{stats['valid_lines']}[/]")
    table.add_row("JSON errors", f"[red]{stats['json_errors']}[/]" if stats["json_errors"] else "0")
    table.add_row("Missing fields", f"[red]{stats['missing_fields']}[/]" if stats["missing_fields"] else "0")
    table.add_row("Empty outputs", f"[yellow]{stats['empty_outputs']}[/]" if stats["empty_outputs"] else "0")
    table.add_row("Long sequences (>{max_token_length}t)", str(stats["long_sequences"]))

    if stats["output_lengths"]:
        avg_words = sum(stats["output_lengths"]) / len(stats["output_lengths"])
        color = "green" if avg_words > 50 else "yellow"
        table.add_row("Avg output words", f"[{color}]{avg_words:.1f}[/]")

    console.print(table)

    if stats["json_errors"] or stats["missing_fields"]:
        console.print("\n[bold red]❌ Dataset has errors — fix before training![/]")
    elif stats["empty_outputs"] > 0:
        console.print("\n[bold yellow]⚠ Some empty outputs — consider cleaning.[/]")
    else:
        console.print("\n[bold green]✅ Dataset looks good for training![/]")

    stats["valid"] = stats["json_errors"] == 0 and stats["missing_fields"] == 0
    return stats


# ──────────────────────────── CLI ─────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA Data Preparation Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/prepare_data.py                                  # Generate sample data
  python scripts/prepare_data.py --validate data/train.jsonl      # Validate dataset
  python scripts/prepare_data.py --source csv --csv_path qa.csv   # Convert CSV
  python scripts/prepare_data.py --source hf --hf_name tatsu-lab/alpaca --max_samples 5000
        """,
    )

    parser.add_argument("--source", choices=["sample", "csv", "hf"], default="sample",
                        help="Data source type (default: sample)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--validate", metavar="FILE",
                        help="Validate an existing JSONL dataset and exit")

    # CSV options
    parser.add_argument("--csv_path", help="Path to CSV file")
    parser.add_argument("--instruction_col", default="instruction", help="CSV column for instruction")
    parser.add_argument("--input_col", default="input", help="CSV column for input")
    parser.add_argument("--output_col", default="output", help="CSV column for output")
    parser.add_argument("--system_prompt", help="Override instruction with a fixed system prompt")

    # HuggingFace options
    parser.add_argument("--hf_name", help="HuggingFace dataset name (e.g., tatsu-lab/alpaca)")
    parser.add_argument("--hf_split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--max_samples", type=int, help="Maximum samples to include")

    # General
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of sample entries to generate (default: 10)")

    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    # Validation mode
    if args.validate:
        validate_dataset(args.validate)
        return

    # Data generation / conversion
    if args.source == "sample":
        generate_sample_data(output_path=args.output, num_samples=args.num_samples)

    elif args.source == "csv":
        if not args.csv_path:
            console.print("[bold red]❌ --csv_path is required for CSV source[/]")
            sys.exit(1)
        csv_to_jsonl(
            csv_path=args.csv_path,
            output_path=args.output,
            instruction_col=args.instruction_col,
            input_col=args.input_col,
            output_col=args.output_col,
            system_prompt=args.system_prompt,
        )

    elif args.source == "hf":
        if not args.hf_name:
            console.print("[bold red]❌ --hf_name is required for HuggingFace source[/]")
            sys.exit(1)
        hf_dataset_to_jsonl(
            dataset_name=args.hf_name,
            output_path=args.output,
            split=args.hf_split,
            max_samples=args.max_samples,
        )

    # Validate generated file
    console.print("[dim]Running validation on generated file...[/]")
    validate_dataset(args.output)


if __name__ == "__main__":
    main()
