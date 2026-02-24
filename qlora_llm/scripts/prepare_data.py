"""
QLoRA LLM Fine-Tuning Platform — Data Preparation Utilities
============================================================
Author : Mohamed Noorul Naseem M
Purpose: Convert raw data (CSV, HuggingFace datasets) into JSONL format
         compatible with the QLoRA training pipeline.
Hardware: Optimized for RTX 4060 8GB VRAM

Usage:
    python scripts/prepare_data.py                  # Generate sample data
    python scripts/prepare_data.py --source csv \
        --csv_path data/raw.csv \
        --instruction_col question \
        --output_col answer
    python scripts/prepare_data.py --source hf \
        --hf_dataset tatsu-lab/alpaca \
        --max_samples 10000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "train.jsonl"

SAMPLE_DATA = [
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is machine learning?",
        "output": (
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn and improve from experience without being explicitly "
            "programmed. It focuses on developing algorithms that can access data, "
            "learn from it, and make predictions or decisions. Common approaches "
            "include supervised learning, unsupervised learning, and reinforcement "
            "learning."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "Explain the difference between AI and ML.",
        "output": (
            "Artificial Intelligence (AI) is the broader concept of machines being "
            "able to carry out tasks in a way that we would consider 'smart'. Machine "
            "Learning (ML) is a specific subset of AI that trains a machine how to "
            "learn from data. In other words, all ML is AI, but not all AI is ML. AI "
            "can also include rule-based systems, expert systems, and robotics."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is a neural network?",
        "output": (
            "A neural network is a computing system inspired by the biological neural "
            "networks in the human brain. It consists of interconnected nodes (neurons) "
            "organized in layers: an input layer, one or more hidden layers, and an "
            "output layer. Each connection has a weight that adjusts during training. "
            "Neural networks excel at pattern recognition, classification, and "
            "regression tasks."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is the transformer architecture?",
        "output": (
            "The transformer architecture, introduced in the paper 'Attention Is All "
            "You Need' (2017), is a deep learning model that uses self-attention "
            "mechanisms to process input sequences in parallel rather than sequentially "
            "like RNNs. It consists of an encoder-decoder structure with multi-head "
            "attention layers, feed-forward networks, and residual connections. "
            "Transformers are the foundation for models like BERT, GPT, and LLaMA."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is fine-tuning in the context of LLMs?",
        "output": (
            "Fine-tuning is the process of taking a pre-trained large language model "
            "and further training it on a smaller, domain-specific dataset. This "
            "allows the model to adapt its knowledge to a particular task or domain "
            "while retaining its general language understanding. Techniques like LoRA "
            "and QLoRA make fine-tuning efficient by only updating a small number of "
            "adapter parameters rather than all model weights."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is QLoRA?",
        "output": (
            "QLoRA (Quantized Low-Rank Adaptation) is a parameter-efficient fine-tuning "
            "technique that combines 4-bit quantization with LoRA adapters. It loads the "
            "base model in 4-bit precision (NF4 data type) to drastically reduce memory "
            "usage, then trains small rank-decomposition matrices (LoRA adapters) on top. "
            "This allows fine-tuning 7B+ parameter models on consumer GPUs with as little "
            "as 8GB VRAM, making LLM customization accessible to individual developers."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is gradient checkpointing?",
        "output": (
            "Gradient checkpointing is a memory optimization technique that trades "
            "computation for memory. Instead of storing all intermediate activations "
            "during the forward pass, it only stores a subset and recomputes the rest "
            "during the backward pass. This can reduce memory usage by up to 30-50% "
            "at the cost of approximately 20% more computation time, making it "
            "essential for training large models on limited VRAM."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "Explain the concept of attention in neural networks.",
        "output": (
            "Attention is a mechanism that allows neural networks to focus on relevant "
            "parts of the input when producing output. In self-attention, each element "
            "in a sequence computes a weighted sum over all other elements, where the "
            "weights are learned based on content similarity (query-key dot product). "
            "Multi-head attention runs multiple attention computations in parallel, "
            "allowing the model to attend to different aspects of the input "
            "simultaneously. Attention is the core building block of transformers."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is RLHF and why is it used?",
        "output": (
            "RLHF (Reinforcement Learning from Human Feedback) is a technique used to "
            "align language models with human preferences. It involves three stages: "
            "(1) supervised fine-tuning on demonstrations, (2) training a reward model "
            "on human preference comparisons, and (3) optimizing the language model "
            "using PPO (Proximal Policy Optimization) against the reward model. RLHF "
            "helps make models more helpful, harmless, and honest."
        ),
    },
    {
        "instruction": "You are a helpful AI assistant.",
        "input": "What is RAG (Retrieval-Augmented Generation)?",
        "output": (
            "RAG (Retrieval-Augmented Generation) combines a language model with an "
            "external knowledge retrieval system. When a query is received, relevant "
            "documents are retrieved from a vector database (like ChromaDB or FAISS) "
            "and provided as context to the LLM. This approach reduces hallucinations, "
            "keeps responses grounded in factual data, and allows the model to access "
            "up-to-date information without retraining."
        ),
    },
]


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------
def generate_sample_data(output_path: str = None, num_copies: int = 5) -> str:
    """
    Generate a sample JSONL training dataset with AI/ML-focused Q&A pairs.

    Args:
        output_path: Path to save the generated JSONL file.
                     Defaults to ``data/train.jsonl``.
        num_copies:  How many times to duplicate the sample set to create
                     a larger training file (useful for testing).

    Returns:
        Absolute path of the generated file.
    """
    output_path = Path(output_path or DEFAULT_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records_written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_copies):
            for sample in SAMPLE_DATA:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                records_written += 1

    print(f"✅ Generated {records_written} samples → {output_path}")
    return str(output_path)


def csv_to_jsonl(
    csv_path: str,
    output_path: str = None,
    instruction_col: str = "instruction",
    input_col: str = "input",
    output_col: str = "output",
    system_prompt: str = "You are a helpful AI assistant.",
) -> str:
    """
    Convert a CSV file to JSONL format for QLoRA training.

    Args:
        csv_path:         Path to the source CSV file.
        output_path:      Where to save the JSONL. Defaults to ``data/train.jsonl``.
        instruction_col:  Column name for the system instruction / question.
        input_col:        Column name for optional user context input.
        output_col:       Column name for the desired response.
        system_prompt:    Default system prompt if the CSV has no instruction column.

    Returns:
        Absolute path of the generated JSONL file.
    """
    output_path = Path(output_path or DEFAULT_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"📂 Loaded CSV: {len(df)} rows, columns = {list(df.columns)}")

    records_written = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            instruction = (
                str(row[instruction_col])
                if instruction_col in df.columns and pd.notna(row.get(instruction_col))
                else system_prompt
            )
            user_input = (
                str(row[input_col])
                if input_col in df.columns and pd.notna(row.get(input_col))
                else ""
            )
            output = (
                str(row[output_col])
                if output_col in df.columns and pd.notna(row.get(output_col))
                else ""
            )

            # Skip rows with empty output
            if not output.strip():
                skipped += 1
                continue

            record = {
                "instruction": instruction,
                "input": user_input,
                "output": output,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    print(f"✅ Converted {records_written} rows → {output_path}")
    if skipped:
        print(f"⚠️  Skipped {skipped} rows with empty output")
    return str(output_path)


def hf_dataset_to_jsonl(
    dataset_name: str,
    output_path: str = None,
    split: str = "train",
    max_samples: int = None,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    system_prompt: str = "You are a helpful AI assistant.",
) -> str:
    """
    Download a HuggingFace dataset and convert it to JSONL format.

    Args:
        dataset_name:      HuggingFace dataset identifier (e.g. ``tatsu-lab/alpaca``).
        output_path:       Where to save the JSONL. Defaults to ``data/train.jsonl``.
        split:             Which split to use (``train``, ``test``, ``validation``).
        max_samples:       Maximum number of samples to include.
        instruction_field: Field name for the instruction column (dataset-dependent).
        input_field:       Field name for the user input column.
        output_field:      Field name for the output/response column.
        system_prompt:     Fallback system prompt.

    Returns:
        Absolute path of the generated JSONL file.
    """
    from datasets import load_dataset

    output_path = Path(output_path or DEFAULT_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"⬇️  Downloading dataset: {dataset_name} (split={split})...")
    dataset = load_dataset(dataset_name, split=split)

    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        print(f"🔀 Selected {max_samples} random samples")

    records_written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            instruction = sample.get(instruction_field, system_prompt) or system_prompt
            user_input = sample.get(input_field, "") or ""
            output = sample.get(output_field, "") or ""

            if not output.strip():
                continue

            record = {
                "instruction": instruction,
                "input": user_input,
                "output": output,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    print(f"✅ Downloaded & converted {records_written} samples → {output_path}")
    return str(output_path)


def validate_dataset(jsonl_path: str) -> dict:
    """
    Validate a JSONL dataset for training readiness.

    Checks for:
    - Valid JSON on every line
    - Required fields (instruction, input, output)
    - Empty outputs
    - Output length statistics

    Args:
        jsonl_path: Path to the JSONL file to validate.

    Returns:
        Dictionary with validation statistics.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"❌ File not found: {jsonl_path}")
        return {"valid": False, "error": "File not found"}

    stats = {
        "total_lines": 0,
        "valid_records": 0,
        "empty_outputs": 0,
        "missing_fields": 0,
        "parse_errors": 0,
        "avg_output_words": 0,
        "min_output_words": float("inf"),
        "max_output_words": 0,
    }

    output_word_counts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stats["total_lines"] += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_errors"] += 1
                print(f"  ⚠️  Line {line_num}: Invalid JSON")
                continue

            # Check required fields
            required = ["instruction", "output"]
            if not all(k in record for k in required):
                stats["missing_fields"] += 1
                missing = [k for k in required if k not in record]
                print(f"  ⚠️  Line {line_num}: Missing fields: {missing}")
                continue

            # Check empty output
            output_text = record.get("output", "").strip()
            if not output_text:
                stats["empty_outputs"] += 1
                print(f"  ⚠️  Line {line_num}: Empty output")
                continue

            word_count = len(output_text.split())
            output_word_counts.append(word_count)
            stats["valid_records"] += 1

    # Compute statistics
    if output_word_counts:
        stats["avg_output_words"] = round(
            sum(output_word_counts) / len(output_word_counts), 1
        )
        stats["min_output_words"] = min(output_word_counts)
        stats["max_output_words"] = max(output_word_counts)
    else:
        stats["min_output_words"] = 0

    # Print summary
    print("\n" + "=" * 50)
    print("📊 Dataset Validation Report")
    print("=" * 50)
    print(f"  File           : {jsonl_path}")
    print(f"  Total lines    : {stats['total_lines']}")
    print(f"  Valid records  : {stats['valid_records']}")
    print(f"  Empty outputs  : {stats['empty_outputs']}")
    print(f"  Missing fields : {stats['missing_fields']}")
    print(f"  Parse errors   : {stats['parse_errors']}")
    print(f"  Avg output len : {stats['avg_output_words']} words")
    print(f"  Min output len : {stats['min_output_words']} words")
    print(f"  Max output len : {stats['max_output_words']} words")

    if stats["avg_output_words"] < 50:
        print("\n⚠️  WARNING: Average output length < 50 words.")
        print("   Short outputs can lead to poor fine-tuning results.")
        print("   Consider enriching your training data with longer responses.")

    if stats["valid_records"] < 100:
        print("\n⚠️  WARNING: Less than 100 valid records.")
        print("   At least 500+ records recommended for meaningful fine-tuning.")

    is_valid = (
        stats["valid_records"] > 0
        and stats["parse_errors"] == 0
        and stats["empty_outputs"] == 0
    )
    stats["valid"] = is_valid

    if is_valid:
        print("\n✅ Dataset is READY for training!")
    else:
        print("\n❌ Dataset has issues — please fix before training.")

    return stats


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="QLoRA Data Preparation Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data
  python scripts/prepare_data.py

  # Convert CSV to JSONL
  python scripts/prepare_data.py --source csv \\
      --csv_path data/raw.csv \\
      --instruction_col question \\
      --output_col answer

  # Download HuggingFace dataset
  python scripts/prepare_data.py --source hf \\
      --hf_dataset tatsu-lab/alpaca \\
      --max_samples 10000

  # Validate existing dataset
  python scripts/prepare_data.py --validate data/train.jsonl
        """,
    )

    parser.add_argument(
        "--source",
        choices=["sample", "csv", "hf"],
        default="sample",
        help="Data source type (default: sample)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSONL path (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to input CSV file (for --source csv)",
    )
    parser.add_argument(
        "--instruction_col",
        type=str,
        default="instruction",
        help="CSV column name for instructions",
    )
    parser.add_argument(
        "--input_col",
        type=str,
        default="input",
        help="CSV column name for user input",
    )
    parser.add_argument(
        "--output_col",
        type=str,
        default="output",
        help="CSV column name for desired output",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (for --source hf)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to extract from HF dataset",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="Default system prompt for the dataset",
    )
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Path to JSONL file to validate",
    )
    parser.add_argument(
        "--num_copies",
        type=int,
        default=5,
        help="Number of times to duplicate sample data (for --source sample)",
    )

    args = parser.parse_args()

    # Validate mode
    if args.validate:
        validate_dataset(args.validate)
        return

    # Generate / convert data
    if args.source == "sample":
        generate_sample_data(
            output_path=args.output_path,
            num_copies=args.num_copies,
        )
    elif args.source == "csv":
        if not args.csv_path:
            print("❌ --csv_path is required when --source csv")
            sys.exit(1)
        csv_to_jsonl(
            csv_path=args.csv_path,
            output_path=args.output_path,
            instruction_col=args.instruction_col,
            input_col=args.input_col,
            output_col=args.output_col,
            system_prompt=args.system_prompt,
        )
    elif args.source == "hf":
        if not args.hf_dataset:
            print("❌ --hf_dataset is required when --source hf")
            sys.exit(1)
        hf_dataset_to_jsonl(
            dataset_name=args.hf_dataset,
            output_path=args.output_path,
            max_samples=args.max_samples,
            system_prompt=args.system_prompt,
        )

    # Auto-validate after generation
    final_path = args.output_path or str(DEFAULT_OUTPUT)
    if os.path.exists(final_path):
        print("\n--- Running auto-validation ---")
        validate_dataset(final_path)


if __name__ == "__main__":
    main()
