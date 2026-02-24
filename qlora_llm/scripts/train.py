#!/usr/bin/env python3
"""
train.py — QLoRA Training Pipeline for 7B+ LLMs
=================================================

End-to-end fine-tuning pipeline using:
  • 4-bit NF4 quantization via bitsandbytes
  • LoRA adapters via PEFT (r=16, α=32)
  • SFTTrainer from TRL for supervised fine-tuning
  • Paged AdamW optimizer (offloads optimizer states to CPU)
  • Gradient checkpointing (~30% VRAM saved)
  • fp16 mixed precision training

Optimized for RTX 4060 8GB VRAM — peak usage ~6.5–7.5 GB.

Usage:
  python scripts/train.py \\
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \\
    --dataset_path data/train.jsonl \\
    --output_dir models/my-finetuned-model \\
    --epochs 3 \\
    --batch_size 2 \\
    --lr 2e-4

Author : Mohamed Noorul Naseem M
Hardware: Lenovo LOQ | RTX 4060 8GB | i7-13650HX | 24GB RAM
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer

console = Console()

# ──────────────────────────── Defaults ────────────────────────────
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_DATASET = os.path.join("data", "train.jsonl")
DEFAULT_OUTPUT = os.path.join("models", "qlora-mistral-7b")

# LoRA Hyperparameters (Design Doc §4.2)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ──────────────────────────── Utilities ───────────────────────────


def print_gpu_info():
    """Display GPU information and current VRAM usage."""
    if not torch.cuda.is_available():
        console.print("[bold red]❌ CUDA is NOT available! Training requires a CUDA GPU.[/]")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    current_mem = torch.cuda.memory_allocated(0) / 1e9
    reserved_mem = torch.cuda.memory_reserved(0) / 1e9

    table = Table(title="🖥️ GPU Information", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("GPU", gpu_name)
    table.add_row("Total VRAM", f"{total_mem:.1f} GB")
    table.add_row("Allocated", f"{current_mem:.2f} GB")
    table.add_row("Reserved", f"{reserved_mem:.2f} GB")
    table.add_row("CUDA version", torch.version.cuda or "N/A")
    table.add_row("PyTorch version", torch.__version__)
    console.print(table)


def format_prompt(sample: dict) -> str:
    """
    Format a dataset sample into Mistral instruct template.

    Template:  <s>[INST] {instruction}\n\n{input} [/INST] {output}</s>
    """
    instruction = sample.get("instruction", "You are a helpful assistant.")
    user_input = sample.get("input", "")
    output = sample.get("output", "")

    prompt = f"<s>[INST] {instruction}\n\n{user_input} [/INST] {output}</s>"
    return prompt


def load_jsonl_dataset(filepath: str, test_size: float = 0.1, seed: int = 42):
    """
    Load a JSONL file into a HuggingFace DatasetDict with train/eval split.

    Args:
        filepath: Path to the JSONL file.
        test_size: Fraction for evaluation split (default: 10%).
        seed: Random seed for reproducibility.

    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    if not os.path.exists(filepath):
        console.print(f"[bold red]❌ Dataset not found: {filepath}[/]")
        console.print("   Run `python scripts/prepare_data.py` first to generate sample data.")
        sys.exit(1)

    dataset = load_dataset("json", data_files=filepath, split="train")
    console.print(f"📂 Loaded [bold]{len(dataset)}[/] samples from [cyan]{filepath}[/]")

    # Split into train / eval
    dataset_split = dataset.train_test_split(test_size=test_size, seed=seed)
    console.print(
        f"   Train: [green]{len(dataset_split['train'])}[/] | "
        f"Eval: [yellow]{len(dataset_split['test'])}[/] "
        f"(split: {1 - test_size:.0%} / {test_size:.0%})"
    )
    return dataset_split


# ──────────────────────────── Main Training ───────────────────────


def train(args):
    """Execute the full QLoRA training pipeline."""
    start_time = time.time()

    console.print(Panel(
        "[bold cyan]🚀 QLoRA Fine-Tuning Pipeline[/]\n"
        f"Model: [white]{args.model_name}[/]\n"
        f"Dataset: [white]{args.dataset_path}[/]\n"
        f"Output: [white]{args.output_dir}[/]",
        title="Training Configuration",
        border_style="cyan",
    ))

    print_gpu_info()

    # ── 1. Load dataset ──
    console.print("\n[bold]📂 Step 1: Loading dataset...[/]")
    dataset = load_jsonl_dataset(args.dataset_path, seed=args.seed)

    # ── 2. Configure quantization ──
    console.print("\n[bold]⚙️ Step 2: Configuring 4-bit quantization...[/]")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Saves ~0.4GB extra VRAM
    )

    console.print("   ✅ NF4 4-bit quantization + double quantization enabled")

    # ── 3. Load tokenizer ──
    console.print("\n[bold]📝 Step 3: Loading tokenizer...[/]")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    console.print(f"   Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # ── 4. Load base model ──
    console.print(f"\n[bold]🧠 Step 4: Loading base model ({args.model_name})...[/]")
    console.print("   [dim]This will download ~14GB on first run...[/]")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Enable gradient checkpointing (saves ~30% VRAM)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Print VRAM after model load
    mem_used = torch.cuda.memory_allocated(0) / 1e9
    console.print(f"   ✅ Model loaded — VRAM used: [bold]{mem_used:.2f} GB[/]")

    # ── 5. Configure LoRA ──
    console.print("\n[bold]🔧 Step 5: Injecting LoRA adapters...[/]")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = trainable_params / total_params * 100

    table = Table(title="🔧 LoRA Configuration", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Rank (r)", str(args.lora_r))
    table.add_row("Alpha (α)", str(args.lora_alpha))
    table.add_row("Dropout", str(LORA_DROPOUT))
    table.add_row("Target modules", ", ".join(LORA_TARGET_MODULES))
    table.add_row("Trainable params", f"{trainable_params:,}")
    table.add_row("Total params", f"{total_params:,}")
    table.add_row("Trainable %", f"{trainable_pct:.4f}%")
    console.print(table)

    # ── 6. Training arguments ──
    console.print("\n[bold]🏋️ Step 6: Configuring training arguments...[/]")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=-1,
        group_by_length=True,
        report_to="wandb" if args.use_wandb else "tensorboard",
        seed=args.seed,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # ── 7. Initialize trainer ──
    console.print("\n[bold]🚂 Step 7: Initializing SFTTrainer...[/]")

    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.patience,
            early_stopping_threshold=0.01,
        ))
        console.print(f"   Early stopping enabled (patience={args.patience})")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        formatting_func=format_prompt,
        max_seq_length=args.max_seq_len,
        callbacks=callbacks,
    )

    # ── 8. Train! ──
    console.print("\n" + "=" * 60)
    console.print("[bold green]🚀 Starting training...[/]")
    console.print("=" * 60 + "\n")

    mem_before = torch.cuda.memory_allocated(0) / 1e9
    trainer.train()
    mem_after = torch.cuda.max_memory_allocated(0) / 1e9

    # ── 9. Save the final adapter ──
    adapter_save_path = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    console.print(f"\n✅ [bold green]Final adapter saved[/] → [cyan]{adapter_save_path}[/]")

    # ── 10. Training summary ──
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    summary = Table(title="📊 Training Summary", show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="bold")
    summary.add_row("Total time", f"{hours}h {minutes}m {seconds}s")
    summary.add_row("Peak VRAM", f"{mem_after:.2f} GB")
    summary.add_row("VRAM before training", f"{mem_before:.2f} GB")
    summary.add_row("Final train loss", f"{trainer.state.log_history[-1].get('train_loss', 'N/A')}")
    summary.add_row("Adapter path", adapter_save_path)
    summary.add_row("Epochs", str(args.epochs))
    summary.add_row("Batch size", f"{args.batch_size} (× {args.grad_accum} accum)")
    summary.add_row("Learning rate", str(args.lr))
    console.print(summary)

    # Save config as JSON
    config_path = os.path.join(args.output_dir, "training_config.json")
    config = {
        "model_name": args.model_name,
        "dataset_path": args.dataset_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "max_seq_len": args.max_seq_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "peak_vram_gb": round(mem_after, 2),
        "training_time_seconds": round(elapsed, 2),
        "seed": args.seed,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    console.print(f"   Config saved → [cyan]{config_path}[/]")

    console.print("\n[bold green]✅ Training complete![/] 🎉\n")


# ──────────────────────────── CLI ─────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA Fine-Tuning Pipeline — Train 7B LLMs on RTX 4060",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core arguments
    parser.add_argument("--model_name", default=DEFAULT_MODEL,
                        help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET,
                        help=f"Path to training JSONL (default: {DEFAULT_DATASET})")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT,
                        help=f"Output directory for checkpoints (default: {DEFAULT_OUTPUT})")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size (default: 2)")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=LORA_R,
                        help=f"LoRA rank (default: {LORA_R})")
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA,
                        help=f"LoRA alpha (default: {LORA_ALPHA})")

    # Training options
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--early_stopping", action="store_true", default=True,
                        help="Enable early stopping (default: True)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (default: 3)")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging (default: TensorBoard)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
