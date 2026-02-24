"""
QLoRA LLM Fine-Tuning Platform — Training Pipeline
====================================================
Author : Mohamed Noorul Naseem M
Purpose: Fine-tune 7B+ LLMs using 4-bit QLoRA on consumer GPUs.
Hardware: Optimized for NVIDIA RTX 4060 8GB VRAM

Key Optimizations:
    - NF4 4-bit quantization (BitsAndBytes)
    - Double quantization for additional ~0.4 GB savings
    - Gradient checkpointing (~30% VRAM reduction)
    - Paged AdamW 8-bit optimizer (~2 GB VRAM savings)
    - FP16 mixed precision training
    - Batch size 2 + gradient accumulation = effective batch size 8

Usage:
    python scripts/train.py \\
        --model_name mistralai/Mistral-7B-Instruct-v0.2 \\
        --dataset_path data/train.jsonl \\
        --output_dir models/my-finetuned-model \\
        --epochs 3 --batch_size 2 --lr 2e-4

    # With W&B logging:
    python scripts/train.py \\
        --model_name mistralai/Mistral-7B-Instruct-v0.2 \\
        --dataset_path data/train.jsonl \\
        --output_dir models/my-finetuned-model \\
        --use_wandb
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_DATASET = str(PROJECT_ROOT / "data" / "train.jsonl")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "models" / "qlora-mistral-7b")

# RTX 4060-optimised defaults
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 3
DEFAULT_MAX_SEQ_LEN = 1024          # 2048 possible but tight on 8 GB
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SAVE_STEPS = 100
DEFAULT_LOGGING_STEPS = 10

# LoRA target modules — works for Mistral / LLaMA / Gemma / Phi
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ---------------------------------------------------------------------------
# Data Formatting
# ---------------------------------------------------------------------------
def format_instruction(sample: dict) -> str:
    """
    Format a single training sample into the instruction-following template.

    Uses the Alpaca-style prompt format compatible with Mistral, LLaMA, Gemma:
        ### Instruction:
        {instruction}

        ### Input:
        {input}            ← only if non-empty

        ### Response:
        {output}
    """
    instruction = sample.get("instruction", "You are a helpful AI assistant.")
    user_input = sample.get("input", "")
    output = sample.get("output", "")

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

    return text


def load_jsonl_dataset(path: str) -> Dataset:
    """Load a JSONL file into a HuggingFace ``Dataset``."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    dataset = Dataset.from_list(records)
    print(f"📂 Loaded {len(dataset)} training samples from {path}")
    return dataset


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_quantized_model(model_name: str):
    """
    Load a model in 4-bit NF4 quantization with double quantization.

    Returns:
        (model, tokenizer) tuple with the quantized model and its tokenizer.
    """
    print(f"\n🔧 Loading model: {model_name}")
    print(f"   Quantization : NF4 4-bit + double quantization")
    print(f"   Compute dtype: float16")

    # --- BitsAndBytes 4-bit config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",             # NormalFloat4 for better accuracy
        bnb_4bit_compute_dtype=torch.float16,   # FP16 compute for speed
        bnb_4bit_use_double_quant=True,         # Double quantization saves ~0.4 GB
    )

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Load quantized model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",                      # Auto-map layers to GPU/CPU
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",            # Fallback for compatibility
    )

    # Enable gradient checkpointing for VRAM savings
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Print model size info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model loaded:")
    print(f"   Total parameters   : {total:,}")
    print(f"   Trainable (LoRA)   : {trainable:,}")
    print(f"   Trainable %        : {100 * trainable / total:.4f}%")

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA Adapter Configuration
# ---------------------------------------------------------------------------
def create_lora_config(
    r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
) -> LoraConfig:
    """Create a LoRA configuration targeting all attention + MLP projection layers."""
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )

    print(f"\n🔌 LoRA Config:")
    print(f"   Rank (r)        : {r}")
    print(f"   Alpha           : {lora_alpha}")
    print(f"   Dropout         : {lora_dropout}")
    print(f"   Target modules  : {TARGET_MODULES}")
    return config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    lr: float = DEFAULT_LR,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    use_wandb: bool = False,
    wandb_project: str = "qlora-finetune",
    save_steps: int = DEFAULT_SAVE_STEPS,
    logging_steps: int = DEFAULT_LOGGING_STEPS,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = 42,
):
    """
    Main training function: loads model, prepares data, trains with QLoRA,
    and saves the adapter weights.
    """
    print("=" * 60)
    print("🚀 QLoRA Fine-Tuning Pipeline")
    print("=" * 60)
    print(f"   Model           : {model_name}")
    print(f"   Dataset         : {dataset_path}")
    print(f"   Output          : {output_dir}")
    print(f"   Epochs          : {epochs}")
    print(f"   Batch size      : {batch_size}")
    print(f"   Grad accum      : {grad_accum}")
    print(f"   Effective batch : {batch_size * grad_accum}")
    print(f"   Learning rate   : {lr}")
    print(f"   Max seq len     : {max_seq_len}")
    print(f"   Seed            : {seed}")
    print(f"   W&B logging     : {use_wandb}")
    print("=" * 60)

    # --- GPU check ---
    if not torch.cuda.is_available():
        print("❌ CUDA not available. QLoRA requires a CUDA-capable GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    print(f"\n🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # --- Load model & tokenizer ---
    model, tokenizer = load_quantized_model(model_name)

    # --- Apply LoRA ---
    lora_config = create_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, lora_config)

    # Recount trainable params after LoRA
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n🔌 LoRA applied:")
    print(f"   Trainable params : {trainable:,}")
    print(f"   Trainable %      : {100 * trainable / total:.4f}%")

    # --- Load dataset ---
    dataset = load_jsonl_dataset(dataset_path)

    # --- W&B setup ---
    report_to = "none"
    if use_wandb:
        try:
            import wandb
            report_to = "wandb"
            wandb.init(project=wandb_project, config={
                "model_name": model_name,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_seq_len": max_seq_len,
            })
            print("📊 W&B logging enabled")
        except Exception as e:
            print(f"⚠️  W&B init failed: {e}")
            print("   Continuing without W&B...")
            report_to = "none"

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=True,                                  # FP16 mixed precision
        bf16=False,                                 # RTX 4060 supports bf16 too
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,                         # Keep only last 3 checkpoints
        optim="paged_adamw_8bit",                   # Paged optimizer saves ~2 GB VRAM
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to=report_to,
        seed=seed,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        group_by_length=True,                       # Group similar-length samples
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=max_seq_len,
        formatting_func=format_instruction,
        packing=False,                              # No packing for instruction data
        dataset_text_field=None,
    )

    # --- Train ---
    print("\n🏋️ Starting training...")
    start_time = time.time()

    train_result = trainer.train()

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n⏱️  Training completed in {int(hours)}h {int(mins)}m {int(secs)}s")

    # --- Log metrics ---
    metrics = train_result.metrics
    print(f"\n📊 Training Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    # --- Save final adapter ---
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"\n💾 Final adapter saved → {final_adapter_dir}")

    # --- Save training config ---
    config_path = os.path.join(output_dir, "training_config.json")
    training_config = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "lr": lr,
        "max_seq_len": max_seq_len,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "training_time_seconds": round(elapsed, 2),
        "final_metrics": metrics,
        "gpu": gpu_name,
    }
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2, default=str)
    print(f"📝 Training config saved → {config_path}")

    # --- Cleanup ---
    if use_wandb and report_to == "wandb":
        try:
            wandb.finish()
        except Exception:
            pass

    print("\n✅ Training pipeline complete!")
    print(f"   Adapter location: {final_adapter_dir}")
    print(f"   Next step: python scripts/inference.py --adapter_path {final_adapter_dir}")

    return trainer


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="QLoRA Fine-Tuning Pipeline — 7B LLMs on RTX 4060",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with Mistral-7B
  python scripts/train.py \\
      --model_name mistralai/Mistral-7B-Instruct-v0.2 \\
      --dataset_path data/train.jsonl \\
      --output_dir models/my-model \\
      --epochs 3 --batch_size 2 --lr 2e-4

  # LLaMA-3 8B (needs HF login)
  python scripts/train.py \\
      --model_name meta-llama/Meta-Llama-3-8B-Instruct

  # With W&B logging
  python scripts/train.py --use_wandb

  # Low-VRAM mode
  python scripts/train.py --batch_size 1 --max_seq_len 512
        """,
    )

    # Model & data
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET,
                        help=f"Path to training JSONL (default: {DEFAULT_DATASET})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Per-device batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM,
                        help=f"Gradient accumulation steps (default: {DEFAULT_GRAD_ACCUM})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN,
                        help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LEN})")
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO,
                        help=f"Warmup ratio (default: {DEFAULT_WARMUP_RATIO})")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY,
                        help=f"Weight decay (default: {DEFAULT_WEIGHT_DECAY})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R,
                        help=f"LoRA rank (default: {DEFAULT_LORA_R})")
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA,
                        help=f"LoRA alpha (default: {DEFAULT_LORA_ALPHA})")
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT,
                        help=f"LoRA dropout (default: {DEFAULT_LORA_DROPOUT})")

    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="qlora-finetune",
                        help="W&B project name")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS,
                        help=f"Save checkpoint every N steps (default: {DEFAULT_SAVE_STEPS})")
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_LOGGING_STEPS,
                        help=f"Log every N steps (default: {DEFAULT_LOGGING_STEPS})")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
