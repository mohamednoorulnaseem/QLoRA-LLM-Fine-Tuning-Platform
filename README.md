<div align="center">
  <h1>Enterprise-Grade QLoRA LLM Fine-Tuning Platform</h1>
  <p><strong>End-to-end MLOps pipeline for fine-tuning 7B+ parameter models on consumer hardware.</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
  [![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-ffcc00.svg)](https://huggingface.co/)
  [![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://docker.com)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 📌 Overview

Training large language models (LLMs) traditionally requires massive GPU clusters. This platform democratizes LLM fine-tuning by providing a fully containerized, reproducible pipeline that fine-tunes state-of-the-art models (like Mistral-7B and LLaMA-3-8B) on proprietary datasets—all strictly constrained within **8GB of VRAM**.

By leveraging **Quantized Low-Rank Adaptation (QLoRA)**, 4-bit NormalFloat quantization, and aggressive memory optimization techniques, this architecture delivers enterprise-quality fine-tuning on standard consumer hardware.

### Key Capabilities
* **Efficient Fine-Tuning:** Full SFT (Supervised Fine-Tuning) pipeline optimized for ≤7.5GB VRAM.
* **Production Deployment:** Ships with a containerized FastAPI REST server and Docker Compose setup for immediate cloud or edge deployment.
* **Extensive Evaluation:** Built-in benchmarking suite, Perplexity scoring, and ROUGE metrics.
* **Edge-Ready Export:** Automated LoRA merging and GGUF export scripts for `llama.cpp` CPU inference.
* **Data Engineering:** Automated curation, formatting, and JSONL conversion pipelines for HuggingFace datasets and raw CSVs.

---

## 🏗️ Architecture & System Design

```text
qlora_llm/
├── configs/
│   └── benchmark_prompts.json   # Automated evaluation benchmarks
├── data/
│   └── train.jsonl              # Compiled domain-specific training data
├── models/                      # Checkpoints and merged adapters
├── scripts/
│   ├── prepare_data.py          # ETL pipelines for dataset curation
│   ├── train.py                 # QLoRA training loop with W&B integration
│   ├── evaluate.py              # Quantitative and qualitative model scoring
│   ├── inference.py             # CLI Chat & FastAPI REST Server
│   ├── merge_datasets.py        # Dataset blending utility
│   └── export_gguf.py           # Base model + adapter merging for edge deployment
├── Dockerfile                   # Production container definition
├── docker-compose.yml           # Multi-container orchestration
└── requirements.txt             # Pinned dependency matrix
```

---

## 🚀 Quick Start Guide

### 1. Environment Initialization
Ensure CUDA Toolkit 12.1+ is installed on the host system.

```bash
# Clone the repository
git clone https://github.com/yourusername/QLoRA-LLM-Fine-Tuning-Platform.git
cd QLoRA-LLM-Fine-Tuning-Platform

# Create isolated environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Dataset Curation
Process your proprietary data or pull from HuggingFace.

```bash
# Example: Convert a HuggingFace dataset to properly formatted JSONL
python scripts/prepare_data.py --source hf --hf_name tatsu-lab/alpaca --max_samples 5000
```

### 3. Model Training
Initiate the memory-optimized training loop. Configurations are pre-tuned for stability and convergence on 8GB GPUs.

```bash
python scripts/train.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset_path data/train.jsonl \
  --output_dir models/mistral-finetuned \
  --epochs 3 \
  --batch_size 1 \
  --grad_accum 16 \
  --max_seq_len 512
```

### 4. Production Deployment (Docker API)
Serve your fine-tuned model via a high-performance REST API.

```bash
# Spin up the FastAPI server via Docker Compose
docker-compose up -d

# Query the endpoint
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the architecture of Transformers.", "max_new_tokens": 200}'
```

---

## 🧠 VRAM Optimization Engineering

Achieving 7B parameter fine-tuning on 8GB VRAM requires aggressive, multi-layered memory optimization. This platform implements:

| Optimization Layer | Implementation | VRAM Impact |
| :--- | :--- | :--- |
| **Quantization** | 4-bit NormalFloat (NF4) via `bitsandbytes` | **-50% Base Memory** |
| **Nested Quantization** | Double Quantization (quantizing the quantization constants) | **-0.4 GB** |
| **Gradient Optimization** | Gradient Checkpointing (Activation Recomputation) | **-30% Training Memory** |
| **Optimizer Offloading** | Paged AdamW 32-bit (CPU Offloading) | **-2.0 GB** |
| **Precision Scaling** | BF16/FP16 Mixed Precision Training | **Increased Throughput** |

**Result:** Peak VRAM strictly capped between **6.5GB - 7.5GB**.

---

## 📊 Evaluation & Edge Export

**Benchmarking:**
Run comprehensive perplexity and prompt-based evaluations:
```bash
python scripts/evaluate.py --adapter_path models/mistral-finetuned/final_adapter
```

**Edge Deployment (GGUF):**
Merge LoRA adapters into the base weights for edge-device deployment using `llama.cpp`.
```bash
python scripts/export_gguf.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter_path models/mistral-finetuned/final_adapter \
  --output_path models/merged-model
```

---

## 🤝 Contributing
Contributions are welcome. Please ensure your code passes standard `flake8` and `black` formatting checks before submitting a PR. For major architectural changes, please open an issue first to discuss the proposed modifications.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
