# 🚀 QLoRA Fine-tuning: 7B LLM on RTX 4060

> Fine-tune Mistral-7B or LLaMA-3-8B on your own data using 4-bit quantization.
> Optimized for **8GB VRAM** — runs fully on your Lenovo LOQ RTX 4060.

---

## 📁 Project Structure

```
qlora_llm/
├── scripts/
│   ├── train.py          ← Main training script
│   ├── inference.py      ← Chat / API deployment
│   ├── evaluate.py       ← Evaluation metrics
│   └── prepare_data.py   ← Data preparation utilities
├── data/
│   └── train.jsonl       ← Your training data (JSONL format)
├── models/               ← Saved LoRA adapters (auto-created)
├── configs/
│   └── benchmark_prompts.json
└── requirements.txt
```

---

## ⚙️ Step 1: Environment Setup

```bash
# 1. Install CUDA Toolkit 12.1+ (if not already installed)
# Check: nvidia-smi

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Verify GPU is detected
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## 📂 Step 2: Prepare Your Data

Your data must be in **JSONL format** (one JSON object per line):

```jsonl
{"instruction": "You are a helpful assistant.", "input": "What is AI?", "output": "AI is..."}
{"instruction": "You are a helpful assistant.", "input": "Explain deep learning.", "output": "Deep learning is..."}
```

### Option A: Use sample data to test

```bash
python scripts/prepare_data.py
# Creates data/train.jsonl with example samples
```

### Option B: Convert your CSV

Edit `prepare_data.py` and call:

```python
csv_to_jsonl(
    csv_path="your_data.csv",
    output_path="data/train.jsonl",
    instruction_col="question",
    output_col="answer",
    system_prompt="You are a domain expert assistant."
)
```

### Option C: Use a HuggingFace dataset

```python
hf_dataset_to_jsonl(
    dataset_name="tatsu-lab/alpaca",   # Or your preferred dataset
    output_path="data/train.jsonl",
    max_samples=10000
)
```

---

## 🏋️ Step 3: Train

### Basic training (Mistral-7B)

```bash
python scripts/train.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset_path data/train.jsonl \
  --output_dir models/my-finetuned-model \
  --epochs 3 \
  --batch_size 2 \
  --lr 2e-4
```

### With Weights & Biases logging

```bash
wandb login   # First time only
python scripts/train.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset_path data/train.jsonl \
  --output_dir models/my-finetuned-model \
  --use_wandb
```

### Alternative models (all fit in 8GB VRAM with QLoRA)

```bash
# LLaMA 3 8B
--model_name meta-llama/Meta-Llama-3-8B-Instruct

# Gemma 7B
--model_name google/gemma-7b-it

# Phi-3 Mini (faster, less VRAM)
--model_name microsoft/Phi-3-mini-4k-instruct
```

### Expected Training Time (RTX 4060):

| Dataset Size   | Epochs | Time      |
| -------------- | ------ | --------- |
| 1,000 samples  | 3      | ~30 min   |
| 10,000 samples | 3      | ~4 hours  |
| 50,000 samples | 3      | ~18 hours |

---

## 💬 Step 4: Run Inference

### Interactive Chat

```bash
python scripts/inference.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter_path models/my-finetuned-model/final_adapter \
  --mode chat
```

### Single prompt

```bash
python scripts/inference.py \
  --adapter_path models/my-finetuned-model/final_adapter \
  --mode single \
  --prompt "Explain transformers in simple terms"
```

### Deploy as REST API

```bash
python scripts/inference.py \
  --adapter_path models/my-finetuned-model/final_adapter \
  --mode api \
  --port 8000

# Test the API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "max_new_tokens": 200}'
```

---

## 📊 Step 5: Evaluate

```bash
python scripts/evaluate.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter_path models/my-finetuned-model/final_adapter \
  --eval_data data/train.jsonl
```

---

## 🎯 VRAM Optimization Tips (for your RTX 4060 8GB)

| Technique                       | VRAM Saving | Applied? |
| ------------------------------- | ----------- | -------- |
| 4-bit quantization (NF4)        | ~50%        | ✅       |
| Double quantization             | ~0.4GB      | ✅       |
| Gradient checkpointing          | ~30%        | ✅       |
| Paged AdamW optimizer           | ~2GB        | ✅       |
| Batch size = 2 + grad accum = 4 | Controlled  | ✅       |
| fp16 mixed precision            | ~50%        | ✅       |

**Expected peak VRAM usage: 6.5–7.5GB out of 8GB ✅**

---

## 🔧 Troubleshooting

**CUDA OOM Error:**

```bash
# Reduce max_seq_len
--max_seq_len 1024  # Instead of 2048

# Reduce batch size
--batch_size 1
```

**Model download slow / fails:**

```bash
# Set HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
```

**bitsandbytes error on Windows:**

```bash
pip install bitsandbytes-windows
```

---

## 🚀 Real-World Application Ideas

1. **Customer Support Bot** — Fine-tune on your company's FAQ/tickets
2. **Medical QA** — Domain-specific clinical Q&A (use medical JSONL datasets)
3. **Code Copilot** — Fine-tune on your codebase's style/patterns
4. **Document Q&A** — Train on internal documents (combine with RAG later)
5. **Tamil/Multilingual** — Fine-tune for regional language support

---

## 📚 Next Steps After Fine-tuning

1. **Quantize for deployment**: Convert to GGUF with llama.cpp for CPU inference
2. **Add RAG**: Combine fine-tuned model with vector DB for document retrieval
3. **RLHF**: Use TRL's PPO trainer to add human preference alignment
4. **Deploy**: Wrap in FastAPI + Docker, deploy to free tier cloud (Hugging Face Spaces)
