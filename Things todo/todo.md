# 📋 TODO — QLoRA LLM Fine-Tuning Platform

> **Project:** QLoRA 7B LLM Fine-Tuning Platform  
> **Author:** Mohamed Noorul Naseem M  
> **Hardware:** Lenovo LOQ | RTX 4060 8GB | i7-13650HX | 24GB RAM  
> **Updated:** February 2026

---

## 🔴 Sprint 1 — Core Pipeline (Week 1–2)

### Environment Setup
- [ ] Install CUDA Toolkit 12.1+ and verify `nvidia-smi` output
- [ ] Create Python 3.10/3.11 virtual environment (`python -m venv venv`)
- [ ] Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- [ ] Verify CUDA available: `python -c "import torch; print(torch.cuda.is_available())"` → must print `True`
- [ ] Install bitsandbytes: `pip install bitsandbytes>=0.43.0`
- [ ] Install all requirements: `pip install -r requirements.txt`
- [ ] Verify GPU name: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
- [ ] Login to HuggingFace: `huggingface-cli login` (needed for gated models like LLaMA-3)
- [ ] Setup W&B: `wandb login` (optional but recommended for tracking)

### Data Preparation (`prepare_data.py`)
- [ ] Run `python scripts/prepare_data.py` to generate sample `data/train.jsonl`
- [ ] Validate sample dataset output — check for empty outputs
- [ ] Collect or identify your real domain data source (CSV / web / PDF / HF dataset)
- [ ] Write domain-specific data conversion using `csv_to_jsonl()` or `hf_dataset_to_jsonl()`
- [ ] Curate minimum 500 high-quality instruction-output pairs for domain
- [ ] Run `validate_dataset("data/train.jsonl")` and fix any issues found
- [ ] Confirm average output length > 50 words (short outputs = poor fine-tuning)
- [ ] Create holdout eval set (10% of data) for unbiased evaluation

### Training Script (`train.py`)
- [ ] Run first test training with sample data (10 steps) to verify no OOM errors
- [ ] Confirm peak VRAM ≤ 7.5GB using `nvidia-smi dmon` during training
- [ ] Tune `--batch_size` (start at 2) and `--max_seq_len` (start at 1024)
- [ ] Run full training on real domain data for 3 epochs
- [ ] Confirm checkpoints are saved at `models/qlora-mistral-7b/checkpoint-N`
- [ ] Verify `final_adapter/` folder exists after training completes

### Evaluation (`evaluate.py`)
- [ ] Run evaluation: `python scripts/evaluate.py`
- [ ] Record baseline perplexity (before fine-tuning) for comparison
- [ ] Record fine-tuned perplexity — target: < 5.0
- [ ] Record inference speed — target: ≥ 15 tokens/sec
- [ ] Run benchmark prompts and check keyword match scores

---

## 🟠 Sprint 2 — Inference & Deployment (Week 3–4)

### Inference Script (`inference.py`)
- [ ] Test interactive chat mode: `python scripts/inference.py --mode chat`
- [ ] Test single prompt mode with a domain-specific question
- [ ] Verify streaming output works (tokens appear progressively)
- [ ] Compare fine-tuned vs base model responses on 5 domain prompts
- [ ] Document response quality observations in `notebooks/model_comparison.md`

### FastAPI Server
- [ ] Start API server: `python scripts/inference.py --mode api --port 8000`
- [ ] Test `/health` endpoint: `curl http://localhost:8000/health`
- [ ] Test `/generate` endpoint with curl (see README for command)
- [ ] Test API with Python `requests` library
- [ ] Add basic authentication header (API key) for security
- [ ] Test error handling — what happens with empty prompt or very long input
- [ ] Document API endpoints in `README.md`

### Monitoring & Logging
- [ ] Open TensorBoard: `tensorboard --logdir models/qlora-mistral-7b`
- [ ] Verify loss curves are decreasing smoothly
- [ ] Check for loss spikes (indicates LR too high or bad data batches)
- [ ] Set up W&B run comparison — compare different hyperparameter configs
- [ ] Log final eval metrics to W&B run summary

---

## 🟡 Sprint 3 — Advanced Features (Week 5–6)

### Model Comparison & Selection
- [ ] Fine-tune alternative base model: `microsoft/Phi-3-mini-4k-instruct` (faster, less VRAM)
- [ ] Compare perplexity across: Mistral-7B vs Phi-3-Mini on same dataset
- [ ] Compare inference speed: Mistral-7B vs Phi-3-Mini
- [ ] Document which model is better for your domain use case

### Hyperparameter Experiments
- [ ] Experiment A: LoRA rank r=8 vs r=16 vs r=32 — compare perplexity
- [ ] Experiment B: LR 1e-4 vs 2e-4 vs 5e-4 — compare convergence
- [ ] Experiment C: max_seq_len 1024 vs 2048 — compare quality vs VRAM
- [ ] Create comparison table in `notebooks/hparam_tuning.md`
- [ ] Pick best hyperparameter config for final production model

### GGUF Export (for CPU deployment)
- [ ] Install llama.cpp: `git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make`
- [ ] Merge LoRA adapter into full model weights (use `merge_and_unload()` script)
- [ ] Convert merged model to GGUF format: `python convert-hf-to-gguf.py ...`
- [ ] Quantize GGUF to Q4_K_M: `./quantize model.gguf model-q4.gguf Q4_K_M`
- [ ] Test GGUF inference on CPU: `./main -m model-q4.gguf -p "Your prompt"`
- [ ] Compare GGUF CPU speed vs GPU VRAM inference

### Benchmark Suite
- [ ] Expand `configs/benchmark_prompts.json` with 20+ domain-specific prompts
- [ ] Add expected keywords for each prompt
- [ ] Run full qualitative benchmark: `python scripts/evaluate.py`
- [ ] Document pass rates in `notebooks/benchmark_results.md`

---

## 🟢 Sprint 4 — Polish & Production (Week 7–8)

### Documentation
- [ ] Update `README.md` with actual perplexity and speed numbers achieved
- [ ] Write `CONTRIBUTING.md` for anyone who wants to extend the project
- [ ] Add docstrings to all functions in all 4 scripts
- [ ] Create `notebooks/quickstart.ipynb` — end-to-end demo notebook
- [ ] Record a short demo video of the chat mode (for portfolio)

### Docker (Optional)
- [ ] Write `Dockerfile` with CUDA base image
- [ ] Build image: `docker build -t qlora-llm .`
- [ ] Test container: `docker run --gpus all qlora-llm python scripts/evaluate.py`
- [ ] Push to Docker Hub or GitHub Container Registry

### HuggingFace Hub Upload (Optional)
- [ ] Create model card: `README.md` in HF format with model description
- [ ] Upload LoRA adapter to HF Hub: `huggingface-cli upload username/model-name ./final_adapter`
- [ ] Add model card tags: `fine-tuned`, `qlora`, `mistral`, your domain
- [ ] Make repository public and share the link

### CI/CD (Optional)
- [ ] Set up GitHub Actions workflow for linting (`flake8`, `black`)
- [ ] Add automated test: quick forward pass sanity check (no full training)
- [ ] Badge your README with CI status

---

## 🔵 Backlog / Future (Post v1.0)

- [ ] **RLHF**: Add PPO reward model using TRL's `PPOTrainer` for preference alignment
- [ ] **RAG Integration**: Connect fine-tuned model to ChromaDB for document retrieval
- [ ] **Multi-GPU**: Implement DeepSpeed ZeRO-3 for multi-GPU training
- [ ] **Web UI**: Build Gradio or Streamlit interface for non-technical users
- [ ] **Multimodal**: Extend to LLaVA for image+text input (vision fine-tuning)
- [ ] **Cloud Deploy**: Deploy to HuggingFace Spaces (free tier) for public demo
- [ ] **Tamil Language**: Fine-tune on Tamil instruction dataset for regional support
- [ ] **Function Calling**: Add tool use / function calling fine-tuning support

---

## 📊 Progress Tracker

| Sprint | Status | Completion |
|--------|--------|------------|
| Sprint 1 — Core Pipeline | 🔄 In Progress | 0% |
| Sprint 2 — Inference & Deployment | ⏳ Planned | 0% |
| Sprint 3 — Advanced Features | ⏳ Planned | 0% |
| Sprint 4 — Polish & Production | ⏳ Planned | 0% |

---

## 🐛 Known Issues & Notes

> Add issues here as you encounter them during development.

- [ ] **Issue tracker**: If bitsandbytes fails, check CUDA path with `python -c "import bitsandbytes; print(bitsandbytes.__version__)"`
- [ ] **Note**: First model download (Mistral-7B) = ~14GB — ensure SSD has 30GB free before starting
- [ ] **Note**: Windows users may need `SET CUDA_VISIBLE_DEVICES=0` before running training

---

## ✅ Completed

> Move items here when done.

- [x] Project scaffolding created (`scripts/`, `data/`, `models/`, `configs/`)
- [x] `train.py` — QLoRA training pipeline written
- [x] `inference.py` — Chat + FastAPI inference server written
- [x] `prepare_data.py` — Data utilities written
- [x] `evaluate.py` — Full evaluation suite written
- [x] `requirements.txt` — All dependencies pinned
- [x] `README.md` — Full documentation written
- [x] PRD, Design Doc, Tech Stack documents created
- [x] `todo.md` — This file
- [x] Git repository initialized
