**DESIGN DOCUMENT**

**QLoRA LLM Fine-Tuning Platform**

*System Architecture, Component Design & Data Flow*

**1. System Architecture Overview**

The platform is designed as a linear pipeline with four distinct phases:
Data Preparation → Model Training → Evaluation → Deployment. Each phase
is implemented as an independent module with clean interfaces, allowing
components to be replaced or upgraded without affecting others.

**1.1 High-Level Architecture**

+-----------------+-----------------+-----------------+-----------------+
| **📂 Data       | **🏋️ Training   | **📊 Eval       | **🚀 Serving    |
| Layer**         | Layer**         | Layer**         | Layer**         |
|                 |                 |                 |                 |
| Raw CSV/JSONL → | 4-bit Quantized | Perplexity +    | Merged Model →  |
| Validated JSONL | Model + LoRA    | ROUGE + Speed   | FastAPI Server  |
| → HuggingFace   | Adapters →      | Benchmark →     | → REST          |
| Dataset         | SFTTrainer →    | Eval Report     | Endpoints       |
|                 | Saved           |                 |                 |
|                 | Checkpoint      |                 |                 |
+-----------------+-----------------+-----------------+-----------------+

**2. Component Design**

**2.1 prepare_data.py --- Data Layer**

+-----------------------------------------------------------------------+
| **Data Preparation Module**                                           |
+-----------------------------------------------------------------------+
| -   Input: Raw CSV files, HuggingFace dataset names, or existing      |
|     JSONL                                                             |
|                                                                       |
| -   Output: Validated train.jsonl in instruction-input-output format  |
|                                                                       |
| -   Validation: Checks for empty outputs, token length overflow, JSON |
|     syntax errors                                                     |
|                                                                       |
| -   Formatting: Applies Mistral \[INST\] template wrapping            |
|     automatically                                                     |
|                                                                       |
| -   HF Integration: Downloads and converts datasets via               |
|     load_dataset()                                                    |
+-----------------------------------------------------------------------+

**Data Format Specification**

Input format --- each line in train.jsonl must be a valid JSON object:

> {\"instruction\": \"System prompt here\", \"input\": \"User query\",
> \"output\": \"Expected response\"}

After formatting, the trainer receives the complete Mistral instruct
template:

> \<s\>\[INST\] {instruction}\\n\\n{input} \[/INST\] {output}\</s\>

**2.2 train.py --- Training Layer**

+-----------------------------------------------------------------------+
| **QLoRA Training Module**                                             |
+-----------------------------------------------------------------------+
| -   BitsAndBytesConfig: NF4 4-bit quantization + double quantization  |
|     (saves \~0.4GB VRAM)                                              |
|                                                                       |
| -   LoraConfig: r=16, alpha=32,                                       |
|     targe                                                             |
| t_modules=\[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj\] |
|                                                                       |
| -   SFTTrainer: Supervised fine-tuning with group_by_length for       |
|     batching efficiency                                               |
|                                                                       |
| -   EarlyStoppingCallback: patience=3, monitors eval_loss to prevent  |
|     overfitting                                                       |
|                                                                       |
| -   Paged AdamW: Off-GPU optimizer states --- critical for 8GB VRAM   |
|     budget                                                            |
|                                                                       |
| -   Gradient Checkpointing: Trades compute for memory, saves \~30%    |
|     VRAM                                                              |
|                                                                       |
| -   Checkpoint frequency: every 50 steps (save_strategy=\'steps\')    |
+-----------------------------------------------------------------------+

**VRAM Budget Breakdown**

  ------------------------------ --------------- -------------------------
  **Component**                  **VRAM          **Notes**
                                 (approx.)**     

  7B Model (4-bit NF4 quantized) \~3.5 GB        Base model weights

  LoRA Adapter Weights (r=16)    \~0.2 GB        Trainable parameters only

  Activations (gradient          \~1.5 GB        Reduced from \~4GB
  checkpointing)                                 

  Paged AdamW Optimizer States   \~0.5 GB        Paged to CPU when needed

  CUDA Kernel Overhead           \~0.5 GB        Fixed runtime overhead

  **TOTAL**                      **\~6.2 GB**    **Peak ≤ 7.5 GB with
                                                 batch=2**
  ------------------------------ --------------- -------------------------

**2.3 inference.py --- Serving Layer**

+-----------------------------------------------------------------------+
| **Inference & Deployment Module**                                     |
+-----------------------------------------------------------------------+
| -   merge_and_unload(): Merges LoRA weights into base model for       |
|     zero-overhead inference                                           |
|                                                                       |
| -   TextStreamer: Token-by-token streaming to terminal for            |
|     interactive mode                                                  |
|                                                                       |
| -   FastAPI /generate: POST endpoint accepting {prompt,               |
|     max_new_tokens, temperature}                                      |
|                                                                       |
| -   FastAPI /health: GET endpoint returning GPU name and status for   |
|     health checks                                                     |
|                                                                       |
| -   Three modes: chat (interactive CLI), single (one-shot), api (REST |
|     server)                                                           |
|                                                                       |
| -   Temperature control: 0.1 = deterministic, 0.9 = creative          |
|     (default: 0.7)                                                    |
+-----------------------------------------------------------------------+

**API Contract**

POST /generate --- Request body:

> { \"prompt\": \"string\", \"max_new_tokens\": 512, \"temperature\":
> 0.7 }

POST /generate --- Response:

> { \"response\": \"string\" }

GET /health --- Response:

> { \"status\": \"ok\", \"gpu\": \"NVIDIA GeForce RTX 4060 Laptop GPU\"
> }

**2.4 evaluate.py --- Evaluation Layer**

+-----------------------------------------------------------------------+
| **Evaluation Suite Module**                                           |
+-----------------------------------------------------------------------+
| -   compute_perplexity(): Batch-processes eval set, computes exp(mean |
|     NLL) --- lower = better                                           |
|                                                                       |
| -   compute_rouge(): ROUGE-1, ROUGE-2, ROUGE-L F1 scores for          |
|     generation quality                                                |
|                                                                       |
| -   speed_benchmark(): 3-run average tokens/sec with GPU warmup pass  |
|                                                                       |
| -   run_benchmark_prompts(): JSON-driven qualitative eval with        |
|     keyword matching                                                  |
|                                                                       |
| -   VRAM reporting: memory_allocated() / total_memory for resource    |
|     tracking                                                          |
+-----------------------------------------------------------------------+

**3. Data Flow Diagrams**

**3.1 Training Data Flow**

  -----------------------------------------------------------------------
  Raw Data (CSV / HF Dataset / JSONL)

  ↓ prepare_data.py

  Format Validation + Mistral Template Wrapping

  ↓

  data/train.jsonl (instruction, input, output)

  ↓ train.py

  HuggingFace Dataset → 90% train / 10% eval split

  ↓

  BitsAndBytesConfig (4-bit NF4) → Quantized Base Model

  ↓

  prepare_model_for_kbit_training() → LoRA Injection

  ↓

  SFTTrainer (paged_adamw, fp16, grad_checkpoint)

  ↓ Every 50 steps

  Checkpoint: models/qlora-model/checkpoint-N

  ↓ On completion

  Final Adapter: models/qlora-model/final_adapter
  -----------------------------------------------------------------------

**3.2 Inference Data Flow**

  -----------------------------------------------------------------------
  User Prompt (string)

  ↓ inference.py

  Tokenize: \<s\>\[INST\] {prompt} \[/INST\]

  ↓

  Load Base Model + PeftModel.from_pretrained(adapter)

  ↓ merge_and_unload()

  Merged Full-Precision Model (fp16)

  ↓ model.generate()

  Token-by-Token Sampling (temperature=0.7, top_p=0.9)

  ↓ tokenizer.decode()

  Response String → Return to caller
  -----------------------------------------------------------------------

**4. Model Architecture**

**4.1 Base Model: Mistral-7B-Instruct-v0.2**

  ---------------------- ------------------------------------------------
  **Architecture**       Transformer Decoder (32 layers)

  **Parameters**         7.24 billion

  **Context Window**     32,768 tokens (Sliding Window Attention)

  **Attention Heads**    32 heads (GQA --- 8 KV heads)

  **Hidden Dimension**   4,096

  **Intermediate Size**  14,336 (SwiGLU activation)

  **Vocabulary Size**    32,000 tokens (sentencepiece BPE)

  **Fine-tuning Format** Mistral Instruct: \<s\>\[INST\] \... \[/INST\]
                         \...\</s\>
  ---------------------- ------------------------------------------------

**4.2 LoRA Adapter Design**

LoRA injects trainable low-rank matrices (A and B) into specific weight
matrices. For a weight matrix W, the update is: W\' = W + BA, where B is
d×r and A is r×d (r \<\< d). Only A and B are trained --- W stays
frozen.

  -------------------- --------------- ------------------------------------
  **Hyperparameter**   **Value**       **Rationale**

  Rank (r)             16              Balances capacity vs parameter count
                                       (\~6M params added)

  Alpha (α)            32              α/r=2; standard scaling that works
                                       well empirically

  Dropout              0.05            Light regularization to prevent LoRA
                                       overfitting

  Target Modules       7 layers        q/k/v/o projection + gate/up/down
                                       MLP layers

  Trainable %          \~0.08%         Only \~6M of 7.24B params are
                                       trained
  -------------------- --------------- ------------------------------------

**5. File & Directory Structure**

> qlora_llm/
>
> ├── scripts/
>
> │ ├── train.py ← QLoRA training pipeline (core)
>
> │ ├── inference.py ← Model serving (chat + FastAPI)
>
> │ ├── evaluate.py ← Metrics: perplexity, ROUGE, speed
>
> │ └── prepare_data.py ← Data ingestion and validation
>
> ├── data/
>
> │ └── train.jsonl ← Training data (JSONL format)
>
> ├── models/
>
> │ └── qlora-mistral-7b/
>
> │ ├── checkpoint-50/ ← Intermediate checkpoints
>
> │ ├── checkpoint-100/
>
> │ └── final_adapter/ ← Final LoRA weights + tokenizer
>
> ├── configs/
>
> │ └── benchmark_prompts.json
>
> ├── notebooks/ ← Jupyter exploration notebooks
>
> ├── requirements.txt
>
> ├── todo.md
>
> └── README.md

**6. Key Design Decisions**

  ------------------ -------------------------- -----------------------------
  **Decision**       **Chosen Approach**        **Why Not Alternative**

  Quantization       NF4 4-bit (bitsandbytes)   GPTQ: requires calibration
                                                data; GGUF: inference-only

  PEFT Method        LoRA (peft library)        Full fine-tune: exceeds 8GB
                                                VRAM; Prefix tuning: worse
                                                performance

  Optimizer          paged_adamw_32bit          Standard AdamW: OOM; 8-bit
                                                Adam: less stable with LoRA

  Base Model         Mistral-7B-Instruct-v0.2   LLaMA-3: requires HF
                                                approval; Gemma: lower
                                                benchmark scores

  Training Framework TRL SFTTrainer             Raw PyTorch: more bugs;
                                                Axolotl: overkill for this
                                                scope

  API Framework      FastAPI + uvicorn          Flask: slower; Triton:
                                                over-engineered for single
                                                GPU
  ------------------ -------------------------- -----------------------------
