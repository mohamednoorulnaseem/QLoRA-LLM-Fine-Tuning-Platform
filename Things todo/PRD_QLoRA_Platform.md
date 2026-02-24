**PRODUCT REQUIREMENTS DOCUMENT**

**QLoRA LLM Fine-Tuning Platform**

*Real-World AI Application \| Powered by Mistral-7B + RTX 4060*

  ------------------ ----------------------------------------------------
  **Document         v1.0
  Version**          

  **Status**         Draft --- In Review

  **Author**         Mohamed Noorul Naseem M

  **Institution**    Anand Institute of Higher Technology

  **Created**        February 2026

  **Hardware**       Lenovo LOQ 13th Gen \| Intel i7-13650HX \| RTX 4060
                     8GB \| 24GB RAM
  ------------------ ----------------------------------------------------

**1. Executive Overview**

This document defines the product requirements for a production-grade
Large Language Model fine-tuning platform built using QLoRA (Quantized
Low-Rank Adaptation). The platform enables domain-specific fine-tuning
of 7B+ parameter models on consumer hardware (RTX 4060, 8GB VRAM) and
exposes the resulting model as a deployable REST API.

The product targets two primary use cases: (1) academic research
requiring custom-trained LLMs, and (2) real-world business applications
such as intelligent customer support, domain-specific Q&A systems, and
automated document processing.

**2. Problem Statement**

**2.1 Current Pain Points**

-   Training 7B LLMs from scratch requires 8+ A100 GPUs --- inaccessible
    for individuals and small teams

-   Closed-source LLM APIs (GPT-4, Claude) cannot be fine-tuned on
    proprietary domain data

-   Existing fine-tuning frameworks have high complexity barriers with
    poor documentation

-   No standardized pipeline from raw data → trained model → deployed
    API on local hardware

**2.2 Opportunity**

QLoRA reduces fine-tuning VRAM requirements by \~75% through 4-bit NF4
quantization + LoRA adapters, making 7B model fine-tuning feasible on
RTX 4060 (8GB). This platform packages that capability into a
reproducible, end-to-end system.

**3. Goals & Success Metrics**

**3.1 Primary Goals**

1.  Fine-tune Mistral-7B-Instruct on custom JSONL datasets within RTX
    4060 VRAM constraints (≤7.5GB)

2.  Achieve perplexity \< 5.0 on domain-specific evaluation set after 3
    epochs

3.  Deploy fine-tuned model as a REST API with latency \< 3 seconds for
    256-token responses

4.  Provide full observability: loss curves, VRAM usage, tokens/sec via
    TensorBoard/W&B

**3.2 Success Metrics**

  -------------------------- ------------------ -------------------------
  **Metric**                 **Target**         **Measurement**

  VRAM Peak Usage            ≤ 7.5 GB           nvidia-smi during
                                                training

  Training Perplexity        \< 5.0             evaluate.py on held-out
                                                set

  Inference Speed            ≥ 15 tok/sec       speed_benchmark()

  API Response Time          \< 3 seconds       curl timing with 256
                                                tokens

  Dataset Coverage           \> 1,000 samples   prepare_data.py
                                                validation
  -------------------------- ------------------ -------------------------

**4. Scope**

**4.1 In Scope**

-   QLoRA training pipeline (4-bit quantization, LoRA adapters, gradient
    checkpointing)

-   Support for Mistral-7B, LLaMA-3-8B, Gemma-7B, Phi-3-Mini base models

-   Data preparation utilities (CSV, JSONL, HuggingFace datasets)

-   Evaluation suite: perplexity, ROUGE, benchmark prompts, speed test

-   FastAPI inference server with /generate and /health endpoints

-   Training monitoring via TensorBoard and Weights & Biases

**4.2 Out of Scope (v1.0)**

-   Multi-GPU distributed training (future v2.0)

-   RLHF / PPO reward model training

-   Web-based UI for non-technical users

-   Cloud deployment automation (AWS/GCP/Azure)

**5. Feature Requirements**

**5.1 Priority Matrix**

  ---------------------- -------------- ------------ -------------------------
  **Feature**            **Priority**   **Sprint**   **Notes**

  QLoRA Training Script  **P0**         Sprint 1     Core pipeline --- must
                                                     ship first

  Data Preparation       **P0**         Sprint 1     CSV, JSONL, HF datasets
  Utilities                                          

  Evaluation Suite       **P0**         Sprint 1     Perplexity + ROUGE +
                                                     speed

  FastAPI Inference      **P1**         Sprint 2     REST API for deployment
  Server                                             

  Interactive CLI Chat   **P1**         Sprint 2     For testing fine-tuned
                                                     model

  WandB / TensorBoard    **P1**         Sprint 2     Real-time loss curves
  Logging                                            

  Early Stopping         **P1**         Sprint 2     Prevent overfitting
  Callback                                           

  Benchmark Prompt       **P2**         Sprint 3     JSON-driven qualitative
  Config                                             eval

  GGUF Export            **P2**         Sprint 3     For llama.cpp deployment

  Docker Container       **P3**         Sprint 4     Optional prod packaging
  ---------------------- -------------- ------------ -------------------------

**6. User Stories**

**As an ML Engineer, I want to\...**

-   Fine-tune Mistral-7B on my customer support dataset so that the
    model gives brand-specific answers

-   Monitor training loss in real-time so that I can detect overfitting
    early and stop the run

-   Evaluate my model\'s perplexity against a held-out set so that I can
    compare models objectively

**As a Backend Developer, I want to\...**

-   Call a REST API endpoint with a prompt so that I can integrate the
    fine-tuned model into my application

-   Check model health via /health endpoint so that I can include it in
    my service monitoring

**As a Data Engineer, I want to\...**

-   Convert my existing CSV Q&A data to JSONL format so that it can be
    used for fine-tuning without manual reformatting

-   Validate dataset quality before training so that I can catch empty
    outputs and formatting errors early

**7. Non-Functional Requirements**

**7.1 Performance**

-   Peak VRAM must stay ≤ 7.5GB on RTX 4060 (8GB) during training

-   Inference latency ≤ 3 seconds for 256 new tokens on the same
    hardware

-   Training throughput ≥ 500 tokens/second on 2048-token sequences

**7.2 Reliability**

-   Automatic checkpointing every 50 steps to prevent loss on crash

-   EarlyStoppingCallback with patience=3 to prevent runaway training

-   Input validation in prepare_data.py before data reaches the trainer

**7.3 Reproducibility**

-   All random seeds fixed (seed=42) for reproducible train/eval splits

-   Full config logged to W&B / TensorBoard at run start

-   requirements.txt with pinned versions for environment
    reproducibility

**8. Delivery Timeline**

  ------------- ---------------- ---------------------- -----------------
  **Sprint**    **Duration**     **Deliverables**       **Status**

  Sprint 1      Week 1--2        Setup, train.py,       In Progress
                                 prepare_data.py,       
                                 evaluate.py            

  Sprint 2      Week 3--4        inference.py, FastAPI  Planned
                                 server, W&B            
                                 integration            

  Sprint 3      Week 5--6        Benchmark suite, model Planned
                                 comparison, GGUF       
                                 export                 

  Sprint 4      Week 7--8        Docker, documentation, Planned
                                 demo deployment        
  ------------- ---------------- ---------------------- -----------------

**9. Risks & Mitigations**

  ---------------------- -------------- ------------------------------------
  **Risk**               **Severity**   **Mitigation**

  CUDA OOM during        High           Reduce batch_size to 1, max_seq_len
  training                              to 1024; use gradient checkpointing

  Model underfitting on  Medium         Use tatsu-lab/alpaca (52K) as base +
  small data                            domain-specific data mixing

  bitsandbytes           Medium         Pin version in requirements.txt;
  incompatibility                       test on clean venv before training

  HuggingFace download   Low            Use HF_ENDPOINT mirror or
  throttling                            pre-download models with
                                        huggingface-cli
  ---------------------- -------------- ------------------------------------

**10. Glossary**

  ------------------ ----------------------------------------------------
  **QLoRA**          Quantized Low-Rank Adaptation --- fine-tunes large
                     models with 4-bit quantization

  **NF4**            NormalFloat4 --- 4-bit quantization format optimized
                     for normally-distributed LLM weights

  **LoRA**           Low-Rank Adaptation --- adds small trainable
                     matrices to frozen base model layers

  **VRAM**           Video RAM --- GPU memory; RTX 4060 has 8GB

  **Perplexity**     How \'surprised\' the model is by test data ---
                     lower is better (target: \< 5.0)

  **JSONL**          JSON Lines format --- one JSON object per line, used
                     as training data format

  **SFT**            Supervised Fine-Tuning --- training on
                     instruction-output pairs with teacher forcing

  **ROUGE**          Recall-Oriented Understudy for Gisting Evaluation
                     --- text overlap metric for generation quality
  ------------------ ----------------------------------------------------
