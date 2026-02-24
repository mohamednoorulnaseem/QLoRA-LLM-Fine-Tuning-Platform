**TECH STACK DEFINITION**

**QLoRA LLM Fine-Tuning Platform**

*Canonical Technology Choices & Version Constraints*

**1. Hardware Specifications**

+-----------------------+-----------------------+-----------------------+
| **GPU**               | **CPU**               | **RAM**               |
|                       |                       |                       |
| RTX 4060 8GB GDDR6    | Intel i7-13650HX      | 24GB DDR5-4800        |
+-----------------------+-----------------------+-----------------------+
| **Storage**           | **OS**                | **CUDA**              |
|                       |                       |                       |
| 512GB NVMe SSD        | Windows 11 / Ubuntu   | CUDA 12.1+            |
+-----------------------+-----------------------+-----------------------+

**2. Core ML Stack**

**2.1 Deep Learning Foundation**

  --------------- ------------------ --------------- ------------------------
  **Category**    **Package / Tool** **Version**     **Purpose**

  **Framework**   PyTorch            ≥ 2.1.0         Primary DL framework ---
                                                     CUDA operations, tensor
                                                     math, autograd

                  torchvision        ≥ 0.16          Image utilities --- used
                                                     for future multimodal
                                                     extensions

                  torchaudio         ≥ 2.1           Audio utilities --- used
                                                     if extending to speech
                                                     tasks

  **CUDA**        CUDA Toolkit       12.1+           GPU acceleration driver
                                                     --- must match PyTorch
                                                     CUDA build

                  cuDNN              8.x             NVIDIA deep learning
                                                     primitives for
                                                     convolutions/attention
  --------------- ------------------ --------------- ------------------------

**2.2 HuggingFace Ecosystem**

  ------------------ ------------------ --------------- -----------------------
  **Category**       **Package / Tool** **Version**     **Purpose**

  **Models**         transformers       ≥ 4.40.0        Model loading,
                                                        tokenizers, generation
                                                        configs for all LLMs

  **Datasets**       datasets           ≥ 2.18.0        Efficient dataset
                                                        loading, streaming, HF
                                                        Hub integration

  **Accelerate**     accelerate         ≥ 0.28.0        Multi-device training
                                                        abstraction and
                                                        fp16/bf16 mixed
                                                        precision

  **Fine-tuning**    peft               ≥ 0.10.0        LoRA, QLoRA, IA3
                                                        parameter-efficient
                                                        methods

  **Training**       trl                ≥ 0.8.0         SFTTrainer ---
                                                        supervised fine-tuning
                                                        with reward modeling
                                                        support

  **Tokenization**   tokenizers         ≥ 0.19.0        Fast Rust-based
                                                        tokenizers for
                                                        Mistral/LLaMA

  **Hub**            huggingface_hub    ≥ 0.21.0        Model downloading,
                                                        uploading,
                                                        authentication
  ------------------ ------------------ --------------- -----------------------

**2.3 Quantization**

  -------------- ------------------ --------------- -----------------------
  **Category**   **Package / Tool** **Version**     **Purpose**

  **4-bit**      bitsandbytes       ≥ 0.43.0        NF4/INT8 quantization
                                                    --- core to fitting 7B
                                                    in 8GB VRAM

                 (CUDA extensions)  auto            Compiled at install
                                                    time --- requires CUDA
                                                    headers to be present
  -------------- ------------------ --------------- -----------------------

IMPORTANT: bitsandbytes must be installed AFTER PyTorch with CUDA. On
Windows, use bitsandbytes-windows fork.

**3. Training Infrastructure**

  ---------------- ------------------ --------------- -----------------------
  **Category**     **Package / Tool** **Version**     **Purpose**

  **Logging**      wandb              ≥ 0.16          Experiment tracking ---
                                                      loss curves,
                                                      hyperparams, artifact
                                                      versioning

                   tensorboard        ≥ 2.15          Local alternative to
                                                      W&B --- no account
                                                      needed

  **Monitoring**   nvidia-smi         system          GPU utilization and
                                                      VRAM monitoring during
                                                      training

  **Data**         pandas             ≥ 2.0           CSV/DataFrame
                                                      manipulation for data
                                                      preparation

                   numpy              ≥ 1.26          Numerical operations
                                                      for evaluation metrics

                   scikit-learn       ≥ 1.3           Train/eval splitting,
                                                      basic ML utilities

  **Progress**     tqdm               ≥ 4.66          Progress bars for
                                                      training loops

  **Config**       python-dotenv      ≥ 1.0           Environment variable
                                                      management (HF tokens,
                                                      W&B keys)
  ---------------- ------------------ --------------- -----------------------

**4. Evaluation Stack**

  -------------- ------------------ --------------- -----------------------
  **Category**   **Package / Tool** **Version**     **Purpose**

  **Metrics**    rouge-score        ≥ 0.1.2         ROUGE-1, ROUGE-2,
                                                    ROUGE-L for generation
                                                    quality

                 evaluate           ≥ 0.4           HuggingFace evaluation
                                                    framework (wraps BLEU,
                                                    ROUGE, etc.)

  **NLP**        nltk               ≥ 3.8           Tokenization utilities
                                                    for metric computation

  **Output**     rich               ≥ 13.0          Beautiful terminal
                                                    output for eval results
                                                    display
  -------------- ------------------ --------------- -----------------------

**5. Deployment Stack**

  ---------------- --------------------- --------------- -----------------------
  **Category**     **Package / Tool**    **Version**     **Purpose**

  **API**          fastapi               ≥ 0.110.0       Async REST API
                                                         framework --- /generate
                                                         and /health endpoints

  **Server**       uvicorn\[standard\]   ≥ 0.27          ASGI server for FastAPI
                                                         --- production-grade
                                                         with websocket support

  **Validation**   pydantic              ≥ 2.0           Request/response
                                                         validation and schema
                                                         definition

  **WSGI**         httpx                 ≥ 0.27          Async HTTP client for
                                                         API testing

  **Optional**     llama-cpp-python      latest          GGUF model inference on
                                                         CPU after conversion
                                                         --- for portability

  **Optional**     docker                24+             Containerization for
                                                         reproducible deployment
                                                         environment
  ---------------- --------------------- --------------- -----------------------

**6. Language & Environment**

**6.1 Python Version**

  ------------------ --------------- ------------------------------------
  **Requirement**    **Value**       **Notes**

  Python Version     3.10 --- 3.11   3.12 not yet fully supported by
                                     bitsandbytes

  Virtual            venv (built-in) Do NOT use conda --- CUDA path
  Environment                        conflicts with bitsandbytes

  pip version        ≥ 23.0          Run: pip install \--upgrade pip
                                     before installing requirements

  Node.js (optional) ≥ 18            Only needed if generating docx/pptx
                                     documentation artifacts
  ------------------ --------------- ------------------------------------

**6.2 Installation Order (CRITICAL)**

Install in this EXACT order to avoid CUDA/bitsandbytes conflicts:

> \# Step 1: Install PyTorch with CUDA FIRST
>
> pip install torch torchvision torchaudio \--index-url
> https://download.pytorch.org/whl/cu121
>
> \# Step 2: Verify CUDA is available
>
> python -c \"import torch; print(torch.cuda.is_available())\" \# Must
> print: True
>
> \# Step 3: Install bitsandbytes (needs CUDA already present)
>
> pip install bitsandbytes\>=0.43.0
>
> \# Step 4: Install rest of requirements
>
> pip install -r requirements.txt
>
> \# Step 5: Verify GPU is detected
>
> python -c \"import torch; print(torch.cuda.get_device_name(0))\"
>
> \# Expected: NVIDIA GeForce RTX 4060 Laptop GPU

**7. Version Lock Policy**

All packages are pinned to minimum versions in requirements.txt. The
team must NOT upgrade major versions (e.g., peft 0.x → 1.x) without
testing on RTX 4060 first, as breaking changes in bitsandbytes, peft,
and transformers frequently occur across minor versions in rapidly
evolving LLM tooling.

  ------------------ ------------- --------------------------------------
  **Package**        **Lock        **Reason**
                     Level**       

  bitsandbytes       STRICT (==    Breaking API changes in every minor
                     pin)          version

  peft               STRICT (==    LoRA API changed in 0.7.x and 0.10.x
                     pin)          --- critical

  transformers       MINIMUM (\>=) Generally backwards compatible within
                                   major version

  trl                MINIMUM (\>=) SFTTrainer API stable since 0.7

  torch              MINIMUM (\>=) CUDA version determines compatible
                                   range

  fastapi            LOOSE (any)   Stable API --- minor updates safe
  ------------------ ------------- --------------------------------------
