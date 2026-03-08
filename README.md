# Text Generation Transformer — From Scratch

> A decoder-only Transformer built entirely with PyTorch primitives, trained on WikiText-2 with a full training + validation pipeline and production-structured MLOps conventions.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Key Design Decisions](#key-design-decisions)
- [Architecture](#architecture)
- [Core Components](#core-components)
  - [Multi-Head Causal Self-Attention](#multi-head-causal-self-attention)
  - [Feed-Forward Network](#feed-forward-network)
  - [Pre-Layer Normalization](#pre-layer-normalization)
- [Training System](#training-system)
  - [Language Modeling Objective](#language-modeling-objective)
  - [Training Loop](#training-loop)
  - [Gradient Clipping](#gradient-clipping)
  - [Optimizer](#optimizer)
  - [Learning Rate Schedule](#learning-rate-schedule)
- [Validation & Evaluation](#validation--evaluation)
  - [Validation Loop](#validation-loop)
  - [Perplexity Metric](#perplexity-metric)
  - [Early Stopping & Best Model Tracking](#early-stopping--best-model-tracking)
- [Metric Logging](#metric-logging)
- [Checkpointing](#checkpointing)
- [Data Pipeline](#data-pipeline)
- [Reproducibility](#reproducibility)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Quickstart](#quickstart)
- [Roadmap](#roadmap)
- [Dependencies](#dependencies)
- [References](#references)

---

## Overview

This project implements a GPT-style, decoder-only Transformer **from scratch** — no `transformers` library, no high-level wrappers. Every component (attention, positional encoding, training loop, validation loop, scheduling) is written directly in PyTorch to demonstrate both architectural understanding and clean ML engineering practice.

The codebase is structured with MLOps discipline in mind: reproducible data pipelines, YAML-based configuration, token-weighted loss aggregation, perplexity tracking, early stopping, dual checkpointing (`last.pt` / `best.pt`), and structured JSONL metric logging, multiple sampling strategies for text generation, and production-ready FastAPI serving.

---

## 📚 Detailed Guides

New to training Transformers or need step-by-step guidance? Check out these comprehensive guides:

- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** — Complete walkthrough of the training process, hyperparameter tuning, monitoring, troubleshooting, and GPU setup
- **[TESTING_GUIDE.md](docs/TESTING_GUIDE.md)** — How to evaluate your trained model on the test set, interpret metrics, and compare results

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | Decoder-only Transformer | Suited for autoregressive text generation (GPT-style) |
| Positional Encoding | Learnable embeddings | Simpler than sinusoidal; competitive on short sequences |
| Layer Normalization | Pre-LN (before attention/FFN) | More stable gradients; avoids vanishing gradient pathology in deep stacks |
| Optimizer | AdamW | Adaptive gradient scaling critical for Transformer stability; SGD was tested and discarded |
| LR Schedule | Linear warmup + cosine annealing | Prevents unstable early updates; smooth decay through training |
| Tokenizer | GPT-2 BPE | Standard, well-understood; consistent with the training objective |
| Loss aggregation | Token-weighted sum | Avoids batch-size bias when computing average loss across variable-length batches |
| Validation metric | Perplexity (`exp(avg_loss)`) | Standard LM benchmark metric; interpretable as effective vocabulary size the model is "choosing between" |
| Early stopping | Patience-based on `val_loss` | Prevents overfitting; saves compute when validation stops improving |
| Checkpointing | Dual: `last.pt` + `best.pt` | `last.pt` enables resumption; `best.pt` captures peak generalisation |
| Metric logging | JSONL append-log | Machine-readable, easy to parse for plotting; decoupled from Trainer |
| Generation strategy | Temperature + Top-P (nucleus) | Industry standard (GPT-3, ChatGPT); best balance of quality and diversity |
| API framework | FastAPI | High performance, automatic validation, interactive docs, production-ready |

---

## Architecture

```
Input Tokens  (B, T)
      ↓
Token Embedding  +  Positional Embedding    →  (B, T, C)
      ↓
N × Transformer Block
  ├── Pre-LayerNorm
  ├── Multi-Head Causal Self-Attention
  ├── Residual Connection
  ├── Pre-LayerNorm
  ├── Position-wise FFN  [C → 4C → C]
  └── Residual Connection
      ↓
Final LayerNorm
      ↓
Linear Projection  →  (B, T, vocab_size)
      ↓
Logits
```

**Tensor shape notation:** `B` = batch size · `T` = sequence length · `C` = embedding dimension · `V` = vocabulary size

---

## Core Components

### Multi-Head Causal Self-Attention

Each attention head computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where `d_k = C / num_heads`. The `1/√d_k` scaling prevents large dot-products from saturating the softmax and collapsing gradient flow.

A lower-triangular causal mask is applied before softmax, ensuring token `t` only attends to positions `≤ t`:

```python
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float('-inf'))
```

### Feed-Forward Network

A position-wise two-layer MLP applied identically to each token:

```
Linear(C → 4C) → ReLU → Linear(4C → C)
```

The 4× expansion factor follows the original "Attention Is All You Need" design, giving the model enough capacity to learn rich per-token representations before projecting back.

### Pre-Layer Normalization

LayerNorm is applied **before** each sublayer (Pre-LN convention), not after:

```python
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

This makes gradient norms more predictable across depth and allows training without additional learning rate tricks.

---

## Training System

### Language Modeling Objective

Given a token sequence `(x₁, x₂, ..., xₜ)`, the model is trained to maximise:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_{t+1} \mid x_{\leq t})$$

Input and target sequences are constructed with a one-position offset. Both `Trainer` and `Evaluator` apply the same shift before computing loss:

```python
logits  = logits[:, :-1, :]    # (B, T-1, V)  — drop last prediction
targets = input_ids[:, 1:]     # (B, T-1)     — drop first token
```

This ensures the model never sees the token it's predicting.

### Training Loop

The training lifecycle is split across three methods in `Trainer`:

```
fit()
 └── for each epoch:
       train_epoch()
        └── for each batch:
              train_step()
                ├── forward pass         → logits (B, T, V)
                ├── shift logits/targets → (B, T-1, V) / (B, T-1)
                ├── cross-entropy loss
                ├── zero_grad()
                ├── loss.backward()
                ├── clip_grad_norm_(max=1.0)
                ├── optimizer.step()
                └── scheduler.step()
       evaluator.evaluate()
       logger.log()
       save_checkpoint(last.pt)
       save_checkpoint(best.pt)   ← only if val_loss improved
       early stopping check
```

Loss is accumulated **token-weighted** across the epoch to avoid batch-size bias:

```python
total_loss += loss.item() * targets.numel()   # weight by token count
total_tokens += targets.numel()
avg_loss = total_loss / total_tokens          # true per-token average
```

### Gradient Clipping

The total gradient norm is clipped to `1.0` after every backward pass:

$$
g \leftarrow g \cdot \frac{\text{max\_norm}}{\|g\|} \quad \text{where} \quad \|g\| = \sqrt{\sum_i \|g_i\|^2}
$$

An observed pre-clip gradient norm of ~220 confirms this is a necessary safeguard for stable Transformer training.

### Optimizer

AdamW is used with configurable weight decay:

```python
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

SGD was tested and produced large loss oscillations. AdamW's per-parameter adaptive scaling is critical for Transformers, where gradient magnitudes vary significantly across layers.

### Learning Rate Schedule

**Linear warmup** — ramps LR from 1% of peak to full over `warmup_steps`:

$$
lr_t = lr_{\max} \cdot \frac{t}{\text{warmup\_steps}}
$$

**Cosine annealing** — smooth decay after warmup:

$$lr_t = lr_{\min} + \frac{1}{2}(lr_{\max} - lr_{\min})\left(1 + \cos\!\left(\frac{t}{T}\pi\right)\right)$$

Implemented as `SequentialLR(LinearLR, CosineAnnealingLR)` with the milestone at `warmup_steps`.

---

## Validation & Evaluation

### Validation Loop

`Evaluator.evaluate()` runs after every training epoch. Key implementation details:

```python
self.model.eval()                   # disables dropout
with torch.no_grad():               # no gradient tracking
    for batch in dataloader:
        logits = self.model(input_ids)
        logits, targets = self.shift_logits_targets(logits, input_ids)
        loss = criterion(logits.reshape(-1, V), targets.reshape(-1))

        total_loss   += loss.item() * targets.numel()   # token-weighted accumulation
        total_tokens += targets.numel()

avg_loss   = total_loss / total_tokens
perplexity = math.exp(avg_loss)
```

Two key design choices:
- `model.eval()` correctly disables dropout so validation scores reflect inference-time behaviour, not stochastic training behaviour.
- `torch.no_grad()` eliminates gradient bookkeeping, cutting memory usage and speeding up the validation pass.

### Perplexity Metric

Perplexity is the primary evaluation metric for language models:

$$\text{PPL} = \exp\left(\frac{1}{N}\sum_{i=1}^{N} -\log P(x_i)\right) = \exp(\text{avg\_loss})$$

Intuitively, PPL represents the effective number of equally likely tokens the model is "choosing between" at each position. A PPL of 100 means the model is as uncertain as if it were picking uniformly from 100 tokens. Lower is better.

| Perplexity | Interpretation |
|---|---|
| < 50 | Excellent: model has strong next-token prediction |
| 50–100 | Good: competitive on standard benchmarks |
| 100–200 | Acceptable: small models on challenging data |
| > 200 | Needs improvement: underfitting or hyperparameter issues |

```python
def calc_perplexity(loss: float) -> float:
    return math.exp(loss)
```

Both `val_loss` and `val_perplexity` are logged each epoch, giving a complete picture of generalisation quality.

### Early Stopping & Best Model Tracking

`Trainer.fit()` implements patience-based early stopping against validation loss:

```python
patience = config["training"].get("patience", 5)   # default: 5 epochs
epochs_without_improvement = 0

if val_loss < self.best_val_loss:
    self.best_val_loss = val_loss
    save_checkpoint(..., "best.pt")      # save new best
else:
    epochs_without_improvement += 1

if epochs_without_improvement >= patience:
    print("Early stopping triggered.")
    break
```

`best_val_loss` is initialised to `float("inf")`, so the first epoch always saves a best checkpoint. The counter only increments on epochs where validation loss does **not** improve, so a single bad epoch won't terminate training prematurely.

---

## Metric Logging

`Logger` is a deliberately thin class, kept separate from `Trainer` to maintain single-responsibility:

```python
logger.log({
    "epoch":           epoch,
    "train_loss":      train_loss,
    "val_loss":        val_loss,
    "val_perplexity":  val_ppl,
    "epoch_time_sec":  epoch_time,
    "global_step":     self.global_step,
    "current_lr":      current_lr,
})
```

Metrics are written to `runs/experiment_1/metrics.jsonl` in newline-delimited JSON format — one record per epoch. This makes downstream analysis trivial:

```python
import pandas as pd
df = pd.read_json("runs/experiment_1/metrics.jsonl", lines=True)
df.plot(x="epoch", y=["train_loss", "val_loss"])
```

---

## Checkpointing

Two checkpoints are maintained per run:

| File | Saved when | Purpose |
|---|---|---|
| `checkpoints/last.pt` | Every epoch | Resume training after interruption |
| `checkpoints/best.pt` | `val_loss` improves | Best generalising model for inference |

Each checkpoint contains the full training state:

```python
{
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "global_step":          step,
    "epoch":                epoch,
}
```

Checkpoint files are excluded from version control via `.gitignore`.

---

## Data Pipeline

```
HuggingFace datasets (wikitext-2-raw-v1)
          ↓
  data/raw/            ← immutable source, persisted to disk on first run
          ↓
  GPT-2 Tokenizer      ← each document tokenized individually, then concatenated
          ↓
  data/processed/      ← token tensors (.pt), loaded directly on subsequent runs
          ↓
  DataLoader (train)   ← shuffle=True
  DataLoader (val)     ← shuffle=False
```

Documents are tokenized independently then concatenated into a single flat 1D tensor:

```python
self.input_ids = torch.cat(all_input_ids, dim=0)   # shape: (N,)
```

Fixed-length non-overlapping sequences are chunked for batching:

```
num_sequences = floor((N - 1) / seq_len)
chunks: [0:L], [L:2L], [2L:3L], ...
```

---

## Reproducibility

All random sources are seeded at the top of `train.py`:

```python
set_seed(42)

# inside set_seed():
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

This guarantees consistent weight initialisation, data shuffling order, and dropout behaviour across runs.

---

## Inference & Text Generation

### Generation Strategies

The `TextGenerator` class implements 5 different sampling strategies for converting model logits into generated text:

#### 1. **Greedy Decoding** (Deterministic)
```python
next_token = torch.argmax(logits, dim=-1)
```
- Always picks the highest probability token
- Fast, reproducible, but repetitive
- Use case: Testing, debugging

#### 2. **Temperature Sampling**
```python
logits = logits / temperature
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```
- Lower temperature (0.5–0.7): More focused, conservative
- Higher temperature (1.2–1.5): More random, creative
- Use case: Creative writing, varying output diversity

#### 3. **Top-K Sampling**
```python
top_k_values, top_k_indices = torch.topk(logits, k)
# Keep only top K tokens, set rest to -inf
# Sample from filtered distribution
```
- Filters out unlikely tokens, keeps top K
- Prevents nonsense while maintaining diversity
- Use case: Safe generation with controlled randomness

#### 4. **Top-P / Nucleus Sampling** ⭐
```python
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
# Keep tokens until cumsum >= top_p
```
- **Adaptive**: Uses fewer tokens when model is confident, more when uncertain
- Industry standard (GPT-3, ChatGPT, Claude)
- Use case: **Best balance of quality and diversity**

#### 5. **Combined (Temperature + Top-P)** 🏆 PRODUCTION STANDARD
```python
logits = logits / temperature     # Adjust sharpness
probs = F.softmax(logits, dim=-1)  # Convert to probabilities
probs = top_p_filter(probs, p)     # Keep nucleus
next_token = torch.multinomial(probs, 1)  # Sample
```
- Combines temperature control with adaptive filtering
- **Used by all major LLMs in production**
- Use case: **Default choice for all applications**

### Strategy Selection Guide

| Goal | Strategy | Settings |
|------|----------|----------|
| Quick test | `greedy` | - |
| Creative story | `combined` | temp=1.2, top_p=0.95 |
| Focused summary | `combined` | temp=0.6, top_p=0.85 |
| Chatbot | `combined` | temp=0.8, top_p=0.9 |
| Code completion | `top_p` | temp=0.5, top_p=0.9 |

### Repetition Penalty

To prevent repetitive loops ("the the the..."), a penalty factor is applied to already-generated tokens:

```python
for token_id in generated_tokens:
    logits[0, token_id] /= repetition_penalty
```

**Typical values**: 1.0 (no penalty) to 2.0 (strong penalty). Default: 1.2

---

# API Serving

### REST API Endpoints

The FastAPI application (`api.py`) provides a production-ready REST API with the following endpoints:

#### **GET /** — Root / Health Check
Returns API status and available endpoints.

#### **GET /health** — Detailed Health Check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

#### **GET /models/info** — Model Information
```json
{
  "model_name": "Decoder-Only Transformer",
  "vocabulary_size": 50257,
  "total_parameters": 2703616,
  "max_sequence_length": 128,
  "device": "cpu",
  "checkpoint_loaded": "/path/to/best.pt",
  "epoch": 11
}
```

#### **POST /generate** — Generate Text ⭐ Main Endpoint
**Request:**
```json
{
  "prompt": "The future of artificial intelligence is",
  "max_length": 50,
  "strategy": "combined",
  "temperature": 0.8,
  "top_p": 0.9,
  "repetition_penalty": 1.2
}
```

**Response:**
```json
{
  "generated_text": "The future of artificial intelligence is bright, with advances in...",
  "tokens_generated": 45,
  "strategy_used": "combined"
}
```

#### **POST /compare** — Compare All Strategies
Runs the same prompt through all 5 generation strategies side-by-side for comparison.

### Interactive Documentation

FastAPI automatically generates interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs` — Interactive testing interface
- **ReDoc**: `http://localhost:8000/redoc` — Pretty documentation

No Postman needed — test directly in your browser!

---

## Project Structure

```
text-generation-mlops/
│
├── data/
│   ├── raw/                          # Immutable downloaded dataset
│   └── processed/                    # Tokenized tensors (.pt files)
│
├── docs/
│   ├── TRAINING_GUIDE.md             # Comprehensive training guide
│   ├── TESTING_GUIDE.md              # Model evaluation guide
│   ├── GENERATION_STRATEGIES_GUIDE.md # Text generation deep dive
│   ├── FASTAPI_GUIDE.md              # FastAPI tutorial
│   └── SETUP_GUIDE.md                # Deployment guide
│
├── runs/
│   └── experiment_1/
│       ├── metrics.jsonl             # Per-epoch training + validation metrics
│       ├── test_results.yaml         # Final test set results
│       └── checkpoints/
│           ├── last.pt               # Latest checkpoint (resumption)
│           └── best.pt               # Best checkpoint (inference)
│
├── src/
│   ├── config/
│   │   └── config.yaml               # Hyperparameters
│   │
│   ├── data/
│   │   └── dataset.py                # Data pipeline
│   │
│   ├── model/
│   │   ├── decoder_transformer.py    # Top-level model
│   │   ├── transformer.py            # Transformer block
│   │   └── positional_embedding.py   # Positional embeddings
│   │
│   ├── training/
│   │   ├── train.py                  # Training entry point
│   │   ├── test.py                   # Test evaluation
│   │   ├── trainer.py                # Training loop
│   │   ├── evaluator.py              # Validation loop
│   │   ├── logger.py                 # Metric logging
│   │   └── checkpoint.py             # Checkpoint utilities
│   │
│   ├── inference/
│   │   ├── text_generator.py         # Generation engine (5 strategies)
│   │   └── test_generator.py         # Test generation strategies
│   │
│   ├── api/
│   │   ├── schema                    # API schema
│   │   │   └── schema.py 
│   │   ├── api_client.py             # Python client library
│   │   └── api.py                    # FastAPI application
│   │
│   └── utils/
│       └── seed.py                   # Reproducibility
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Configuration

All hyperparameters are centralised in `src/config/config.yaml`:

```yaml
device: cuda

data:
  dataset_name: wikitext
  dataset_config: wikitext-2-raw-v1
  seq_length: 128

model:
  dim_model: 256
  num_heads: 8
  num_layers: 4
  dim_ff: 1024

training:
  epochs: 20
  batch_size: 32
  learning_rate: 3.0e-4
  weight_decay: 0.05
  warmup_steps: 200
  num_steps: 5000
  dropout: 0.2
  patience: 10           # early stopping patience (epochs)
```

---

## Quickstart

### Training

```bash
# 1. Clone the repo
git clone https://github.com/athul-v-nair/text-generation-mlops.git
cd text-generation-mlops

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run data pipeline (downloads WikiText-2, tokenizes, caches to disk)
python src/data/dataset.py

# 4. Train (runs full training + validation loop with early stopping)
python src/training/train.py

# 5. Test the trained model
python src/training/test.py

# 6. Monitor training metrics
python -c "
import pandas as pd
df = pd.read_json('runs/experiment_1/metrics.jsonl', lines=True)
print(df[['epoch','train_loss','val_loss','val_perplexity']].to_string())
"
```

### Inference & API

```bash
# 1. Install API dependencies
pip install -r requirements_api.txt

# 2. Test text generation strategies
python src/inference/test_generator.py

# 3. Start the API server
uvicorn src.api.api:app --reload --host 0.0.0.0 --port 8000

# 4. Test the API
# Visit: http://localhost:8000/docs (Interactive Swagger UI)
# Or open: src/api/frontend.html (Web interface)

# 5. Use Python client
python -c "
from src.api.api_client import TextGenClient
client = TextGenClient('http://localhost:8000')
result = client.generate('The future of AI is', max_length=50)
print(result['generated_text'])
"
```

**📖 For detailed instructions:**
- **Training**: See [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for hyperparameter tuning, monitoring, and troubleshooting
- **Testing**: See [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for evaluating your model on test data

---

## Roadmap


- [x] Tokenization pipeline (WikiText-2)
- [x] Decoder-only Transformer (attention, FFN, causal masking, Pre-LN)
- [x] Training loop with gradient clipping, AdamW, LR scheduling
- [x] Validation loop with perplexity computation
- [x] Early stopping with patience
- [x] Dual checkpointing (`last.pt` / `best.pt`)
- [x] JSONL metric logging (decoupled Logger)
- [x] Test set evaluation
- [x] Comprehensive training and testing guides
- [x] Text generation with 5 sampling strategies
- [x] Repetition penalty for better outputs
- [x] REST API serving with FastAPI
- [x] Interactive Swagger UI documentation
- [x] Web frontend for testing
- [x] Python client library
- [x] Complete inference documentation
- [ ] Experiment tracking (MLflow / Weights & Biases)
- [ ] Loss curve visualization script
- [ ] Token streaming (real-time generation)
- [ ] Dockerized deployment

---

## Dependencies

### Core Training

| Package | Purpose |
|---|---|
| `torch` | Model, training, tensor ops |
| `datasets` | WikiText-2 download and caching |
| `transformers` | GPT-2 tokenizer only |
| `pyyaml` | Config loading |
| `numpy` | Seeding, preprocessing utils |

### API & Inference

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response validation |
| `requests` | Client library HTTP calls |

---

## Performance

**Training Results (WikiText-2):**
- Model size: 2.7M parameters
- Final perplexity: ~167
- Training time: ~6 hours (CPU, 11 epochs)
- Hardware: CPU-only training

**Inference Performance:**
- Greedy: ~50 tokens/sec (CPU)
- Combined (temp+top-p): ~45 tokens/sec (CPU)
- API latency: ~100ms for 50 tokens (CPU)

---

## References & Resources

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://openai.com/research/language-unsupervised) (GPT-2, 2019)
- Xiong et al., [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745) (Pre-LN analysis, 2020)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## Author

**Athul V Nair** — [GitHub](https://github.com/athul-v-nair)

*Built to understand Transformers from first principles and demonstrate production ML engineering practices — from training to serving.*

---

## Acknowledgments

- OpenAI for the GPT-2 tokenizer
- Hugging Face for the datasets library
- The PyTorch team for an excellent framework
- The FastAPI community for the modern web framework

---