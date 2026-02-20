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

The codebase is structured with MLOps discipline in mind: reproducible data pipelines, YAML-based configuration, token-weighted loss aggregation, perplexity tracking, early stopping, dual checkpointing (`last.pt` / `best.pt`), and structured JSONL metric logging.

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

$$g \leftarrow g \cdot \frac{\text{max\_norm}}{\|g\|} \quad \text{where} \quad \|g\| = \sqrt{\sum_i \|g_i\|^2}$$

An observed pre-clip gradient norm of ~220 confirms this is a necessary safeguard for stable Transformer training.

### Optimizer

AdamW is used with configurable weight decay:

```python
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

SGD was tested and produced large loss oscillations. AdamW's per-parameter adaptive scaling is critical for Transformers, where gradient magnitudes vary significantly across layers.

### Learning Rate Schedule

**Linear warmup** — ramps LR from 1% of peak to full over `warmup_steps`:

$$lr_t = lr_{\max} \cdot \frac{t}{\text{warmup\_steps}}$$

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

$$\text{PPL} = \exp\!\left(\frac{1}{N}\sum_{t=1}^{N} -\log P(x_{t+1} \mid x_{\leq t})\right) = e^{\mathcal{L}_{\text{avg}}}$$

Intuitively, PPL represents the effective number of equally likely tokens the model is "choosing between" at each position. A PPL of 100 means the model is as uncertain as if it were picking uniformly from 100 tokens. Lower is better.

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

## Project Structure

```
text-generation-mlops/
│
├── data/
│   ├── raw/                          # Immutable downloaded dataset (never modified)
│   └── processed/                    # Tokenized tensors (.pt files)
│
├── runs/
│   └── experiment_1/
│       ├── metrics.jsonl             # Per-epoch training + validation metrics
│       └── checkpoints/
│           ├── last.pt               # Latest epoch checkpoint (for resumption)
│           └── best.pt               # Best val_loss checkpoint (for inference)
│
├── src/
│   ├── config/
│   │   └── config.yaml               # Single source of truth for all hyperparameters
│   │
│   ├── data/
│   │   └── dataset.py                # Download → tokenize → cache pipeline
│   │
│   ├── model/
│   │   ├── transformer.py            # Top-level model (embedding + block stack + head)
│   │   ├── decoder_transformer.py    # Transformer block (attention + FFN + residuals)
│   │   └── positional_embedding.py  # Learnable positional embeddings
│   │
│   ├── training/
│   │   ├── train.py                  # Entry point: wires dataset, model, optimizer, trainer
│   │   ├── trainer.py                # Training loop: train_step, train_epoch, fit
│   │   ├── evaluator.py              # Validation loop + perplexity computation
│   │   ├── logger.py                 # JSONL metric logging (decoupled from Trainer)
│   │   └── checkpoint.py            # save_checkpoint / load_checkpoint utilities
│   │
│   └── utils/
│       └── seed.py                   # Global reproducibility seeding
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
  weight_decay: 0.01
  warmup_steps: 200
  num_steps: 5000
  dropout: 0.1
  patience: 5           # early stopping patience (epochs)
```

---

## Quickstart

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

# 5. Monitor training metrics
python -c "
import pandas as pd
df = pd.read_json('runs/experiment_1/metrics.jsonl', lines=True)
print(df[['epoch','train_loss','val_loss','val_perplexity']].to_string())
"
```

---

## Roadmap

- [x] Tokenization pipeline (WikiText-2)
- [x] Decoder-only Transformer (attention, FFN, causal masking, Pre-LN)
- [x] Training loop with gradient clipping, AdamW, LR scheduling
- [x] Validation loop with perplexity computation
- [x] Early stopping with patience
- [x] Dual checkpointing (`last.pt` / `best.pt`)
- [x] JSONL metric logging (decoupled Logger)
- [ ] Experiment tracking (MLflow / Weights & Biases)
- [ ] Loss curve visualisation script
- [ ] REST API serving (FastAPI)
- [ ] Dockerized deployment
- [ ] CI/CD pipeline (GitHub Actions)

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model, training, tensor ops |
| `datasets` | WikiText-2 download and caching |
| `transformers` | GPT-2 tokenizer only |
| `pyyaml` | Config loading |
| `numpy` | Seeding, preprocessing utils |

---

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://openai.com/research/language-unsupervised) (GPT-2, 2019)
- Xiong et al., [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745) (Pre-LN analysis, 2020)

---

## Author

**Athul V Nair** — [GitHub](https://github.com/athul-v-nair)

*Built to understand Transformers from first principles and demonstrate production ML engineering practices.*