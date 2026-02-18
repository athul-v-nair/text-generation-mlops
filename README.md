# 🧠 Text Generation Transformer (From Scratch + MLOps)

A minimal, production-structured decoder-only Transformer built in PyTorch and trained on WikiText-2.

This project is designed to:

- Deepen understanding of Transformer mechanics  
- Implement clean ML engineering practices  
- Apply reproducible data pipelines  
- Incrementally introduce MLOps discipline  

---

## 📌 Project Goals

- Build a decoder-only Transformer from scratch (using PyTorch primitives)  
- Train using next-token prediction objective  
- Implement structured data pipeline (raw → processed)  
- Ensure reproducibility  
- Prepare foundation for experiment tracking and deployment  

---

## 📂 Project Structure

```bash
text-generation-mlops/
│
├── data/
│   ├── raw/           # Immutable source dataset
│   ├── processed/     # Tokenized tensors (.pt files)
│
├── src/
│   ├── config/
│   │   └── config.yaml
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   └── transformer.py
│   │   └── decoder_transformer.py
│   │   └── positional_embedding.py
│   ├── training/
│   ├── inference/
│   ├── utils/
│   │   └── seed.py
│
├── tests/
├── api/
├── docker/
├── requirements.txt
└── README.md
```
---

### 📚 Dataset

We use:

**WikiText-2 (`wikitext-2-raw-v1`)**  
A standard language modeling benchmark dataset.

---

### 🔄 Data Pipeline

#### First Run

1. Dataset is downloaded using HuggingFace `datasets`
2. Saved to `data/raw/`
3. Tokenized using GPT-2 tokenizer
4. Token tensors saved to `data/processed/`

#### Subsequent Runs

- Raw dataset loaded from disk  
- Tokenized tensors loaded directly (no reprocessing)

This ensures:

- Reproducibility  
- Faster iteration  
- Clean raw vs processed separation  

---

### 🔤 Tokenization Strategy

We use the GPT-2 tokenizer.

Each document is tokenized individually and concatenated into a single continuous token stream.

```python
self.input_ids = torch.cat(all_input_ids, dim=0)
```

This produces a long 1D tensor:

```python
[t1, t2, t3, ..., tN]
```

### 🧮 Language Modeling Objective

We train using **causal next-token prediction**.

For a sequence:

```python  
[t1, t2, t3, t4]
```

Input (x):

```python
[t1, t2, t3, t4]
```

Target (y):

```python   
[t2, t3, t4, t5]
```

This is implemented as:

```python
x = input_ids[start:end]  y = input_ids[start + 1:end + 1]
```

Mathematical Formulation
------------------------

Given a token sequence:

```ini
x=(x1,x2,...,xT)x = (x\_1, x\_2, ..., x\_T)x=(x1​,x2​,...,xT​)
```

The model is trained to maximize:

```ini
∏_{t=1}^{T} P(x_{t+1} | x_1, ..., x_t)
```

Loss function used:

```ini
L = - ∑_{t=1}^{T} log P(x_{t+1} | x_{≤t})
```

This is equivalent to **Cross-Entropy Loss** over next-token predictions.

### 📦 Dataset Construction

Sequences are chunked into fixed-length blocks:

If:

*   Total tokens = NNN
*   Sequence length = LLL

Then:

```ini
num_sequences = floor((N - 1) / L)
```

This ensures valid shifted targets.

Chunks are non-overlapping:

```csharp
[0:L]  [L:2L]  [2L:3L]  ...   
```

This matches standard GPT-style training.

### 🔁 Reproducibility

We fix all major randomness sources:

```python
random.seed(seed)  np.random.seed(seed)  torch.manual_seed(seed)  torch.cuda.manual_seed_all(seed)   
```

This ensures consistent:

*   Weight initialization
*   Data shuffling
*   Dropout behavior (as much as possible)

Reproducibility is critical for ML system reliability.

### ⚙️ Configuration Management

Hyperparameters are stored in:

```plain  
src/config/config.yaml   
```

Example:

```plain  
data:    dataset_name: wikitext    dataset_config: wikitext-2-raw-v1    seq_length: 128  training:    batch_size: 32   
```
#### ✅ Update - 1

The tokenization pipeline is implemented

##  Transformer Architecture Implementation

### 🏗 Model Overview

**Architecture**:

```bash
Input Tokens
    ↓
Token Embedding
    ↓
Positional Embedding
    ↓
N × Transformer Blocks
    ↓
LayerNorm
    ↓
Linear Projection (vocab)
    ↓
Logits (B, T, vocab_size)
```

🔢 **Tensor Shapes**

Notation:

```
B = batch size

T = sequence length

C = embedding dimension

H = number of attention heads

head_dim = C / H

Input tokens:

(B, T)

After embedding:

(B, T, C)

Output logits:

(B, T, vocab_size)
```

### 📍 Positional Embedding

Transformers are permutation invariant. They require explicit positional information.

We use learnable embeddings:

```python
self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
```

Added to token embeddings:

```python
x = token_embedding + positional_embedding
```

### 🧠 Multi-Head Self-Attention

Core formula:
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

Steps:

```
Linear projection → Q, K, V

Reshape into multiple heads

Scaled dot-product attention

Concatenate heads

Final projection

Scaling factor:

1 / sqrt(head_dim)
```

Prevents large dot-product values that destabilize softmax.

### 🔒 Causal Masking

Prevents attending to future tokens.

Lower-triangular mask:

torch.tril(torch.ones(T, T))

Ensures token t only attends to tokens ≤ t.

### 🔁 Transformer Block (Pre-LN)

Structure:

```python
x = x + Attention(LN(x))
x = x + FFN(LN(x))
```

**Why Pre-LN?**

More stable gradients

Better training dynamics

Residual connections allow gradient flow through deep stacks.

### 🧮 Feed Forward Network

Position-wise MLP:

```
Linear(C → 4C)
ReLU
Linear(4C → C)
```

Expands representation, applies non-linearity, projects back.

#### ✅ Update - 2

The model is now structurally correct and ready for training experiments.

## Training System Implementation

### Core training procedure:

```python
self.model.train()
logits = self.model(batch)
logits, targets = self.shift_logits_targets(logits, batch)
loss = CrossEntropy(logits, targets)
loss.backward()
clip_grad_norm_(parameters, 1.0)
optimizer.step()
scheduler.step()
Step-by-step Explanation

model.train()
Enables dropout and training-specific behavior.

Forward pass
Produces logits of shape (B, T, vocab_size).

Shift logits and targets
Aligns predictions with next-token targets.

Loss computation
Cross-entropy applied over reshaped tensors:

logits.reshape(-1, vocab_size)
targets.reshape(-1)

Backward pass
Computes gradients for all parameters.

Gradient clipping
Prevents exploding gradients.

Optimizer step
Updates parameters.

Scheduler step
Updates learning rate per training step.
```

### 🧮 Gradient Norm Monitoring

Observed total gradient norm:

≈ 220

Gradient norm formula:

```latex
∥g∥=∑i∥gi∥2\|g\| = \sqrt{\sum_i \|g_i\|^2}∥g∥=i∑​∥gi​∥2​
```

If norm exceeds threshold (1.0):

```latex
g←g⋅max_norm∥g∥g \leftarrow g \cdot \frac{\text{max\_norm}}{\|g\|}g←g⋅∥g∥max_norm​
```

This stabilizes deep Transformer training.

### 📉 Optimizer Experiments

#### Tested:

AdamW → Stable convergence

SGD → Large loss oscillations

Why AdamW Works Better

AdamW normalizes updates by gradient variance, which is critical for Transformers.

#### 📈 Learning Rate Scheduling


**Linear Warmup**

Gradually increases LR from near zero

```latex
lrt=lrmax⋅twarmup_stepslr_t = lr_{max} \cdot \frac{t}{\text{warmup\_steps}}lrt​=lrmax​⋅warmup_stepst​
```

Prevents unstable early updates.

**Cosine Annealing**

```latex
lrt=lrmin+12(lrmax−lrmin)(1+cos⁡(tTπ))lr_t = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})
\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)lrt​=lrmin​+21​(lrmax​−lrmin​)(1+cos(Tt​π))
```

Provides smooth decay and stable convergence.

#### 🧪 Overfitting Sanity Test

Before real training, we intentionally overfit a small random batch.

Purpose:

Verify gradient flow

Ensure loss decreases

Validate architecture correctness

If the model cannot overfit one batch → implementation bug.

Result: Loss decreased successfully.

### 💾 Checkpointing

Saved components:

model.state_dict()

optimizer.state_dict()

scheduler.state_dict()

Step count

This enables:

Training resumption

Experiment reproducibility

Model comparison

Checkpoints are excluded from Git using .gitignore.

#### ✅ Update - 3

Training loop successfully implemented.