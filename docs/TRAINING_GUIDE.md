# Training Your Transformer Model - Complete Guide

## Quick Start

Train the model with default settings:

```bash
python src/training/train.py
```

The model will train for 3 epochs on WikiText-2 and automatically save checkpoints.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Training Pipeline Overview](#training-pipeline-overview)
- [Configuration](#configuration)
- [Understanding the Training Process](#understanding-the-training-process)
- [Monitoring Training](#monitoring-training)
- [Checkpoints](#checkpoints)
- [Early Stopping](#early-stopping)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Common Issues](#common-issues)
- [Training on GPU](#training-on-gpu)

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print('Transformers installed')"
```

### 3. Prepare Data (Optional - Auto-runs on first training)

```bash
python src/data/dataset.py
```

This downloads WikiText-2 and tokenizes it. Files are cached in:
- `data/raw/` - Original dataset
- `data/processed/` - Tokenized tensors

---

## Training Pipeline Overview

```
1. Load Configuration (config.yaml)
   ↓
2. Initialize Dataset & DataLoader
   ↓
3. Initialize Model (DecoderTransformer)
   ↓
4. Initialize Optimizer (AdamW)
   ↓
5. Initialize Scheduler (Warmup + Cosine)
   ↓
6. Training Loop (for each epoch):
   │
   ├─→ Train on all batches
   │   ├─ Forward pass
   │   ├─ Compute loss
   │   ├─ Backward pass
   │   ├─ Gradient clipping
   │   ├─ Optimizer step
   │   └─ Scheduler step
   │
   ├─→ Validate on validation set
   │   ├─ Compute validation loss
   │   └─ Compute perplexity
   │
   ├─→ Log metrics (metrics.jsonl)
   │
   ├─→ Save checkpoint (last.pt)
   │
   ├─→ Save best checkpoint (if improved)
   │
   └─→ Early stopping check
```

---

## Configuration

All hyperparameters are in `src/config/config.yaml`:

```yaml
data:
  dataset_name: wikitext
  dataset_config: wikitext-2-raw-v1
  seq_length: 128              # Length of input sequences

training:
  epochs: 3                     # Number of training epochs
  batch_size: 2                 # Batch size (reduce if OOM)
  
  # Model architecture
  dim_model: 256               # Embedding dimension
  num_layers: 4                # Number of transformer blocks
  num_heads: 4                 # Number of attention heads
  dim_ff: 1024                 # Feed-forward hidden dimension
  dropout: 0.1                 # Dropout rate
  
  # Optimization
  learning_rate: 3e-4          # Peak learning rate
  weight_decay: 0.01           # L2 regularization
  
  # Learning rate schedule
  num_steps: 3000              # Total training steps
  warmup_steps: 300            # Warmup steps (10% of total)
  
  # Early stopping
  patience: 2                  # Epochs without improvement before stopping

device: "cpu"                  # "cpu" or "cuda"
```

---

## Understanding the Training Process

### Training Step

Each training step performs:

1. **Forward Pass**
   ```python
   logits = model(input_ids)  # Shape: (B, T, vocab_size)
   ```

2. **Target Shifting**
   ```python
   # Predict token t+1 from tokens ≤ t
   logits = logits[:, :-1, :]   # Drop last prediction
   targets = input_ids[:, 1:]   # Drop first token
   ```

3. **Loss Calculation**
   ```python
   loss = CrossEntropyLoss(logits, targets)
   ```

4. **Backward Pass**
   ```python
   optimizer.zero_grad()
   loss.backward()
   clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()
   scheduler.step()
   ```

### Validation Step

After each epoch:

1. **Model Evaluation Mode**
   ```python
   model.eval()  # Disables dropout
   ```

2. **No Gradient Tracking**
   ```python
   with torch.no_grad():
       # Compute validation loss
   ```

3. **Metrics Calculation**
   - Average loss per token
   - Perplexity: `exp(avg_loss)`

---

## Monitoring Training

### Real-time Console Output

During training, you'll see:

```
Starting training epoch: 0
Starting training step: 0
Starting training step: 1
...
Saved best model!!

Starting training epoch: 1
...
Early stopping triggered.
```

### Metrics File

All metrics are logged to `runs/experiment_1/metrics.jsonl`:

```jsonl
{"epoch": 0, "train_loss": 6.234, "val_loss": 5.891, "val_perplexity": 362.45, ...}
{"epoch": 1, "train_loss": 5.123, "val_loss": 5.234, "val_perplexity": 187.23, ...}
{"epoch": 2, "train_loss": 4.567, "val_loss": 5.012, "val_perplexity": 150.67, ...}
```

### Viewing Metrics

```bash
# View all metrics
python -c "
import pandas as pd
df = pd.read_json('runs/experiment_1/metrics.jsonl', lines=True)
print(df.to_string())
"

# View specific columns
python -c "
import pandas as pd
df = pd.read_json('runs/experiment_1/metrics.jsonl', lines=True)
print(df[['epoch', 'train_loss', 'val_loss', 'val_perplexity']])
"

# Plot learning curves
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('runs/experiment_1/metrics.jsonl', lines=True)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['val_perplexity'])
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Validation Perplexity')

plt.tight_layout()
plt.savefig('training_curves.png')
print('Saved to training_curves.png')
"
```

---

## Checkpoints

Two checkpoints are saved during training:

### 1. Last Checkpoint (`last.pt`)
- Saved **every epoch**
- Location: `runs/experiment_1/checkpoints/last.pt`
- Purpose: Resume training if interrupted

### 2. Best Checkpoint (`best.pt`)
- Saved **only when validation loss improves**
- Location: `runs/experiment_1/checkpoints/best.pt`
- Purpose: Use for inference and testing

### Checkpoint Contents

Each checkpoint contains:
```python
{
    'model_state_dict': ...,      # Model weights
    'optimizer_state_dict': ...,  # Optimizer state
    'scheduler_state_dict': ...,  # Scheduler state
    'global_step': ...,           # Total training steps
    'epoch': ...                  # Current epoch
}
```

### Resuming Training

To resume from `last.pt`, modify `train.py`:

```python
# After initializing model, optimizer, scheduler:
from src.training.checkpoint import load_checkpoint

checkpoint_path = "runs/experiment_1/checkpoints/last.pt"
if os.path.exists(checkpoint_path):
    global_step, start_epoch = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path, device
    )
    trainer.global_step = global_step
else:
    start_epoch = 0

# Update training loop:
for epoch in range(start_epoch, config['training']['epochs']):
    # ... training code
```

---

## Early Stopping

### How It Works

The trainer monitors **validation loss** and stops if it doesn't improve:

```python
patience = 2  # Wait 2 epochs for improvement

if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint("best.pt")
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1

if epochs_without_improvement >= patience:
    print("Early stopping triggered.")
    break
```

### Why Use Early Stopping?

- **Prevents overfitting**: Stop before model memorizes training data
- **Saves time**: No wasted compute after peak performance
- **Automatic**: No manual intervention needed

### Adjusting Patience

In `config.yaml`:
```yaml
training:
  patience: 5  # More patient (default: 2)
```

- **Lower patience (1-2)**: Stops quickly, good for fast iteration
- **Higher patience (5-10)**: More forgiving, useful if validation is noisy

---

## Hyperparameter Tuning

### Key Parameters to Tune

#### 1. Learning Rate
```yaml
learning_rate: 3e-4  # Default
```
- **Too high**: Training unstable, loss spikes
- **Too low**: Training very slow, may not converge
- **Good range**: 1e-4 to 5e-4 for small models

#### 2. Model Size
```yaml
dim_model: 256       # Embedding dimension
num_layers: 4        # Transformer blocks
num_heads: 4         # Attention heads
```
- **Larger model**: Better performance, more memory, slower
- **Smaller model**: Faster, less memory, may underfit

#### 3. Batch Size
```yaml
batch_size: 2  # Very small for CPU
```
- **Larger**: Faster training, more stable gradients, needs more memory
- **Smaller**: Slower, noisier gradients, less memory
- **Rule of thumb**: Largest that fits in memory

#### 4. Warmup Steps
```yaml
warmup_steps: 300  # 10% of total steps
```
- **Purpose**: Prevents unstable early training
- **Typical**: 5-10% of total training steps

#### 5. Dropout
```yaml
dropout: 0.1
```
- **Higher (0.2-0.3)**: More regularization, prevents overfitting
- **Lower (0.05-0.1)**: Less regularization, better if underfitting

### Tuning Strategy

1. **Start with defaults** - Verify training works
2. **Tune learning rate** - Most impactful parameter
3. **Scale model** - Adjust size based on performance/resources
4. **Adjust regularization** - If overfitting (high val loss)
5. **Fine-tune batch size** - Balance speed and stability

---

## Common Issues

### 1. Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory` or system freeze

**Solutions**:
```yaml
# Reduce batch size
batch_size: 1

# Reduce sequence length
seq_length: 64

# Reduce model size
dim_model: 128
num_layers: 2
```

### 2. Loss Not Decreasing

**Symptom**: Loss stays constant or increases

**Solutions**:
- Check learning rate (may be too low)
- Verify data preprocessing (check tokenization)
- Increase model capacity
- Train for more epochs

### 3. Loss Spikes / NaN

**Symptom**: Loss suddenly jumps to very high values or NaN

**Solutions**:
```yaml
# Reduce learning rate
learning_rate: 1e-4

# Ensure gradient clipping is enabled (it is by default)
# In trainer.py:
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### 4. Overfitting

**Symptom**: Training loss much lower than validation loss

**Solutions**:
```yaml
# Increase dropout
dropout: 0.2

# Add more weight decay
weight_decay: 0.05

# Use early stopping (already enabled)
patience: 3
```

### 5. Very Slow Training

**Solutions**:
- Switch to GPU (see next section)
- Reduce model size
- Increase batch size
- Use mixed precision training (FP16)

---

## Training on GPU

### 1. Update Configuration

```yaml
device: "cuda"
```

### 2. Increase Batch Size

GPUs have more memory:
```yaml
batch_size: 32  # or 64, 128
```

### 3. Scale Model

You can afford larger models:
```yaml
dim_model: 512
num_layers: 6
num_heads: 8
dim_ff: 2048
```

### 4. Monitor GPU Usage

```bash
# While training is running:
watch -n 1 nvidia-smi
```

### 5. Expected Speedup

- **CPU (default config)**: ~5-10 min/epoch
- **GPU (scaled config)**: ~30-60 sec/epoch

---

## Training Workflow Example

### First Training Run

```bash
# 1. Start training with defaults
python src/training/train.py

# 2. Monitor progress
tail -f runs/experiment_1/metrics.jsonl

# 3. Check results after completion
python -c "
import pandas as pd
df = pd.read_json('runs/experiment_1/metrics.jsonl', lines=True)
print(df[['epoch', 'train_loss', 'val_loss', 'val_perplexity']])
"
```

### Iterative Tuning

```bash
# Run 1: Baseline (default config)
python src/training/train.py
# Result: val_ppl = 150

# Run 2: Increase learning rate
# Edit config.yaml: learning_rate: 5e-4
python src/training/train.py
# Result: val_ppl = 140 (better!)

# Run 3: Add capacity
# Edit config.yaml: num_layers: 6, dim_model: 512
python src/training/train.py
# Result: val_ppl = 120 (even better!)
```

---

## Next Steps

After successful training:

1. **Test the model**: See [TESTING_GUIDE.md](TESTING_GUIDE.md)
2. **Generate text**: Create an inference script
3. **Experiment tracking**: Integrate MLflow or Weights & Biases
4. **Deployment**: Build a REST API with FastAPI

---

## Quick Reference

### Essential Commands

```bash
# Train model
python src/training/train.py

# View metrics
python -c "import pandas as pd; print(pd.read_json('runs/experiment_1/metrics.jsonl', lines=True))"

# Check checkpoint
python -c "import torch; ckpt = torch.load('runs/experiment_1/checkpoints/best.pt', map_location='cpu'); print(f'Epoch: {ckpt[\"epoch\"]}, Step: {ckpt[\"global_step\"]}')"
```

### File Locations

- Config: `src/config/config.yaml`
- Training script: `src/training/train.py`
- Metrics: `runs/experiment_1/metrics.jsonl`
- Checkpoints: `runs/experiment_1/checkpoints/`

---

**Ready to train!** Run `python src/training/train.py` to get started.