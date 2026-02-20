# Testing Your Trained Model - Complete Guide

## Quick Start

After training your model with `train.py`, run:

```bash
python src/training/test.py
```

This will:
1. Load the best saved checkpoint (`runs/experiment_1/checkpoints/best.pt`)
2. Evaluate on the WikiText-2 test set
3. Print test loss and perplexity
4. Save results to `runs/experiment_1/test_results.yaml`

---

## File Organization

You need to add these files to your project:

### 1. `src/training/checkpoint.py` (NEW - Required)
Contains `save_checkpoint()` and `load_checkpoint()` functions that your trainer already imports.

### 2. `src/training/test.py` (NEW - Main testing script)
Standalone script to evaluate your trained model on test data.

---

## What Happens During Testing

```
1. Load config.yaml
   ↓
2. Initialize test dataset (WikiText-2 test split)
   ↓
3. Initialize model architecture (same as training)
   ↓
4. Load weights from best.pt checkpoint
   ↓
5. Run evaluation loop (no gradients)
   ↓
6. Calculate test loss & perplexity
   ↓
7. Print results & save to test_results.yaml
```

---

## Expected Output

```
Using device: cpu
Loading test dataset...
Loading tokenized data from data/processed/test_tokens.pt
Initializing model architecture...
Loading checkpoint from runs/experiment_1/checkpoints/best.pt...
Loaded model from epoch 2
Model was trained for 1500 steps

Evaluating on test set...

==================================================
TEST SET RESULTS
==================================================
Test Loss:       3.2456
Test Perplexity: 25.67
==================================================

Test results saved to runs/experiment_1/test_results.yaml
```

---

## Understanding the Metrics

### Test Loss
- Cross-entropy loss on the test set
- Lower is better
- Typical range: 2.0 - 5.0 for small models on WikiText-2

### Test Perplexity
- `PPL = exp(test_loss)`
- Measures how "surprised" the model is by test data
- Lower is better
- Example: PPL of 25 means the model is as uncertain as randomly picking from 25 tokens

---

## Comparing Train vs Test

After testing, compare your results:

```bash
# View training metrics
python -c "
import pandas as pd
df = pd.read_json('runs/experiment_1/metrics.jsonl', lines=True)
print(df[['epoch', 'train_loss', 'val_loss', 'val_perplexity']].tail())
"

# View test results
cat runs/experiment_1/test_results.yaml
```

**Healthy model behavior:**
- Test perplexity should be close to validation perplexity
- If test PPL >> val PPL: Model may have overfit to validation set
- If test PPL ≈ val PPL: Good generalization!

---

## Testing with Last Checkpoint (Alternative)

If you want to test the **last** checkpoint instead of the **best**:

```python
# In test.py, change this line:
checkpoint_path = "runs/experiment_1/checkpoints/last.pt"
```

---

## Troubleshooting

### Error: "Checkpoint not found"
**Solution:** Train your model first:
```bash
python src/training/train.py
```

### Error: "No module named 'src.training.checkpoint'"
**Solution:** Create `src/training/checkpoint.py` with the provided code.

### Test perplexity is very high (>100)
**Possible causes:**
- Model undertrained (train for more epochs)
- Learning rate too high/low
- Model too small for the task

### GPU out of memory during testing
**Solution:** Reduce batch size in config.yaml:
```yaml
training:
  batch_size: 16  # or even smaller
```

---

## Advanced: Resume Training Then Test

If you stopped training early and want to continue:

```python
# Add to train.py after initializing model/optimizer/scheduler:

from src.training.checkpoint import load_checkpoint

checkpoint_path = "runs/experiment_1/checkpoints/last.pt"
if os.path.exists(checkpoint_path):
    global_step, start_epoch = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path, device
    )
    print(f"Resuming from epoch {start_epoch}")
else:
    start_epoch = 0

# Then modify your training loop:
for epoch in range(start_epoch, config['training']['epochs']):
    # ... rest of training code
```

---

## Next Steps

After testing, you can:

1. **Analyze results:** Compare train/val/test perplexities
2. **Tune hyperparameters:** Adjust learning rate, model size, etc.
3. **Generate text:** Write an inference script to generate text
4. **Visualize metrics:** Plot learning curves from metrics.jsonl

---

## File Checklist

Before testing, ensure you have:

- [ ] `src/training/test.py` ✓
- [ ] `src/training/checkpoint.py` ✓
- [ ] `runs/experiment_1/checkpoints/best.pt` (created by training)
- [ ] `data/processed/test_tokens.pt` (auto-created on first test run)

---

**You're ready to test!** Run `python src/training/test.py` after training completes.