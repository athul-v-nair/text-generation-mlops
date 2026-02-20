import torch
import os, json, yaml
from torch.utils.data import DataLoader

from src.model.decoder_transformer import DecoderTransformer
from src.training.evaluator import Evaluator
from src.data.dataset import TextGenerationDataset

from src.constants import CONFIG_PATH
from src.constants import CURRENT_RUN

def test():
    """
    Load best trained model and evaluate on test set.
    """
    # Load configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        config['device'] if torch.cuda.is_available() else "cpu"
    )

    # ---- Load Test Dataset ----
    print("Loading test dataset...")
    test_dataset = TextGenerationDataset(split="test", config_path=CONFIG_PATH)
    vocabulary_size = test_dataset.vocab_size

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    # ---- Initialize Model Architecture ----
    print("Initializing model architecture...")
    model = DecoderTransformer(
        vocabulary_size=vocabulary_size,
        max_seq_len=config['data']['seq_length'],
        dim_model=config['training']['dim_model'],
        num_layers=config['training']['num_layers'],
        num_heads=config['training']['num_heads'],
        dim_ff=config['training']['dim_ff'],
        dropout=config['training']['dropout']
    ).to(device)

    # Loading the best model
    checkpoint_path = CURRENT_RUN + 'checkpoints/best.pt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Please train the model first using train.py"
        )
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load only model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Model was trained for {checkpoint.get('step', 'unknown')} steps")

    # ---- Initialize Evaluator ----
    evaluator = Evaluator(model, config, device)

    # ---- Run Evaluation on Test Set ----
    print("\nEvaluating on test set...")
    test_loss, test_ppl = evaluator.evaluate(test_loader)

    print("~"*20)
    print("Test Results")
    print("~"*20)
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test Perplexity: {test_ppl:.2f}")
    print("~"*20)

    # ---- Save Test Results ----
    results = {
        "test_loss": test_loss,
        "test_perplexity": test_ppl,
        "checkpoint_epoch": checkpoint.get('epoch', None),
        "checkpoint_step": checkpoint.get('global_step', None)
    }
    
    results_path = CURRENT_RUN + "test_results.jsonl"
    with open(results_path, 'w') as f:
        f.write(json.dumps(results) + "\n")
    
    print(f"\nTest results saved to {results_path}")

    return test_loss, test_ppl

if __name__ == "__main__":
    test()