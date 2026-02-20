import torch
import torch.optim as optim
import yaml,os
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

from src.model.decoder_transformer import DecoderTransformer
from src.training.trainer import Trainer
from src.training.logger import Logger
from src.training.evaluator import Evaluator
from src.data.dataset import TextGenerationDataset
from src.utils.seed import set_seed

from src.constants import CONFIG_PATH

def train():
    set_seed(42)

    with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

    device = torch.device(
        config['device'] if torch.cuda.is_available() else "cpu"
    )

    run_dir = "runs/experiment_1"
    os.makedirs(run_dir, exist_ok=True)

    # ---- Dataset ----
    train_dataset = TextGenerationDataset(split="train", config_path= CONFIG_PATH)
    val_dataset = TextGenerationDataset(split="validation", config_path= CONFIG_PATH)
    
    vocabulary_size = train_dataset.vocab_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    # Initialize Transformer decoder model with config hyperparameters
    model = DecoderTransformer(
        vocabulary_size=vocabulary_size,
        max_seq_len=config['data']['seq_length'],
        dim_model=config['training']['dim_model'],
        num_layers=config['training']['num_layers'],
        num_heads=config['training']['num_heads'],
        dim_ff=config['training']['dim_ff'],
        dropout=config['training']['dropout']
    ).to(device)

    # Initialize AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay']
    )
    
    # Linear warmup scheduler: gradually increase LR at start
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=config['training']['warmup_steps'])
    
    # Cosine decay scheduler: smoothly decrease LR after warmup
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_steps'] - config['training']['warmup_steps'])
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config['training']['warmup_steps']]
    )

    # Initializing logger and evaluator
    logger = Logger(run_dir)
    evaluator = Evaluator(model, config, device)
    
    # Initializing Trainer
    trainer = Trainer(model, optimizer, scheduler, config, device, evaluator, logger, run_dir)

    # Start the training
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    train()