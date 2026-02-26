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
from src.constants import CURRENT_RUN

def train():
    set_seed(42)

    with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

    device = torch.device(
        config['device'] if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(CURRENT_RUN, exist_ok=True)

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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Initialize AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay']
    )
    
    # Calculate actual training steps
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['training']['epochs']
    warmup_steps = int(0.1 * total_steps)

    print(f"\nTraining schedule:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total epochs: {config['training']['epochs']}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Estimated time: {steps_per_epoch * 0.6 * config['training']['epochs'] / 3600:.1f} hours")
    print()

    # Linear warmup scheduler: gradually increase LR at start
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01,
        total_iters=warmup_steps
    )
    
    # Cosine decay scheduler: smoothly decrease LR after warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps # Use actual remaining steps
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Initializing logger and evaluator
    logger = Logger(CURRENT_RUN)
    evaluator = Evaluator(model, config, device)
    
    # Initializing Trainer
    trainer = Trainer(model, optimizer, scheduler, config, device, evaluator, logger, CURRENT_RUN)

    # Start the training
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    train()