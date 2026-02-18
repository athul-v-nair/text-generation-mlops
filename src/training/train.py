import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.model.decoder_transformer import DecoderTransformer
from src.training.trainer import Trainer
from src.training.checkpoint import save_checkpoint

CONFIG_PATH='src/config/config.yaml'

def train():
    with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

    device = torch.device(
        config['device'] if torch.cuda.is_available() else "cpu"
    )

    # Initialize Transformer decoder model with config hyperparameters
    model = DecoderTransformer(
        vocabulary_size=config['training']['vocabulary_size'],
        max_seq_len=config['training']['max_seq_len'],
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

    trainer = Trainer(model, optimizer, scheduler, config, device)

    # Create dummy batch for overfitting sanity check
    # Random integers represent token IDs
    dummy_batch = torch.randint(
        0,
        config['training']['vocabulary_size'],
        (config['training']['batch_size'], config['training']['max_seq_len'])
    ).to(device)

    # Training Loop
    for step in range(config['training']['num_steps']):
        loss = trainer.train_step(dummy_batch)

        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss:.4f}")

    # Save final model checkpoint after training
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        config['training']['num_steps'],
        config['training']['checkpoint_path']
    )


if __name__ == "__main__":
    train()