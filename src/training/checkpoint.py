import torch, os

def save_checkpoint(model, optimizer, scheduler, step, path, epoch):
    """
    Save model checkpoint with all training state.
    
    Args:
        model: The neural network model
        optimizer: The optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        global_step: Total number of training steps completed
        path: Full path where checkpoint should be saved
        epoch: Current epoch number
    """
    
    # check if directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    """
    Load model checkpoint and restore training state.
    
    Args:
        model: The neural network model (initialized architecture)
        optimizer: The optimizer (initialized)
        scheduler: Learning rate scheduler (initialized)
        path: Path to the checkpoint file
        device: Device to load the checkpoint to ('cpu' or 'cuda')
    
    Returns:
        tuple: (global_step, epoch) from the checkpoint
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    global_step = checkpoint.get('global_step', 0)
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Checkpoint loaded from {path}")
    print(f"Resuming from epoch {epoch}, step {global_step}")
    
    return global_step, epoch