import os, time, json
import torch
import torch.nn as nn

from src.training.checkpoint import save_checkpoint

class Trainer:
    def __init__(self, model, optimizer, scheduler, config, device, evaluator, logger ,run_dir):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
        # Instead of target probability = 1.0, use 0.9. Distributes 0.1 probability across other tokens
        
        self.evaluator = evaluator
        self.logger = logger

        self.run_dir = run_dir
        self.global_step=0
        self.best_val_loss = float("inf")

        # Ensure checkpoint directory exists
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)

    def shift_logits_targets(self, logits, input_ids):
        """
        Align predictions with next-token targets.

        logits:  (B, T, V)
        targets: (B, T)

        After shift:
            logits -> (B, T-1, V)
            targets -> (B, T-1)
        """
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        return logits, targets

    # Single training step
    def train_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        input_ids = batch["input_ids"]

        # Forward pass: compute model outputs (logits)
        logits=self.model(input_ids)

        # Shift logits and targets for next-token prediction (input t predicts token t+1)
        logits, targets = self.shift_logits_targets(logits, input_ids)

        # Compute cross-entropy loss
        # Reshape logits to 2D (batch*seq_len, vocab_size)
        # Reshape targets to 1D (batch*seq_len)
        loss=self.criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        # Clear previous gradients
        self.optimizer.zero_grad()
        # Backpropagation: compute gradients
        loss.backward()
        # Clip gradient norm to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Update model parameters using optimizer and update rate scheduler
        self.optimizer.step()        
        self.scheduler.step()

        # Add this for monitoring
        if self.global_step % 100 == 0:
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Step {self.global_step} | LR: {current_lr:.6f} | Loss: {loss.item():.4f}")
    

        self.global_step += 1

        # returns loss value and tokens count
        return loss.item(), targets.numel()     
    
    def train_epoch(self, dataloader, epoch):
        """
        Train model for full epochs as listed in config.
        Returns average loss per token.
        """
        # passed epoch jst for logging
        print(f"Starting training epoch: ", {epoch})

        # Set model to training mode (enables dropout, etc.)
        self.model.train()
        
        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            loss, token_count = self.train_step(batch)
            
            total_loss += loss * token_count
            total_tokens += token_count

        avg_loss = total_loss / total_tokens
        return avg_loss
    
    # Full training lifecycle
    def fit(self, train_loader, val_loader):
        patience = self.config["training"].get("patience", 5)
        epochs_without_improvement = 0
        

        for epoch in range(self.config['training']['epochs']):
            print(f"Starting training and validation epoch: ", {epoch})
            start_time = time.time()

            # training loss
            train_loss=self.train_epoch(train_loader, epoch)

            # validation loss and perplexity
            val_loss, val_ppl = self.evaluator.evaluate(val_loader)

            epoch_time = time.time() - start_time

            current_lr = self.optimizer.param_groups[0]["lr"]

            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "epoch_time_sec": epoch_time,
                "step": self.global_step,
                "current_lr": current_lr
            }
            self.logger.log(log_data)

            #  Save last checkpoint using YOUR function
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.global_step,
                os.path.join(self.run_dir, "checkpoints", "last.pt"),
                epoch
            )

            # Save best model
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                epochs_without_improvement = 0

                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.global_step,
                    os.path.join(self.run_dir, "checkpoints", "best.pt"),
                    epoch
                )
                print("Saved best model!!")
                print(f"✓ New best model! (improved by {improvement:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")


            # Early stopping
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break