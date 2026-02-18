import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, optimizer, scheduler, config, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def shift_logits_targets(self, logits, targets):
        """
        Align predictions with next-token targets.

        logits:  (B, T, V)
        targets: (B, T)

        After shift:
            logits -> (B, T-1, V)
            targets -> (B, T-1)
        """
        return logits[:,:-1,:], targets[:, 1:]
    
    def train_step(self, batch):
        # Set model to training mode (enables dropout, etc.)
        self.model.train()
        
        batch=batch.to(self.device)
        
        # Forward pass: compute model outputs (logits)
        logits=self.model(batch)

        # Shift logits and targets for next-token prediction (input t predicts token t+1)
        logits, targets = self.shift_logits_targets(logits, batch)

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

        return loss.item()       