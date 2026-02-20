import torch
import torch.nn as nn
import math

# Perplexity calculation
def calc_perplexity(loss: float)->float:
    """
    Perplexity (PPL) is a primary evaluation metric that measures how well a probability model predicts a sample, 
    with lower scores indicating better performance and higher confidence.
    Compute perplexity from average cross-entropy loss. 
    
    PPL = exp(loss)
    """
    return math.exp(loss)


class Evaluator():
    def __init__(self, model, config, device):
        self.model = model
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

    def evaluate(self, dataloader):
        """
        Run validation loop.
        Returns:
            avg_loss (float)
            perplexity (float)
        """
        # setting model to evaluation mode
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        # disabling gradient tracking on evaluation
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
        
                input_ids = batch["input_ids"]

                # Forward pass: compute model outputs (logits)
                logits=self.model(input_ids)

                # Shift logits and targets for next-token prediction (input t predicts token t+1)
                logits, targets = self.shift_logits_targets(logits, input_ids)

                # Reshape logits for cross entropy:
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
                
                # Multiply by number of tokens
                # This avoids batch-size bias
                total_loss += loss.item() * targets.numel()

                # Count total tokens
                total_tokens += targets.numel()

        # Compute average loss per token
        avg_loss = total_loss / total_tokens

        # Compute perplexity
        perplexity = calc_perplexity(avg_loss)

        return avg_loss, perplexity