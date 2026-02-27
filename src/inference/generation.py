import torch
import yaml, datetime
from transformers import AutoTokenizer

from src.model.decoder_transformer import DecoderTransformer

from src.constants import CONFIG_PATH
from src.constants import CURRENT_RUN

class TextGeneration:
    def __init__(self, checkpoint_path, config_path, device='cpu'):
        self.device = device

        with open(config_path,'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer)

        self.model = DecoderTransformer(
            vocabulary_size=self.vocab_size,
            max_seq_len=self.config['data']['seq_length'],
            dim_model=self.config['training']['dim_model'],
            num_layers=self.config['training']['num_layers'],
            num_heads=self.config['training']['num_heads'],
            dim_ff=self.config['training']['dim_ff'],
            dropout=self.config['training']['dropout']
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"✓ Trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"✓ Device: {self.device}\n")

    # ==================== SAMPLING STRATEGIES ====================
    def _greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Strategy 1: Greedy Decoding
        Simply pick the token with highest probability.
        
        Args:
            logits: (1, vocab_size)
        
        Returns:
            next_token: (1,)
        """
        # argmax returns the INDEX of the maximum value
        # Example: logits = [0.1, 0.7, 0.2] → argmax = 1
        next_token = torch.argmax(logits, dim=-10)
        return next_token
    
    def _temperature_sample(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Strategy 2: Temperature Sampling
        
        Lower temperature → more focused (conservative)
        Higher temperature → more random (creative)
        
        Args:
            logits: (1, vocab_size)
            temperature: float (typically 0.7 to 1.5)
        
        Returns:
            next_token: (1,)
        """
        # Divide logits by temperature BEFORE softmax
        # This changes the "sharpness" of the distribution
        logits = logits / temperature

        # Convert to probabilities
        probabilities = torch.nn.functional.softmax(logits)

        # Sample from the probability distribution
        # multinomial picks one token based on probabilities
        next_token = torch.multinomial(probabilities, num_samples=1)
        return next_token.squeeze()
    
    def _top_k_sample(self, logits: torch.Tensor, top_k: int, temperature: float) -> torch.Tensor:
        """
        Strategy 3: Top-K Sampling
        
        Keep only the top K most probable tokens, set rest to -inf
        
        Args:
            logits: (1, vocab_size)
            top_k: Number of top tokens to keep (e.g., 50)
            temperature: Sampling temperature
        
        Returns:
            next_token: (1,)
        """
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)

        # Create a tensor of -inf with same shape as logits
        filtered_logits = torch.full_like(logits, float('-inf'))

        # Put the top K values back into the filtered tensor
        # eg: [2.625, 5.625, 1.5, ..., 0.2, -inf, -inf, ..., -inf]
        #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^
        #     40 highest values                 rest are -inf
        filtered_logits.scatter_(1, top_k_indices, top_k_logits)

        filtered_logits = filtered_logits / temperature
        probabilities = torch.nn.functional.softmax(logits)
        next_token = torch.multinomial(probabilities, num_samples=1)

        return next_token.squeeze()
    
    def _top_p_sample(self, logits: torch.Tensor, top_p: float, temperature: float) -> torch.Tensor:
        """
        Strategy 4: Top-P (Nucleus) Sampling ⭐
        
        Keep tokens until cumulative probability >= top_p
        This is ADAPTIVE - uses fewer tokens when model is confident,
        more tokens when model is uncertain.
        
        Args:
            logits: (1, vocab_size)
            top_p: Cumulative probability threshold (e.g., 0.9)
            temperature: Sampling temperature
        
        Returns:
            next_token: (1,)
        """
        # Apply temperature first
        logits = logits / temperature
        
        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        # cumsum([0.5, 0.3, 0.15, 0.05]) = [0.5, 0.8, 0.95, 1.0]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find where cumulative probability exceeds top_p
        # This creates a boolean mask
        # Example: if top_p=0.9, keep [0.5, 0.8] but not [0.95, 1.0]
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift right to keep the first token above threshold
        # [False, False, True, True] → [False, False, False, True]
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Create filtered probabilities
        filtered_probs = sorted_probs.clone()
        filtered_probs[sorted_indices_to_remove] = 0.0
        
        # Renormalize (make probabilities sum to 1 again)
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        # Map filtered probabilities back to original order
        original_probs = torch.zeros_like(probs)
        original_probs.scatter_(1, sorted_indices, filtered_probs)
        
        # Sample from the filtered distribution
        next_token = torch.multinomial(original_probs, num_samples=1)
        
        return next_token.squeeze()
    
    def _combined_sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """
        Strategy 5: Combined (Temperature + Top-P) 🏆
        
        1. Apply temperature to control randomness
        2. Apply top-p to filter unlikely tokens
        3. Sample from the result
        
        Args:
            logits: (1, vocab_size)
            temperature: Sampling temperature
            top_p: Nucleus threshold
        
        Returns:
            next_token: (1,)
        """
        # Just call top_p_sample, which already applies temperature internally
        return self._top_p_sample(logits, top_p, temperature)
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, generated_tokens: set, penalty: float) -> torch.Tensor:
        """
        Reduce probability of tokens that have already been generated.
        Helps prevent repetitive loops like "the the the..."
        
        Args:
            logits: (1, vocab_size)
            generated_tokens: Set of token IDs already generated
            penalty: Penalty factor (> 1.0 means penalize, < 1.0 means encourage)
        
        Returns:
            Modified logits
        """
        for token_id in generated_tokens:
            # Divide logits by penalty if token was already used
            # If penalty=1.2 and logit=2.0, new logit = 2.0/1.2 = 1.67
            logits[0, token_id] /= penalty
        
        return logits
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def compare_strategies(self, prompt: str, max_length: int = 30):
        """
        Helpful function to compare all strategies side-by-side.
        Great for educational purposes and debugging!
        
        Args:
            prompt: Input text
            max_length: Tokens to generate
        
        Returns:
            Dictionary with results from each strategy
        """
        results = {}
        
        print(f"\n{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*70}\n")
        
        strategies = [
            ("greedy", {}),
            ("temperature", {"temperature": 0.7}),
            ("temperature", {"temperature": 1.2}),
            ("top_k", {"top_k": 50, "temperature": 0.8}),
            ("top_p", {"top_p": 0.9, "temperature": 0.8}),
            ("combined", {"temperature": 0.8, "top_p": 0.9}),
        ]
        
        for strategy, params in strategies:
            label = f"{strategy}"
            if params:
                label += f" ({', '.join(f'{k}={v}' for k, v in params.items())})"
            
            output = self.generate(
                prompt=prompt,
                max_length=max_length,
                strategy=strategy,
                **params
            )
            
            results[label] = output
            print(f"{label}:")
            print(f"  {output}\n")
        
        return results

    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 50, strategy: str = "combined", temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.2,**kwargs) -> str:
        """
        Main generation function.
        
        Args:
            prompt: Input text to continue
            max_length: Maximum tokens to generate
            strategy: 'greedy', 'temperature', 'top_k', 'top_p', or 'combined'
            temperature: Sampling temperature (0.1 to 2.0)
            top_k: Number of top tokens to keep
            top_p: Cumulative probability threshold (0.0 to 1.0)
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
        
        Returns:
            Generated text as string
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Track which tokens we've generated (for repetition penalty)
        generated_tokens = set(input_ids[0].tolist())

        for _ in range(max_length):
            # Get model predictions for next token
            # input_ids shape: (1, current_length)
            # logits shape: (1, current_length, vocab_size)
            logits = self.model(input_ids)

            next_token_logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, 
                    generated_tokens, 
                    repetition_penalty
                )

            # Next token based on strategy
            if strategy == "greedy":
                next_token = self._greedy_sample(next_token_logits)
            
            elif strategy == "temperature":
                next_token = self._temperature_sample(next_token_logits, temperature)
            
            elif strategy == "top_k":
                next_token = self._top_k_sample(next_token_logits, top_k, temperature)
            
            elif strategy == "top_p":
                next_token = self._top_p_sample(next_token_logits, top_p, temperature)
            
            elif strategy == "combined":
                # This is the BEST strategy
                next_token = self._combined_sample(
                    next_token_logits, 
                    temperature, 
                    top_p
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Add the new token to our sequence
            # input_ids shape: (1, current_length) → (1, current_length + 1)
            next_token = next_token.unsqueeze(0).unsqueeze(1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Track this token for repetition penalty
            generated_tokens.add(next_token.item())
            
            # Step 6: Stop if we generate the end-of-sequence token
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Step 7: Decode token IDs back to text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
            
if __name__ == "__main__":
    checkpoint_path = CURRENT_RUN + 'checkpoints/best.pt'

    print("="*40)
    print("TEXT GENERATION DEMO")
    print("="*40)

    generator = TextGeneration(
        checkpoint_path=checkpoint_path,
        config_path=CONFIG_PATH,
        device='cpu'
    )

    prompt = "The history of Football"

    # Generating using combined generation strategy(top-p & temperature)
    result = generator.generate(prompt, max_length=100, strategy="combined")
    print(result)