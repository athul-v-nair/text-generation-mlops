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

    def generate(self, prompt, temperature = 1.0, top_k = 50, max_new_tokens=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Check if we exceeded the model's max length
                if generated_ids.size(1) > self.config['data']['seq_length']:
                    # Take only the LAST seq_length tokens
                    # [:, -128:] means "all batches, last 128 tokens"
                    input_chunk = generated_ids[:, -self.config['data']['seq_length']: ]
                else:
                    input_chunk = generated_ids

                logits = self.model(input_chunk)
                next_token_logits = logits[:, -1, :]

                # Applying Temperature: Sharpen distribution
                # Including temperature only if value is > 1
                if temperature > 1:
                    next_token_logits = next_token_logits / temperature 

                # Applying top-k filtering: Pick top k-highest logits
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k)
                    filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                    
                    filtered_logits.scatter_(1, top_k_indices, top_k_logits)
                    # eg: [2.625, 5.625, 1.5, ..., 0.2, -inf, -inf, ..., -inf]
                    #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^
                    #     40 highest values                 rest are -inf

                    next_token_logits = filtered_logits

                probability = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probability, num_samples=1)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
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

    # Testing a more creative model
    prompt = "The history of India"
    temperature = 1
    top_k = 40

    result = generator.generate(prompt, temperature, top_k)
    print(result)