import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import yaml
from pathlib import Path

class TextGenerationDataset(Dataset):
    def __init__(self, split, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        dataset_name=self.config['data']['dataset_name']
        dataset_config=self.config['data']['dataset_config']
        
        self.seq_length=self.config['data']['seq_length']
        
        # -------------------------
        # Step 1: Load raw dataset
        # -------------------------
        dataset_path=Path('data/raw')
        
        # Check if folder does not exist OR is empty
        if not dataset_path.exists() or not any(dataset_path.iterdir()):
            print("Downloading dataset")
            dataset=load_dataset(dataset_name, dataset_config)
            dataset.save_to_disk(dataset_path)
        else:
            print("Loading dataset from disk")
            dataset=load_from_disk(dataset_path)

        # splitting the dataset as needed
        dataset_split=dataset[split]

        # -----------------------------
        # Step 2: Load or create tokens
        # -----------------------------
        
        processed_path=Path('data/processed')
        processed_file=Path(f"data/processed/{split}_tokens.pt")
        
        # Check if folder does not exist OR is empty
        if not processed_path.exists() or not any(processed_path.iterdir()):
            # initilizing the tokenizer
            print("Starting tokenization")
            self.tokenizer=AutoTokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token=self.tokenizer.eos_token # setting padding token as End of Sentence token(already known token)

            all_input_ids = []

            # tokenizing document by document
            for text in dataset_split["text"]:
                tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=False
                )["input_ids"][0]
        
                all_input_ids.append(tokens)

            self.input_ids = torch.cat(all_input_ids, dim=0)
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.input_ids, processed_file)
        else:
            # loading tokenized data
            print("Loading tokenized data from ", processed_file)
            self.input_ids=torch.load(processed_file)

        # 3. Calculating sequences
        self.num_sequences=(len(self.input_ids) - 1) // self.seq_length

    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, index):
        start=index * self.seq_length
        end=start + self.seq_length

        x = self.input_ids[start:end]
        y = self.input_ids[start + 1:end + 1]

        return x,y