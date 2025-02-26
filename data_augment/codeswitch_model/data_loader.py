import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MT5Tokenizer

class ParsedDataset(Dataset):
  def __init__(self, file_path, tokenizer, max_length=128):
    self.data = []
    self.tokenizer = tokenizer
    self.max_length = max_length

    with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                
                sentence, pos_tags, dep_rels = parts
                input_text = 'Translate to code-switching language (Spanglish or Chinglish): ' + sentence + ' <POS> ' + pos_tags + ' <DEP> ' + dep_rels

                self.data.append(input_text)
  

  def __len__(self):
    return len(self.data)


  def __getitem__(self, idx):
    input_text = self.data[idx]
    encoding = self.tokenizer(
       input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    
    return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()               # auto-regressive self-supervised learning
        }
  

def get_dataloader(filepath, batch_size=16):
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    dataset = ParsedDataset(filepath, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# sanity check
if __name__ == '__main__':
  filepath = "../outputs/xnli_annotated_dev.txt"  
  dataloader = get_dataloader(filepath)

  # check first batch
  for batch in dataloader:
      print(batch)
      break