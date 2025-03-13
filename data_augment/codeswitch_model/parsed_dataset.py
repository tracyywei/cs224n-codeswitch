import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from transformers import MT5Tokenizer # type: ignore

class ParsedDataset(Dataset):
  def __init__(self, file_path, label_file_path, tokenizer, validation=False, max_length=128):
    self.data = []
    self.tokenizer = tokenizer
    self.max_length = max_length

    # experiment 4
    filename = "dataset/hinglish_top_dataset/validation.tsv" if validation else "dataset/hinglish_top_dataset/train.tsv"
    with open(filename) as f:
      self.data = f.readlines()[1:]
    
    self.data = [line.strip().split("\t")[:2] for line in self.data]

    # experiments 1-3
    # with open(file_path, "r", encoding="utf-8") as file:
    #   with open(label_file_path, "r", encoding="utf-8") as label_file:
    #         for line in file:
    #             label = label_file.readline().strip().split("\t")[1]
    #             parts = line.strip().split("\t")
                
    #             sentence, pos_tags, dep_rels = parts
    #             input_text = sentence + ' <POS> ' + pos_tags + ' <DEP> ' + dep_rels

    #             self.data.append([input_text, label])
  

  def __len__(self):
    return len(self.data)


  def __getitem__(self, idx):
    input_text, label = self.data[idx]
    encoding = self.tokenizer(
       input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
    )

    label_encoding = self.tokenizer(
        label, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    
    return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label_encoding["input_ids"].squeeze()       
        }


def get_dataloader(filepath, label_file_path, batch_size=16):
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    dataset = ParsedDataset(filepath, label_file_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# sanity check
if __name__ == '__main__':
  filepath = "../dataset/annotated/annotated_hinglish_en.txt"  
  label_file_path = "../dataset/enghinglish/dev.txt"
  tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
  dataloader = get_dataloader(filepath, label_file_path)

  # check first batch
  for batch in dataloader:
      print(batch)