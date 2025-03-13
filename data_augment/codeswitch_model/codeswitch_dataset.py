import datasets                             # type: ignore    
import random

from torch.utils.data import Dataset        # type: ignore
from transformers import MT5Tokenizer       # type: ignore

import pandas as pd                         # type: ignore
from indic_transliteration import sanscript # type: ignore

def romanize_hindi():
    '''
    Romanize Hindi text in a TSV file
    '''
    df = pd.read_csv("../dataset/CSPref/train-00000-of-00001.tsv", sep="\t", on_bad_lines='skip')
    column_name = df.columns[1]  # original_l2 - hindi script text

    df_romanized = pd.DataFrame({
        "romanized_text": df[column_name].astype(str).apply(
            lambda x: sanscript.transliterate(x, sanscript.DEVANAGARI, sanscript.ITRANS) if isinstance(x, str) else x
        )
    })

    df_romanized.to_csv("../dataset/CSPref/romanized_cspref_train.tsv", sep="\t", index=False)


class CodeswitchDataset(Dataset):
  def __init__(self, tokenizer, block_size):
    self.data = []
    self.MASK_CHAR = "\u2047" # the doublequestionmark character, for mask
    self.PAD_CHAR = "\u25A1" # the empty square character, for pad
    self.tokenizer = tokenizer
    self.max_length = block_size

    # hugging face dataset 
    self.data = datasets.load_dataset("garrykuwanto/cspref")['train']
    self.data = [sanscript.transliterate(datapoint['original_l2'], sanscript.DEVANAGARI, sanscript.ITRANS) for datapoint in self.data]
    
    # add masking to the dataset -- partially taken from A4
    chars = set()
    for sentence in self.data:
        for char in sentence:
            chars.add(char)
    
    chars = list(chars)
    chars.sort()

    assert self.MASK_CHAR not in chars
    assert self.PAD_CHAR not in chars
    chars.insert(0, self.MASK_CHAR)
    chars.insert(0, self.PAD_CHAR)

    self.stoi = {ch:i for i,ch in enumerate(chars)}
    self.itos = {i:ch for i,ch in enumerate(chars)}

    data_size, vocab_size = len(self.data), len(chars)
    self.block_size = block_size
    self.vocab_size = vocab_size


  def __len__(self):
    return len(self.data)
  

  '''adapted from A4 code'''
  def __getitem__(self, idx):
        # retrieve the element of self.data at the given index
        doc = self.data[idx]

        # randomly truncate the document
        truncated_len = random.randint(4, self.block_size*7 // 8)
        truncated_doc = doc[:truncated_len]

        # break the truncated document into three substrings
        lower_bound = 1
        upper_bound = truncated_len // 2 - 1

        masked_len = random.randint(lower_bound, upper_bound)
        masked_start = (truncated_len - masked_len) // 2 - 1
        prefix = truncated_doc[:masked_start]
        masked_content = truncated_doc[masked_start : masked_start + masked_len]
        suffix = truncated_doc[masked_start + masked_len:]

        # rearrange these substrings into the masked string
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content
        masked_string += self.PAD_CHAR * (self.block_size - len(masked_string) + 1)

        # use masked_string to construct the input and output example pair
        x = masked_string[:-1]
        y = masked_string[1:]

        # encode the resulting input and output strings as Long tensors
        #x = torch.tensor([self.stoi[ch] for ch in x], dtype=torch.long)
        #y = torch.tensor([self.stoi[ch] for ch in y], dtype=torch.long)

        encoding = self.tokenizer(
          x, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        labels = self.tokenizer(
          y, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
           "input_ids": encoding["input_ids"].squeeze(),
           "attention_mask": encoding["attention_mask"].squeeze(),
           "labels": labels["input_ids"].squeeze() }            
  

if __name__ == '__main__':
  #tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
  #dataset = CodeswitchDataset(tokenizer, 128)
  #print(dataset[0])

  romanize_hindi()
  

