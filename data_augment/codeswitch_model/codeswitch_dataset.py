import datasets
import random
import torch                                # type: ignore

from torch.utils.data import Dataset        # type: ignore

class CodeswitchDataset(Dataset):
  def __init__(self, block_size):
    self.data = []

    self.data = datasets.load_dataset("CAiRE/ASCEND")['train']
    self.data = [datapoint['transcription'] for datapoint in self.data]
    
    # add masking to the dataset -- partially taken from A4
    chars = set()
    for sentence in self.data:
        for char in sentence:
            chars.add(char)
    
    self.chars = list(chars)
    self.chars.sort()

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
        x = torch.tensor([self.stoi[ch] for ch in x], dtype=torch.long)
        y = torch.tensor([self.stoi[ch] for ch in y], dtype=torch.long)
        return x, y
  