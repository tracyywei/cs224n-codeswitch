import argparse
import torch # type: ignore

from transformers import MT5ForConditionalGeneration, MT5Tokenizer # type: ignore
from trainer import Trainer, TrainerConfig
from parsed_dataset import ParsedDataset

from codeswitch_dataset import CodeswitchDataset
from torch.utils.tensorboard import SummaryWriter               # type: ignore

def finetune_mT5_codeswitched():
    '''
    STEP 1: Finetune on codeswitched data
    '''
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    dataset = CodeswitchDataset(tokenizer=tokenizer, block_size=128)
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

    device = torch.device("cuda")
    print(f"Device name: {torch.cuda.get_device_name(device)}")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    print(f"Total memory: {total_memory / (1024**3):.2f} GB")
    allocated_memory = torch.cuda.memory_allocated(device)
    print(f"Allocated memory: {allocated_memory / (1024**3):.2f} GB")
    free_memory = total_memory - allocated_memory
    print(f"Free memory: {free_memory / (1024**3):.2f} GB")

    tconf = TrainerConfig(
        max_epochs = 10,         # goal range is 5-10 epochs
        batch_size = 8,          # goal range is 8-32
        learning_rate = 2e-4,    
        betas = (0.9, 0.999), 
        weight_decay = 0.01,     # avoid overregularization
        lr_decay = True,
        warmup_tokens = 1e6,
        final_tokens = 10e9,
        num_workers = 4, 
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        dev_dataset=None,
        config=tconf
    )

    trainer.train()
    torch.save(model.state_dict(), 'mt5_intermediate_finetuned.pth')
    

def finetune_mT5_codeswitched_generation(dataset):
    '''
    STEP 2: Finetune for codeswitch generation on the parsed dataset
    '''
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    # model.load_state_dict(torch.load('mt5_intermediate_finetuned.pth'))   # uncomment for gpu
    model.load_state_dict(torch.load('mt5_intermediate_finetuned.pth', map_location=torch.device('cpu')))   # uncomment for cpu
  
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

    dataset = ParsedDataset(dataset, tokenizer=tokenizer)

    tconf = TrainerConfig(
        max_epochs=8,       # goal range is 5-10 epochs
        batch_size=16,      # reduce from 16 to 8 if OOM occurs 
        learning_rate=3e-4,
        lr_decay=True,
        betas = (0.9, 0.98),           
        weight_decay = 0.01,   # Regularization to prevent overfitting
        num_workers=0,      # change to 2 when running on gpu (0 for cpu)
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        dev_dataset=None,
        config=tconf
    )

    trainer.train()
    torch.save(model.state_dict(), 'mt5_finetuned.pth')


if __name__ == '__main__':
    # run step 1
    # finetune_mT5_codeswitched()

    # run step 2
    dataset = "../outputs/xnli_annotated_dev.txt"
    finetune_mT5_codeswitched_generation(dataset)
