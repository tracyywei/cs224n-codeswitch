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

    tconf = TrainerConfig(
        max_epochs = 5,          # goal range is 5-10 epochs
        batch_size = 16,         # goal range is 8-32
        learning_rate = 2e-4,    
        betas = (0.9, 0.999), 
        weight_decay = 0.01,     # avoid overregularization
        lr_decay = True,
        warmup_tokens = 1e6,
        final_tokens = 10e9,
        ckpt_path = "./checkpoints/mT5_finetuned.pth",
        num_workers = 0, 
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
    # TODO: figure out which argparse params are needed and consider hardcoding the rest
    argp = argparse.ArgumentParser()
    argp.add_argument("--train_file", type=str, required=True)
    argp.add_argument("--dev_file", type=str, required=True)
    argp.add_argument("--output_model", type=str, required=True)
    argp.add_argument("--max_length", type=int, default=128)
    argp.add_argument("--max_epochs", type=int, default=10)
    argp.add_argument("--batch_size", type=int, default=32)
    argp.add_argument("--learning_rate", type=float, default=3e-5)
    args = argp.parse_args()
    
    writer = SummaryWriter(log_dir='expt/%s/%s_%s_pt_lr_%f_ft_lr_%f' % (
      args.train_file,
      args.dev_file,
      args.output_model,
      args.batch_size,
      args.learning_rate))
      
    model = torch.load('mt5_intermediate_finetuned.pth')

    # TODO: fix trainer config to include the proper params 
    tconf = TrainerConfig(
        max_epochs=10,
        batch_size=64,
        learning_rate=3e-5,
        
        lr_decay=True,
        num_workers=0, 
        writer=writer
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
    finetune_mT5_codeswitched()
