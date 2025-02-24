import argparse
import logging
import torch
import dataset
from transformers import BertTokenizerFast
from torch.utils.tensorboard import SummaryWriter
from model import BertForParsing
from trainer import Trainer, TrainerConfig

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--train_file", type=str, required=True)
    argp.add_argument("--dev_file", type=str, required=True)
    argp.add_argument("--output_model", type=str, required=True)
    argp.add_argument("--max_length", type=int, default=128)
    argp.add_argument("--max_epochs", type=int, default=10)
    argp.add_argument("--batch_size", type=int, default=32)
    argp.add_argument("--learning_rate", type=float, default=3e-5)
    args = argp.parse_args()

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    elif torch.backends.mps.is_available():
        device = 'mps'

    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/%s/%s_%s_pt_lr_%f_ft_lr_%f' % (
        args.train_file,
        args.dev_file,
        args.output_model,
        args.batch_size,
        args.learning_rate))
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    
    train_dataset = dataset.UDParsingDataset(args.train_file, tokenizer, max_length=args.max_length)
    dev_dataset = dataset.UDParsingDataset(args.dev_file, tokenizer, 
                                   pos_label2id=train_dataset.pos_label2id, 
                                   dep_label2id=train_dataset.dep_label2id, 
                                   max_length=args.max_length)
    
    num_pos_labels = len(train_dataset.pos_label2id)
    num_dep_labels = len(train_dataset.dep_label2id)
    
    print("Number of POS labels:", num_pos_labels)
    print("Number of dependency labels:", num_dep_labels)

    model = BertForParsing(num_pos_labels, num_dep_labels, max_length=128, pretrained_model_name="bert-base-multilingual-cased")
    
    trainer_config = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                                   lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(train_dataset)*128,
                                   num_workers=4, writer=writer)
    trainer = Trainer(model, train_dataset, dev_dataset, trainer_config)
    
    trainer.train()
    
    # saving model state dict and label mapping
    model_and_labels = {
        "model_state_dict": model.state_dict(),
        "pos_label2id": train_dataset.pos_label2id,
        "dep_label2id": train_dataset.dep_label2id,
        "max_length": args.max_length,
    }
    torch.save(model_and_labels, args.output_model)

    print("Model saved to", args.output_model)
    
if __name__ == "__main__":
    main()
