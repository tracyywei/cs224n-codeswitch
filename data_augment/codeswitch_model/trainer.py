"""
trainer.py file adapted from A4
"""

import math
import logging

from tqdm import tqdm # type: ignore
import numpy as np # type: ignore

import torch           # type: ignore
import torch.optim as optim      # type: ignore
from torch.optim.lr_scheduler import LambdaLR   # type: ignore
from torch.utils.data.dataloader import DataLoader  # type: ignore

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    writer = None
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, dev_dataset, config, stop_early=False):
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.config = config
        self.stop_early = stop_early

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            self.model = self.model.to(self.device)

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config

        # Create optimizer groups with weight decay handling.
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
        step = 0
        self.tokens = 0  # counter for LR decay

        def run_epoch(split):
            nonlocal step
            is_train = (split == 'train')
            model.train(is_train)
            data = self.train_dataset if is_train else self.dev_dataset
            loader = DataLoader(data, batch_size=config.batch_size, shuffle=is_train, num_workers=config.num_workers)
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:
                # Move all batch tensors to the appropriate device.
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                with torch.set_grad_enabled(is_train):
                    output = model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = output.loss
                    logits = output.logits

                    loss = loss.mean()  # if model is wrapped in DataParallel
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        # Update token counter and adjust learning rate.
                        self.tokens += (attention_mask >= 0).sum().item()  # counting tokens processed
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, lr {lr:e}")

                    if config.writer is not None:
                        config.writer.add_scalar('train/loss', loss.item(), step)
                        config.writer.add_scalar('train/lr', lr, step)
                    step += 1
                    
                    if self.stop_early and loss.item() < 0.1:
                        break

            if not is_train:
                logger.info("Dev loss: %f", np.mean(losses))
                print("Dev loss: %f" % np.mean(losses))
            else:
                return loss.item()

        for epoch in range(config.max_epochs):
            loss = run_epoch('train')
            if self.stop_early and loss < 0.1:
                break

            if self.dev_dataset is not None:
                run_epoch('dev')
            self.save_checkpoint()

    def evaluate(self):
        model = self.model
        model.eval()
        dev_loader = DataLoader(self.dev_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                _, _, loss = model(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=labels,)
                total_loss += loss.item()
        avg_loss = total_loss / len(dev_loader)
        logger.info(f"Dev Loss: {avg_loss:.4f}")
        print(f"Dev Loss: {avg_loss:.4f}")
        model.train() 