import logging
import torch
torch.cuda.empty_cache()
import random
import pprint
#2
import model.XNLI.base
import util.tool
import os

import torch.nn as nn
import numpy as np

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

from torch.nn import functional as F

class BERTTool(object):
    def init(args):
        BERTTool.multi_bert = BertModel.from_pretrained(args.multi_bert.location)
        BERTTool.multi_tokener = BertTokenizer.from_pretrained(args.multi_bert.location)
        BERTTool.multi_pad = BERTTool.multi_tokener.convert_tokens_to_ids(["[PAD]"])[0]
        BERTTool.multi_sep = BERTTool.multi_tokener.convert_tokens_to_ids(["[SEP]"])[0]
        BERTTool.multi_cls = BERTTool.multi_tokener.convert_tokens_to_ids(["[CLS]"])[0]
        #BERTTool.multi_bert.eval()
        #BERTTool.en_bert.eval()


class Model(model.XNLI.base.Model):
    def __init__(self, args, DatasetTool, inputs):
        np.random.seed(args.train.seed)
        torch.manual_seed(args.train.seed)
        random.seed(args.train.seed)
        super().__init__(args, DatasetTool, inputs)
        BERTTool.init(self.args)
        _, _, _, _, worddict, _ = inputs
        self.worddict = worddict
        print(self.worddict)
        self.bert = BERTTool.multi_bert
        self.tokener = BERTTool.multi_tokener
        self.pad = BERTTool.multi_pad
        self.sep = BERTTool.multi_sep
        self.cls = BERTTool.multi_cls
        self.classifier = nn.Linear(768, 3)  # 3 classes: Entailment, Neutral, Contradiction

    def set_optimizer(self):
        all_params = set(self.parameters())
        if self.args.train.bert == False:
            bert_params = set(BERTTool.multi_bert.parameters())
            for para in bert_params:
                para.requires_grad=False
            params = [{"params": list(all_params - bert_params), "lr": self.args.lr.default}]
        else:
            bert_params = set(BERTTool.multi_bert.parameters())
            params = [{"params": list(all_params - bert_params), "lr": self.args.lr.default},
                      {"params": list(bert_params), "lr": self.args.lr.bert}
                      ]
        self.optimizer = AdamW(params)

    def save_model(self, epoch):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(BASE_DIR, f"./saved/model_epoch_{epoch}.pt")
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, save_path)
        
        logging.info(f"Model saved at {save_path}")

    def run_eval(self, train, dev, test):
        logging.info("Starting evaluation")
        self.eval()
        summary = {}

        datasets = {"train": train, "dev": dev}
        datasets.update(test)

        for set_name, dataset in datasets.items():
            logging.info(f"Evaluating on {set_name} dataset...")
            _, predictions = self.run_test(dataset)
            eval_results = self.DatasetTool.evaluate(predictions, dataset, self.args)
            summary.update({f"eval_{set_name}_{k}": v for k, v in eval_results.items()})

        logging.info("Evaluation Results:")
        logging.info(pprint.pformat(summary))

        return summary


    def run_train(self, train, dev, test):
        self.set_optimizer()
        iteration = 0
        best = {}
        gradient_accumulation_steps = 2  # Accumulate gradients over multiple steps
        effective_batch_size = self.args.train.batch
        logging.info(f"Starting training with batch size {effective_batch_size}")

        for epoch in range(self.args.train.epoch):
            self.train()
            logging.info(f"Starting training epoch {epoch}")
            summary = self.get_summary(epoch, iteration)
            total_loss = 0.0
            batch_count = 0
            scaler = torch.cuda.amp.GradScaler() 

            for i in range(0, len(train), effective_batch_size):
                batch = train[i : i + effective_batch_size]

                try:
                    with torch.cuda.amp.autocast():
                        loss, _ = self.forward(batch)

                    if torch.isnan(loss) or loss.item() == 0.0:
                        logging.warning(f"Skipping batch {i} due to NaN or zero loss.")
                        self.optimizer.zero_grad()
                        continue

                    scaler.scale(loss).backward()
                    total_loss += loss.item()
                    batch_count += 1

                    if batch_count % gradient_accumulation_steps == 0 or batch_count == len(train):
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()

                    if batch_count % 10 == 0:
                        logging.info(f"Epoch {epoch}, Batch {i}/{len(train)}, Loss: {loss.item():.4f}")

                except torch.cuda.OutOfMemoryError:
                    logging.error("CUDA OOM: Reducing batch size...")
                    torch.cuda.empty_cache()
                    effective_batch_size = max(1, effective_batch_size // 2)
                    continue


            iteration += batch_count
            summary.update({"loss": total_loss / batch_count})
            datasets = {"train": train, "dev": dev, "test": test}

            if not self.args.train.not_eval:
                for set_name, dataset in datasets.items():
                    tmp_summary, pred = self.run_test(dataset)
                    self.DatasetTool.record(pred, dataset, set_name, self.args)
                    summary.update({f"eval_{set_name}_{k}": v for k, v in tmp_summary.items()})

            best = self.update_best(best, summary, epoch)
            logging.info(pprint.pformat(best))
            logging.info(pprint.pformat(summary))

            self.save_model(epoch)  # Save model at each epoch

    def get_info(self, batch):
        token_ids = []
        token_loc = []
        
        for x in batch:
            premise = x["premise"]
            hypothesis = x["hypothesis"]
            
            per_token_ids = [self.cls]  # CLS token at the beginning
            per_token_loc = []
            cur_idx = 1 

            for token in premise.split():
                tmp_ids = self.tokener.encode(token, add_special_tokens=False)
                per_token_ids += tmp_ids
                per_token_loc.append(cur_idx)
                cur_idx += len(tmp_ids)

            per_token_ids += [self.sep]
            cur_idx += 1

            for token in hypothesis.split():
                tmp_ids = self.tokener.encode(token, add_special_tokens=False)
                per_token_ids += tmp_ids
                per_token_loc.append(cur_idx)
                cur_idx += len(tmp_ids)

            per_token_ids += [self.sep]
            token_ids.append(per_token_ids)
            token_loc.append(per_token_loc)

        max_len = max(len(p) for p in token_ids)
        mask_ids = []
        type_ids = []

        for per_token_ids in token_ids:
            per_mask_ids = [1] * len(per_token_ids) + [0] * (max_len - len(per_token_ids))
            per_token_ids += [self.pad] * (max_len - len(per_token_ids))
            per_type_ids = [0] * max_len

            mask_ids.append(per_mask_ids)
            type_ids.append(per_type_ids)

        token_ids = torch.Tensor(token_ids).long().to(self.device)
        mask_ids = torch.Tensor(mask_ids).long().to(self.device)
        type_ids = torch.Tensor(type_ids).long().to(self.device)

        return token_loc, token_ids, type_ids, mask_ids

    def forward(self, batch):
        token_loc, input_ids, type_ids, attention_mask = self.get_info(batch)

        outputs = self.bert(input_ids, token_type_ids=type_ids, attention_mask=attention_mask)
        pooled_output = outputs[1] if isinstance(outputs, tuple) else outputs.pooler_output

        logits = self.classifier(pooled_output)

        loss = torch.tensor(0.0)
        if self.training:
            labels = torch.tensor([x["label"] for x in batch], dtype=torch.long).to(self.device)
            loss = F.cross_entropy(logits, labels)

        predictions = torch.argmax(logits, dim=1).tolist()
        return loss, predictions

    def start(self, inputs):
        train, dev, test, _, _, _ = inputs
        if self.args.model.resume is not None:
            self.load(self.args.model.resume)
        if not self.args.model.test:
            self.run_train(train, dev, test)
        if self.args.model.resume is not None:
            self.run_eval(train, dev, test)