import logging
import torch
import random
import pprint
#2
import model.XNLI.base
import util.tool
import os

import torch.nn as nn
import numpy as np

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

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
        ds = {"train": train, "dev": dev}
        ds.update(test)
        for set_name, dataset in ds.items():
            tmp_summary, pred = self.run_test(dataset)
            self.DatasetTool.record(pred, dataset, set_name, self.args)
            summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
        logging.info(pprint.pformat(summary))

    def run_train(self, train, dev, test):
        self.set_optimizer()
        iteration = 0
        best = {}
        for epoch in range(self.args.train.epoch):
            self.train()
            logging.info("Starting training epoch {}".format(epoch))
            summary = self.get_summary(epoch, iteration)
            loss, iter = self.run_batches(train, epoch)
            iteration += iter
            summary.update({"loss": loss})
            ds = {"train": train, "dev": dev}
            ds.update(test)
            if not self.args.train.not_eval:
                for set_name, dataset in ds.items():
                    tmp_summary, pred = self.run_test(dataset)
                    self.DatasetTool.record(pred, dataset, set_name, self.args)
                    summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
            best = self.update_best(best, summary, epoch)
            logging.info(pprint.pformat(best))
            logging.info(pprint.pformat(summary))

            self.save_model(epoch)

    def cross(self, x, disable=False):
        if not disable and self.training and (self.args.train.cross >= random.random()):
            lan = random.randint(0,len(self.args.dict_list) - 1)
            if x in self.worddict.src2tgt[lan]:
                return self.worddict.src2tgt[lan][x][random.randint(0,len(self.worddict.src2tgt[lan][x]) - 1)]
            else:
                return x
        else:
            return x

    def cross_list(self, x):
        return [self.cross(xx, not (self.training and self.args.train.ratio >= random.random())) for xx in x["utterance"]]

    def get_info(self, batch):
        premises = [self.cross_list(x)["premise"] for x in batch]
        hypotheses = [self.cross_list(x)["hypothesis"] for x in batch]
        inputs = self.tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")

        return inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device)

    def forward(self, batch):
        input_ids, attention_mask = self.get_info(batch)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, 768]
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