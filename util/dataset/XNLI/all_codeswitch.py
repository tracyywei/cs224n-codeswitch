import os
import random

import util.data
import util.convert
import util.tool
from datasets import load_dataset, Dataset

class DatasetTool(object):
    def get_set(file):
        dataset = []
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

        # modified to handle huggingface xnli datasets
        for example in file:
            label = example["label"]
            if isinstance(label, str) and label not in label_map:
                continue 

            dataset.append({
                "premise": example["premise"],
                "hypothesis": example["hypothesis"],
                "label": label_map.get(label, label)
            })

        return dataset

    def get_idx_dict(idx_dict, file, args):
        raw = util.data.Delexicalizer.remove_linefeed(util.data.Reader.read_raw(file))
        if args.train.dict_size is not None:
            raw = raw[:int(len(raw) * args.train.dict_size)]
        idx_dict.src2tgt.append({})
        for line in raw:
            try:
                src, tgt = line.split("\t")
            except:
                src, tgt = line.split(" ")
            
            if src not in idx_dict.src2tgt[-1]:
                idx_dict.src2tgt[-1][src] = [tgt]
            else:
                idx_dict.src2tgt[-1][src].append(tgt)

    def get(args):
        train_file = load_dataset("facebook/xnli", "en", split="train")
        # REPLACE TRAIN_FILE WITH CODESWITCHED XNLI FILE
        dev_file = load_dataset("facebook/xnli", "en", split="validation")
        test_file = load_dataset("facebook/xnli", "hi", split="test")

        train = DatasetTool.get_set(train_file)
        random.shuffle(train)
        dev = DatasetTool.get_set(dev_file)
        test = DatasetTool.get_set(test_file)

        # passing the dictionary as an arg
        args.dict_list = args.dataset.dict.split(" ")
        idx_dict = util.convert.Common.to_args({"src2tgt": []})
        for dict_file in args.dict_list:
            dict_file = os.path.join(args.dir.dataset, dict_file)
            DatasetTool.get_idx_dict(idx_dict, dict_file, args)
        if args.train.train_size is not None:
            train = train[:int(len(train) * args.train.train_size)]
        return train, dev, test, None, idx_dict, None

    def evaluate(pred, dataset, args):
        total = len(dataset)
        correct = sum(1 for p, g in zip(pred, dataset) if p == g["label"])
        accuracy = correct / total if total > 0 else 0.0

        print(f"XNLI Evaluation Accuracy: {accuracy * 100:.2f}%")
        return {"accuracy": accuracy}

    def record(pred, dataset, set_name, args):
        pass