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

    def get(args):
        train_file = load_dataset("facebook/xnli", "en", split="train")
        dev_file = load_dataset("facebook/xnli", "en", split="validation")
        test_file = load_dataset("facebook/xnli", "hi", split="test")

        train = DatasetTool.get_set(train_file)
        random.shuffle(train)
        dev = DatasetTool.get_set(dev_file)
        test = DatasetTool.get_set(test_file)

        return train, dev, test, None, None, None

    def evaluate(pred, dataset, args):
        total = len(dataset)
        correct = sum(1 for p, g in zip(pred, dataset) if p == g["label"])
        accuracy = correct / total if total > 0 else 0.0

        print(f"XNLI Evaluation Accuracy: {accuracy * 100:.2f}%")
        return {"accuracy": accuracy}
