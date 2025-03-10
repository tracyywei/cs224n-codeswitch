import os
import argparse
import torch
from transformers import BertTokenizerFast
from dataset import UDParsingDataset
from model import BertForParsing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HINGLISH_FILE = os.path.join(BASE_DIR, "../dataset/enghinglish/dev.txt")


def annotate_file(input_file, output_file, model, tokenizer, config, pos_label2id, dep_label2id):
    model.eval()
    id2pos_label = {v: k for k, v in pos_label2id.items()}
    id2dep_label = {v: k for k, v in dep_label2id.items()}
    
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) < 1:
                continue
            
            english_sentence = parts[0].strip()
            if not english_sentence:
                continue
            
            words = english_sentence.split()
            encoding = tokenizer(words, is_split_into_words=True, truncation=True, padding="max_length", 
                                 max_length=config.max_length, return_offsets_mapping=True)
            word_ids = encoding.word_ids()
            
            input_ids = torch.tensor([encoding["input_ids"]]).to(model.device)
            attention_mask = torch.tensor([encoding["attention_mask"]]).to(model.device)
            
            with torch.no_grad():
                pos_logits, dep_logits, head_logits, _ = model(input_ids, attention_mask, 
                                                               pos_labels=None, dep_labels=None, head_labels=None)
            
            pos_preds = pos_logits.argmax(dim=-1).squeeze(0).tolist()
            dep_preds = dep_logits.argmax(dim=-1).squeeze(0).tolist()
            
            pred_pos = {}
            pred_dep = {}
            for idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx not in pred_pos:
                    pred_pos[word_idx] = pos_preds[idx]
                    pred_dep[word_idx] = dep_preds[idx]
            
            pos_tags = [id2pos_label[p] if p in id2pos_label else "UNK" for p in pred_pos]
            dep_tags = [id2dep_label[d] if d in id2dep_label else "UNK" for d in pred_dep]
            fout.write(english_sentence + "\t" + " ".join(pos_tags) + "\t" + " ".join(dep_tags) + "\n")


def annotate_hinglish_file(model, tokenizer, config, pos_label2id, dep_label2id):
    output_dir = os.path.join(BASE_DIR, "annotated")
    os.makedirs(output_dir, exist_ok=True)
    annotate_file(HINGLISH_FILE, os.path.join(output_dir, "annotated_hinglish_en.txt"), model, tokenizer, config, pos_label2id, dep_label2id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model state dict")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    pos_label2id = state_dict["pos_label2id"]
    dep_label2id = state_dict["dep_label2id"]
    max_length = state_dict["max_length"]
    num_pos_labels = len(pos_label2id)
    num_dep_labels = len(dep_label2id)

    model = BertForParsing(num_pos_labels, num_dep_labels, max_length=max_length)
    model.load_state_dict(state_dict["model_state_dict"])
    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model.device)
    
    class Config:
        pass
    config = Config()
    config.max_length = args.max_length

    annotate_hinglish_file(model, tokenizer, config, pos_label2id, dep_label2id)
