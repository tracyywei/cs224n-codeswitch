import argparse
import csv
import torch
from transformers import BertTokenizerFast
from dataset import UDParsingDataset
from model import BertForParsing

# evaluating the fine-tuned model on a test CoNLL-U file
def evaluate_finetuned(test_file, output_file, model, tokenizer, config, pos_label2id, dep_label2id):
    test_dataset = UDParsingDataset(test_file, tokenizer, 
                                    pos_label2id=pos_label2id, 
                                    dep_label2id=dep_label2id,
                                    max_length=config.max_length)
    model.eval()
    id2pos = {v: k for k, v in pos_label2id.items()}
    id2dep = {v: k for k, v in dep_label2id.items()}

    with torch.no_grad(), open(output_file, "w", encoding="utf-8") as fout:
        for example in test_dataset:
            input_ids = example["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = example["attention_mask"].unsqueeze(0).to(model.device)
            pos_logits, dep_logits, head_logits, _ = model(input_ids, attention_mask,
                                                           pos_labels=None, dep_labels=None, head_labels=None)
            pos_preds = pos_logits.argmax(dim=-1).squeeze(0).tolist()
            dep_preds = dep_logits.argmax(dim=-1).squeeze(0).tolist()

            valid_indices = [i for i, lbl in enumerate(example["pos_labels"].tolist()) if lbl != -100]
            pos_tags = [id2pos[pos_preds[i]] for i in valid_indices]
            dep_tags = [id2dep[dep_preds[i]] for i in valid_indices]

            text = tokenizer.decode(example["input_ids"].tolist(), skip_special_tokens=True)
            fout.write(text + "\t" + " ".join(pos_tags) + "\t" + " ".join(dep_tags) + "\n")

# annotate new data from XNLI file with finetuned mBERT -> output will be input for seq2seq model
def annotate_xnli(tsv_file, output_file, model, tokenizer, config):
    model.eval()
    with open(tsv_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            lang = row.get("language", "").lower()
            # only process english, chinese, and spanish
            if lang not in {"en", "zh", "es"}:
                continue
            sentence = row.get("sentence", "")
            if not sentence.strip():
                continue
            
            # tokenizing the sentence into words - english/spanish: whitespace, chinese: split by characters
            if lang in {"en", "es"}:
                words = sentence.split()
            elif lang == "zh":
                words = list(sentence)
            else:
                words = sentence.split()

            encoding = tokenizer(words,
                                 is_split_into_words=True,
                                 truncation=True,
                                 padding="max_length",
                                 max_length=config.max_length,
                                 return_offsets_mapping=True)
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
            pos_tags = [str(pred_pos[i]) for i in range(len(words))]
            dep_tags = [str(pred_dep[i]) for i in range(len(words))]
            fout.write(sentence + "\t" + " ".join(pos_tags) + "\t" + " ".join(dep_tags) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval_conllu", "annotate_xnli"], required=True)
    parser.add_argument("--input_file", type=str, required=True, help="Path to input file (test CoNLL-U or XNLI TSV)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the annotated output")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model state dict")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

    state_dict = torch.load(args.model_path, map_location="cpu")
    pos_label2id = state_dict["pos_label2id"]
    dep_label2id = state_dict["dep_label2id"]
    max_length = state_dict["max_length"]
    num_pos_labels = len(pos_label2id)
    num_dep_labels = len(dep_label2id)

    model = BertForParsing(num_pos_labels, num_dep_labels, max_length=args.max_length, pretrained_model_name=args.pretrained_model)
    model.load_state_dict(state_dict["model_state_dict"])
    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model.device)

    class Config:
        pass
    config = Config()
    config.max_length = args.max_length

    if args.mode == "evaluate_finetuned":
        evaluate_finetuned(args.input_file, args.output_file, model, tokenizer, config, pos_label2id, dep_label2id)
    elif args.mode == "annotate_xnli":
        annotate_xnli(args.input_file, args.output_file, model, tokenizer, config)

if __name__ == "__main__":
    main()
