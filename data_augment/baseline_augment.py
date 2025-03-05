# CoSDA-ML's data augmentation framework, modified to run at once instead of a batch-by-batch basis
import argparse
import random
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TSV = os.path.join(BASE_DIR, "dataset/xnli/xnli.dev.tsv")
DICT_ZH = os.path.join(BASE_DIR, "dataset/dict/zh2.txt")
DICT_ES = os.path.join(BASE_DIR, "dataset/dict/es2.txt")
OUTPUT_TSV = os.path.join(BASE_DIR, "dataset/baseline_augmented/spanglish.tsv")
CROSS_PROB = 1.0
RATIO = 0.6
INPUT_COLS = ["sentence1", "sentence2"]

def load_dictionary(dict_path):
    mapping = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            src, tgt = parts[0], parts[1]
            src_lower = src.lower()
            if src_lower not in mapping:
                mapping[src_lower] = []
            mapping[src_lower].append(tgt)
    return mapping

def cross(word, dict_list, cross_prob):
    if random.random() < cross_prob:
        # pick which dictionary to use (0 => zh dict, 1 => es dict)
        lan = 1
        word_lower = word.lower()
        if word_lower in dict_list[lan]:
            possible_translations = dict_list[lan][word_lower]
            return random.choice(possible_translations)
        else:
            return word
    else:
        return word

def cross_str(sentence, dict_list, cross_prob, ratio):
    do_augment = (random.random() < ratio)

    words = sentence.strip().split()
    if do_augment:
        augmented = [
            cross(w, dict_list, cross_prob) for w in words
        ]
    else:
        augmented = words
    return " ".join(augmented)

def main():
    zh_dict = load_dictionary(DICT_ZH)
    es_dict = load_dictionary(DICT_ES)
    dict_list = [zh_dict, es_dict]

    with open(INPUT_TSV, "r", encoding="utf-8") as fin, \
         open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = ["original_sentence", "augmented_sentence"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            # only process rows where language is English ("en")
            if row.get("language", "").strip().lower() == "en":
                for col in INPUT_COLS:
                    if col in row and row[col].strip():
                        original = row[col]
                        augmented = cross_str(original, dict_list, CROSS_PROB, RATIO)
                        writer.writerow({"original_sentence": original, "augmented_sentence": augmented})


if __name__ == "__main__":
    main()
