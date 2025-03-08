# CoSDA-ML's data augmentation framework, modified to run at once instead of a batch-by-batch basis
import random
import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EN_FILE = os.path.join(BASE_DIR, "dataset/parallel/en-es.txt/tico-19.en-es.en")
ES_FILE = os.path.join(BASE_DIR, "dataset/parallel/en-es.txt/tico-19.en-es.es")
DICT_ZH = os.path.join(BASE_DIR, "dataset/dict/zh2.txt")
DICT_ES = os.path.join(BASE_DIR, "dataset/dict/es2.txt")
OUTPUT_TSV = os.path.join(BASE_DIR, "dataset/baseline_augmented/spanglish_health.tsv")
CROSS_PROB = 1.0
RATIO = 0.6

def load_dictionary(dict_path):
    mapping = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src, tgt = parts[0], parts[1]
                src_lower = src.lower()
                if src_lower not in mapping:
                    mapping[src_lower] = []
                mapping[src_lower].append(tgt)
    return mapping

def cross(word, dict_list, cross_prob):
    if random.random() < cross_prob:
        lan = 1  # 0 = zh, 1 = es
        word_lower = word.lower()
        if word_lower in dict_list[lan]:
            return random.choice(dict_list[lan][word_lower])
    return word

def cross_str(sentence, dict_list, cross_prob, ratio):
    if random.random() < ratio:
        words = sentence.strip().split()
        augmented = [cross(w, dict_list, cross_prob) for w in words]
        return " ".join(augmented)
    return sentence

def augment_and_save():
    zh_dict = load_dictionary(DICT_ZH)
    es_dict = load_dictionary(DICT_ES)
    dict_list = [zh_dict, es_dict]

    with open(EN_FILE, "r", encoding="utf-8") as en_f, \
         open(ES_FILE, "r", encoding="utf-8") as es_f, \
         open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as out_f:

        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["original_english", "augmented_spanglish", "original_spanish"])

        for line_idx, (en_line, es_line) in enumerate(zip(en_f, es_f)):
            en_sentence = en_line.strip()
            es_sentence = es_line.strip()

            # Ensure valid sentence pair
            if not en_sentence or not es_sentence:
                print(f"⚠️ Skipping empty line at index {line_idx}")
                continue

            # Apply augmentation to the English sentence
            augmented_en = cross_str(en_sentence, dict_list, CROSS_PROB, RATIO)

            writer.writerow([en_sentence, augmented_en, es_sentence])

if __name__ == "__main__":
    augment_and_save()