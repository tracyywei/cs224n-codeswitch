# CoSDA-ML's data augmentation framework, modified to run at once instead of a batch-by-batch basis
import argparse
import random
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TSV = os.path.join(BASE_DIR, "datasets/xnli/xnli.dev.tsv")
DICT_ZH = os.path.join(BASE_DIR, "datasets/dict/zh2.txt")
DICT_ES = os.path.join(BASE_DIR, "datasets/dict/es2.txt")
OUTPUT_TSV = os.path.join(BASE_DIR, "augmented_output.tsv")
CROSS_PROB = 1.0
RATIO = 0.6
INPUT_COLS = ["sentence1", "sentence2"]

def load_dictionary(dict_path):
    """
    Reads a bilingual dictionary file where each line has:
        SRC_WORD  TGT_WORD
    Returns a dict {src_word: [possible_translations]}.
    """
    mapping = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on whitespace or tab
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
    """
    Randomly replaces a single word if random() < cross_prob
    using one of the dictionaries in dict_list (the 'Chinglish'
    and 'Spanglish' dictionaries).
    This mirrors your original logic, where:
      - cross_prob ~ args.train.cross
    """
    if random.random() < cross_prob:
        # randomly pick which dictionary to use (e.g. 0 => zh dict, 1 => es dict)
        lan = random.randint(0, len(dict_list) - 1)
        # word is used in lowercase in your code
        word_lower = word.lower()
        # see if the current dictionary has a translation
        if word_lower in dict_list[lan]:
            possible_translations = dict_list[lan][word_lower]
            # pick one at random
            return random.choice(possible_translations)
        else:
            return word  # no translation
    else:
        return word

def cross_str(sentence, dict_list, cross_prob, ratio):
    """
    Splits a sentence into words, and for each word, we
    optionally do a cross-lingual replacement. The ratio
    controls whether we do *any* replacements in the entire sentence:
      - If random.random() < ratio, we attempt cross-lingual replacements.
      - Otherwise, we skip augmenting this entire sentence.

    This matches your code's idea:
      cross_list(x, disable=not (training and args.train.ratio >= random.random()))
    but implemented for a single sentence.
    """
    # Decide if we will or won't augment this entire sentence
    # according to ratio (like "if ratio >= random.random() then augment").
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
            # Only process rows where language is English ("en")
            if row.get("language", "").strip().lower() == "en":
                for col in INPUT_COLS:
                    if col in row and row[col].strip():
                        original = row[col]
                        augmented = cross_str(original, dict_list, CROSS_PROB, RATIO)
                        writer.writerow({"original_sentence": original, "augmented_sentence": augmented})


if __name__ == "__main__":
    main()
