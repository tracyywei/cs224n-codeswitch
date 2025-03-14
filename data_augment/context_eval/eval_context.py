import re
import collections

pos_pattern = re.compile(r"\[POS\]?|\<POS\>|\[POS>\]|\[POS>|ADJ|ADP|ADV|AUX|CCONJ|DET|INTJ|IUN|NUM|PART|UNK|cCONJ")
dep_pattern = re.compile(r"\[DEP\]?|\[DEP|\[DEP>\]|\[DEP>|\[DEP\?\?]|acl|acl\-oh|nly|acl:relcl|acl|relcl|advcl|\?\?|cconJ|acll")

def analyze_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    total_tokens = 0
    pos_count = 0
    dep_count = 0
    gibberish_count = 0

    for line in lines:
        tokens = line.strip().split()
        total_tokens += len(tokens)

        for token in tokens:
            if pos_pattern.search(token):
                pos_count += 1
            elif dep_pattern.search(token):
                dep_count += 1
            elif not token.isalpha():
                gibberish_count += 1

    pos_percentage = (pos_count / total_tokens) * 100 if total_tokens else 0
    dep_percentage = (dep_count / total_tokens) * 100 if total_tokens else 0
    gibberish_percentage = (gibberish_count / total_tokens) * 100 if total_tokens else 0

    print(f"POS Tags: {pos_percentage:.2f}%")
    print(f"Dependency Labels: {dep_percentage:.2f}%")
    print(f"Gibberish: {gibberish_percentage:.2f}%")

analyze_dataset("data_augment/context_eval/codeswitched_hinglish_en_test.txt")
