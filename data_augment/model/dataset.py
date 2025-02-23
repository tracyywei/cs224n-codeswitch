import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class UDParsingDataset(Dataset):
    def __init__(self, file_path, tokenizer: BertTokenizerFast, pos_label2id=None, dep_label2id=None, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = []
        self.pos_labels = []
        self.dep_labels = []
        self.heads = []
        
        pos_set = set()
        dep_set = set()
        
        with open(file_path, encoding='utf-8') as f:
            sentence_tokens = []
            sentence_pos = []
            sentence_dep = []
            sentence_heads = []
            for line in f:
                line = line.strip()
                if not line:
                    # End of sentence; if tokens have been collected, append.
                    if sentence_tokens:
                        self.sentences.append(sentence_tokens)
                        self.pos_labels.append(sentence_pos)
                        self.dep_labels.append(sentence_dep)
                        self.heads.append(sentence_heads)
                        sentence_tokens = []
                        sentence_pos = []
                        sentence_dep = []
                        sentence_heads = []
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 10:
                    continue
                if len(parts) > 10:
                    parts = parts[:10]
                if "-" in parts[0] or "." in parts[0]:
                    continue
                token = parts[1]      # FORM
                pos = parts[3]        # UPOS
                dep = parts[7]        # DEPREL
                head = parts[6]       # HEAD
                sentence_tokens.append(token)
                sentence_pos.append(pos)
                sentence_dep.append(dep)
                sentence_heads.append(head)
                pos_set.add(pos)
                dep_set.add(dep)
            if sentence_tokens:
                self.sentences.append(sentence_tokens)
                self.pos_labels.append(sentence_pos)
                self.dep_labels.append(sentence_dep)
                self.heads.append(sentence_heads)
        
        # Build label mappings if not provided
        if pos_label2id is None:
            self.pos_label2id = {label: idx for idx, label in enumerate(sorted(list(pos_set)))}
        else:
            self.pos_label2id = pos_label2id
            
        if dep_label2id is None:
            self.dep_label2id = {label: idx for idx, label in enumerate(sorted(list(dep_set)))}
        else:
            self.dep_label2id = dep_label2id
        
        self.id2pos = {v: k for k, v in self.pos_label2id.items()}
        self.id2dep = {v: k for k, v in self.dep_label2id.items()}
        
        # Preprocess each sentence: tokenize and align labels
        self.examples = []
        for tokens, pos_tags, dep_tags, head_tags in zip(self.sentences, self.pos_labels, self.dep_labels, self.heads):
            encoding = self.tokenizer(tokens,
                                      is_split_into_words=True,
                                      truncation=True,
                                      padding='max_length',
                                      max_length=self.max_length,
                                      return_offsets_mapping=True)
            word_ids = encoding.word_ids()
            pos_label_ids = []
            dep_label_ids = []
            head_label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    pos_label_ids.append(-100)
                    dep_label_ids.append(-100)
                    head_label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    pos_label_ids.append(self.pos_label2id[pos_tags[word_idx]])
                    dep_label_ids.append(self.dep_label2id[dep_tags[word_idx]])
                    # Convert head index from string to integer:
                    # For non-root tokens, subtract 1 to convert from 1-indexed to 0-indexed.
                    # For a root token (head == "0"), use the token's own word index.
                    h = int(head_tags[word_idx])
                    if h == 0:
                        head_label_ids.append(word_idx)
                    else:
                        head_label_ids.append(h - 1)
                else:
                    pos_label_ids.append(-100)
                    dep_label_ids.append(-100)
                    head_label_ids.append(-100)
                previous_word_idx = word_idx
            encoding.pop("offset_mapping")
            encoding["pos_labels"] = pos_label_ids
            encoding["dep_labels"] = dep_label_ids
            encoding["head_labels"] = head_label_ids
            self.examples.append(encoding)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.examples[idx].items()}
        return item

# sanity check
if __name__ == "__main__":
    dummy_conllu = """
# newdoc id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713
# sent_id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713-0001
# newpar id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713-p0001
# text = From the AP comes this story :
1\tFrom\tfrom\tADP\tIN\t_\t3\tcase\t3:case\t_
2\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t3\tdet\t3:det\t_
3\tAP\tAP\tPROPN\tNNP\tNumber=Sing\t4\tobl\t4:obl:from\t_
4\tcomes\tcome\tVERB\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t0:root\t_
5\tthis\tthis\tDET\tDT\tNumber=Sing|PronType=Dem\t6\tdet\t6:det\t_
6\tstory\tstory\tNOUN\tNN\tNumber=Sing\t4\tnsubj\t4:nsubj\t_
7\t:\t:\tPUNCT\t:\t_\t4\tpunct\t4:punct\t_

# sent_id = test-s1
# text = 然而，这样的处理也衍生了一些问题。
# translit = rán'ér,zhèyàngdechùlǐyěyǎnshēngleyīxiēwèntí.
1\t然而\t然而\tSCONJ\tRB\t_\t7\tmark\t_\tSpaceAfter=No|Translit=rán'ér|LTranslit=rán'ér
2\t，\t，\tPUNCT\t,\t_\t1\tpunct\t_\tSpaceAfter=No|Translit=,|LTranslit=,
3\t这样\t这样\tPRON\tPRD\t_\t5\tdet\t_\tSpaceAfter=No|Translit=zhèyàng|LTranslit=zhèyàng
4\t的\t的\tPART\tDEC\tCase=Gen\t3\tcase\t_\tSpaceAfter=No|Translit=de|LTranslit=de
5\t处理\t处理\tNOUN\tNN\t_\t7\tnsubj\t_\tSpaceAfter=No|Translit=chùlǐ|LTranslit=chùlǐ
6\t也\t也\tSCONJ\tRB\t_\t7\tmark\t_\tSpaceAfter=No|Translit=yě|LTranslit=yě
7\t衍生\t衍生\tVERB\tVV\t_\t0\troot\t_\tSpaceAfter=No|Translit=yǎnshēng|LTranslit=yǎnshēng
8\t了\t了\tAUX\tAS\tAspect=Perf\t7\taux\t_\tSpaceAfter=No|Translit=le|LTranslit=le
9\t一些\t一些\tADJ\tJJ\t_\t10\tamod\t_\tSpaceAfter=No|Translit=yīxiē|LTranslit=yīxiē
10\t问题\t问题\tNOUN\tNN\t_\t7\tobj\t_\tSpaceAfter=No|Translit=wèntí|LTranslit=wèntí
11\t。\t。\tPUNCT\t.\t_\t7\tpunct\t_\tSpaceAfter=No|Translit=.|LTranslit=.
"""
    dummy_file = "dummy.conllu"
    with open(dummy_file, "w", encoding="utf-8") as f:
        f.write(dummy_conllu)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    dataset = UDParsingDataset(dummy_file, tokenizer, max_length=16)
    print("Number of examples:", len(dataset))
    for i in range(len(dataset)):
        example = dataset[i]
        tokens = tokenizer.convert_ids_to_tokens(example["input_ids"].tolist())
        print("Tokens:", tokens)
        print("POS labels:", example["pos_labels"].tolist())
        print("Dep labels:", example["dep_labels"].tolist())
        print("Head labels:", example["head_labels"].tolist())