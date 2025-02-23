##! /bin/bash

# annotate sanity check
python model/evaluate.py --mode annotate_xnli --input_file sample.sample_xnli.tsv --output_file xnli/sample_annotated.txt --model_path fine_tuned_bert.pt

# annotate XNLI dataset
python model/evaluate.py --mode annotate_xnli --input_file xnli/xnli.test.tsv --output_file xnli/xnli_annotated.txt --model_path fine_tuned_bert.pt