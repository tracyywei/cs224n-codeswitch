##! /bin/bash

# annotate XNLI dataset - test
python model/evaluate.py --mode annotate_xnli --input_file datasets/xnli/xnli.test.tsv --output_file outputs/xnli_annotated_test.txt --model_path fine_tuned_bert.pt

# annotate XNLI dataset - dev
python model/evaluate.py --mode annotate_xnli --input_file datasets/xnli/xnli.dev.tsv --output_file outputs/xnli_annotated_dev.txt --model_path fine_tuned_bert.pt