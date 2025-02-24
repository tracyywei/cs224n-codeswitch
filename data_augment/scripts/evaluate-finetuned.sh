##! /bin/bash

# Evaluate trained model (on test conllu file)
python model/evaluate.py --mode eval_finetuned --input_file datasets/ud/combined_test.conllu --output_file outputs/combined_eval.txt --model_path fine_tuned_bert.pt

# Compute accuracy score
python model/evaluate.py --mode compute_accuracy --input_file datasets/ud/combined_test.conllu --output_file outputs/combined_accuracy.txt --model_path fine_tuned_bert.pt