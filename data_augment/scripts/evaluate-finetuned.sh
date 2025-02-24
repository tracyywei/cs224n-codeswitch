##! /bin/bash

# Evaluate trained model (sanity check)
python model/evaluate.py --mode eval_finetuned --input_file datasets/sample/sample.conllu --output_file outputs/sample_eval.txt --model_path fine_tuned_bert.pt

# Evaluate trained model (on test conllu file)
python model/evaluate.py --mode eval_finetuned --input_file datasets/ud/en_ewt-ud-test.conllu --output_file outputs/en_ewt_eval.txt --model_path fine_tuned_bert.pt

# Compute accuracy score
python model/evaluate.py --mode compute_accuracy --input_file datasets/ud/en_ewt-ud-test.conllu --output_file outputs/en_eval_accuracy.txt --model_path fine_tuned_bert.pt