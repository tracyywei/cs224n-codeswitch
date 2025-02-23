##! /bin/bash

# Pretrain the model
python model/run.py --train_file datasets/ud/combined_train.conllu --dev_file datasets/ud/combined_dev.conllu --output_model fine_tuned_bert.pt

# Evaluate trained model (sanity check)
python model/evaluate.py --mode eval_finetuned --input_file datasets/sample/sample.conllu --output_file datasets/sample/sample_eval.txt --model_path fine_tuned_bert.pt

# Evaluate trained model (on test conllu file)
python model/evaluate.py --mode eval_finetuned --input_file datasets/ud/en_ewt-ud-test.conllu --output_file datasets/ud/en_ewt_eval.txt --model_path fine_tuned_bert.pt
