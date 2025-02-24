##! /bin/bash

# Pretrain the model
python model/run.py --train_file datasets/ud/combined_train.conllu --dev_file datasets/ud/combined_dev.conllu --output_model fine_tuned_bert.pt