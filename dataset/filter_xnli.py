import pandas as pd
import os

def extract_hindi_xnli(input_file, output_file):
    df = pd.read_csv(input_file, delimiter='\t')
    hindi_df = df[df['language'] == 'hi']
    hindi_df.to_csv(output_file, sep='\t', index=False)
    print(f"Extracted {len(hindi_df)} Hindi sentences and saved to {output_file}")

def extract_english_xnli(input_file, output_file):
    df = pd.read_csv(input_file, delimiter='\t')
    english_df = df[df['language'] == 'en']
    english_df.to_csv(output_file, sep='\t', index=False)
    print(f"Extracted {len(english_df)} English sentences and saved to {output_file}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
input_xnli_path = os.path.join(BASE_DIR, "./XNLI/xnli.test.tsv")
output_hindi_path = os.path.join(BASE_DIR, "./XNLI/xnli.hindi.tsv")
output_english_path = os.path.join(BASE_DIR, "./XNLI/xnli.english.tsv")
extract_hindi_xnli(input_xnli_path, output_hindi_path)
extract_english_xnli(input_xnli_path, output_english_path)
