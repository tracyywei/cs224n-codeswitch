import csv
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
from transformers import MarianMTModel, MarianTokenizer
import torch

nltk.download('all')

en_zh_model = 'Helsinki-NLP/opus-mt-en-zh'  # English → Chinese
zh_en_model = 'Helsinki-NLP/opus-mt-zh-en'  # Chinese → English

tokenizer_en_zh = MarianTokenizer.from_pretrained(en_zh_model)
model_en_zh = MarianMTModel.from_pretrained(en_zh_model)

tokenizer_zh_en = MarianTokenizer.from_pretrained(zh_en_model)
model_zh_en = MarianMTModel.from_pretrained(zh_en_model)

def translate(text, tokenizer, model):
    if not text.strip():
        return ""  # Handle empty input
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def compute_similarity_scores(reference, hypothesis):
    if not reference.strip() or not hypothesis.strip():
        return 0.0, 0.0, 0.0  # Handle empty input
    
    reference_tokens = nltk.word_tokenize(reference)  # Tokenize reference sentence
    hypothesis_tokens = nltk.word_tokenize(hypothesis)  # Tokenize hypothesis sentence

    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)
    
    meteor = meteor_score([reference_tokens], hypothesis_tokens)
    chrf = sentence_chrf(reference, hypothesis)
    return bleu, meteor, chrf

def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for header row

def evaluate_tsv(file_path):
    results = []
    avg_bleu_en = avg_meteor_en = avg_chrf_en = 0.0
    avg_bleu_zh = avg_meteor_zh = avg_chrf_zh = 0.0
    count = 0
    total_lines = count_lines(file_path)

    with open(file_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in tqdm(reader, total=total_lines, desc="processing sentences", ncols=80, ascii=True):
            original_english = row["original_english"]
            augmented_english = row["augmented_chinglish"]
            chinese_sentence = row["original_chinese"]

            # back translate Augmented English → English
            back_translated_english = translate(augmented_english, tokenizer_zh_en, model_zh_en)

            # back translate Augmented English → Chinese
            back_translated_chinese = translate(augmented_english, tokenizer_en_zh, model_en_zh)

            bleu_en, meteor_en, chrf_en = compute_similarity_scores(original_english, back_translated_english)
            bleu_zh, meteor_zh, chrf_zh = compute_similarity_scores(chinese_sentence, back_translated_chinese)

            results.append({
                "original_english": original_english,
                "augmented_chinglish": augmented_english,
                "original_chinese": chinese_sentence,
                "back_translated_english": back_translated_english,
                "bleu_en": bleu_en, "meteor_en": meteor_en, "chrf_en": chrf_en,
                "back_translated_chinese": back_translated_chinese,
                "bleu_zh": bleu_zh, "meteor_zh": meteor_zh, "chrf_zh": chrf_zh,
            })

            avg_bleu_en += bleu_en
            avg_meteor_en += meteor_en
            avg_chrf_en += chrf_en
            avg_bleu_zh += bleu_zh
            avg_meteor_zh += meteor_zh
            avg_chrf_zh += chrf_zh
            count += 1

    if count > 0:
        avg_bleu_en /= count
        avg_meteor_en /= count
        avg_chrf_en /= count
        avg_bleu_zh /= count
        avg_meteor_zh /= count
        avg_chrf_zh /= count

    print(f"\n### Evaluation Summary ###")
    print(f"Avg BLEU (English): {avg_bleu_en:.4f} | Avg METEOR (English): {avg_meteor_en:.4f} | Avg ChrF (English): {avg_chrf_en:.4f}")
    print(f"Avg BLEU (Chinese): {avg_bleu_zh:.4f} | Avg METEOR (Chinese): {avg_meteor_zh:.4f} | Avg ChrF (Chinese): {avg_chrf_zh:.4f}")

    return results

tsv_file_path = "dataset/baseline_augmented/chinglish_health.tsv"
results = evaluate_tsv(tsv_file_path)
