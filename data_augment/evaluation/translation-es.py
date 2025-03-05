import csv
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm

nltk.download('all')

en_es_model = 'Helsinki-NLP/opus-mt-en-es'  # English → spanish
es_en_model = 'Helsinki-NLP/opus-mt-es-en'  # spanish → English

tokenizer_en_es = MarianTokenizer.from_pretrained(en_es_model)
model_en_es = MarianMTModel.from_pretrained(en_es_model)

tokenizer_es_en = MarianTokenizer.from_pretrained(es_en_model)
model_es_en = MarianMTModel.from_pretrained(es_en_model)

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
    
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)

    # Apply smoothing to BLEU score to avoid zero scores
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)
    
    meteor = meteor_score([reference_tokens], hypothesis_tokens)
    chrf = sentence_chrf(reference, hypothesis)
    return bleu, meteor, chrf

def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1 

def evaluate_tsv(file_path):
    results = []
    avg_bleu_en = avg_meteor_en = avg_chrf_en = 0.0
    avg_bleu_es = avg_meteor_es = avg_chrf_es = 0.0
    count = 0
    total_lines = count_lines(file_path)

    with open(file_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in tqdm(reader, total=total_lines, desc="processing sentences", ncols=80, ascii=True):
            original_english = row["original_english"].strip()
            augmented_english = row["augmented_spanglish"].strip()
            spanish_sentence = row["original_spanish"].strip()

            if not original_english or not augmented_english or not spanish_sentence:
                print(f"⚠️ Skipping empty row at line {count+1}")
                continue  # Skip empty rows

            # back translate Augmented English → English
            back_translated_english = translate(augmented_english, tokenizer_es_en, model_es_en)

            # back translate Augmented English → spanish
            back_translated_spanish = translate(augmented_english, tokenizer_en_es, model_en_es)

            bleu_en, meteor_en, chrf_en = compute_similarity_scores(original_english, back_translated_english)
            bleu_es, meteor_es, chrf_es = compute_similarity_scores(spanish_sentence, back_translated_spanish)

            results.append({
                "original_english": original_english,
                "augmented_spanglish": augmented_english,
                "original_spanish": spanish_sentence,
                "back_translated_english": back_translated_english,
                "bleu_en": bleu_en, "meteor_en": meteor_en, "chrf_en": chrf_en,
                "back_translated_spanish": back_translated_spanish,
                "bleu_es": bleu_es, "meteor_es": meteor_es, "chrf_es": chrf_es,
            })

            avg_bleu_en += bleu_en
            avg_meteor_en += meteor_en
            avg_chrf_en += chrf_en
            avg_bleu_es += bleu_es
            avg_meteor_es += meteor_es
            avg_chrf_es += chrf_es
            count += 1

    if count > 0:
        avg_bleu_en /= count
        avg_meteor_en /= count
        avg_chrf_en /= count
        avg_bleu_es /= count
        avg_meteor_es /= count
        avg_chrf_es /= count

    print(f"\n### Evaluation Summary ###")
    print(f"Avg BLEU (English): {avg_bleu_en:.4f} | Avg METEOR (English): {avg_meteor_en:.4f} | Avg ChrF (English): {avg_chrf_en:.4f}")
    print(f"Avg BLEU (spanish): {avg_bleu_es:.4f} | Avg METEOR (spanish): {avg_meteor_es:.4f} | Avg ChrF (spanish): {avg_chrf_es:.4f}")

    return results

tsv_file_path = "dataset/baseline_augmented/spanglish_health.tsv"
results = evaluate_tsv(tsv_file_path)
