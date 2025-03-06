import csv
import nltk
import pickle
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
from transformers import MarianMTModel, MarianTokenizer

nltk.download('all', quiet=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

en_zh_model = 'Helsinki-NLP/opus-mt-en-zh'
zh_en_model = 'Helsinki-NLP/opus-mt-zh-en'

tokenizer_en_zh = MarianTokenizer.from_pretrained(en_zh_model)
model_en_zh = MarianMTModel.from_pretrained(en_zh_model).to(device).half()

tokenizer_zh_en = MarianTokenizer.from_pretrained(zh_en_model)
model_zh_en = MarianMTModel.from_pretrained(zh_en_model).to(device).half()

def compute_similarity_scores(params):
    reference, hypothesis = params
    if not reference.strip() or not hypothesis.strip():
        return 0.0, 0.0, 0.0
    
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)

    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)
    meteor = meteor_score([reference_tokens], hypothesis_tokens)
    chrf = sentence_chrf(reference, hypothesis) 
    return bleu, meteor, chrf

def translate_batch(texts, tokenizer, model):
    if not texts:
        return []
    
    translated_texts = []
    batch_size = 64 if torch.cuda.is_available() else 16

    for i in tqdm(range(0, len(texts), batch_size), desc=desc, ncols=80, ascii=True):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs)
        translated_texts.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    
    return translated_texts

def evaluate_tsv(file_path):
    results = []
    with open(file_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        data = list(reader)

    total_lines = len(data)
    augmented_english_sentences = [row["augmented_chinglish"].strip() for row in data]

    try:
        with open("saved_translations.pkl", "rb") as f:
            translations = pickle.load(f)
        back_translated_english = translations["back_translated_english"]
        back_translated_chinese = translations["back_translated_chinese"]
    except FileNotFoundError:
        back_translated_chinese = translate_batch(augmented_english_sentences, tokenizer_en_zh, model_en_zh)
        back_translated_english = translate_batch(augmented_english_sentences, tokenizer_zh_en, model_zh_en)

        translations = {
            "back_translated_english": back_translated_english,
            "back_translated_chinese": back_translated_chinese
        }
        with open("saved_translations.pkl", "wb") as f:
            pickle.dump(translations, f)

    en_pairs = [(row["original_english"].strip(), back_translated_english[i]) for i, row in enumerate(data)]
    zh_pairs = [(row["original_chinese"].strip(), back_translated_chinese[i]) for i, row in enumerate(data)]
    with Pool(cpu_count()) as pool:
        en_scores = list(tqdm(pool.imap(compute_similarity_scores, en_pairs), total=total_lines, desc="Scoring English", ncols=80, ascii=True))
        zh_scores = list(tqdm(pool.imap(compute_similarity_scores, zh_pairs), total=total_lines, desc="Scoring Chinese", ncols=80, ascii=True))

    avg_bleu_en = avg_meteor_en = avg_chrf_en = 0.0
    avg_bleu_zh = avg_meteor_zh = avg_chrf_zh = 0.0

    for i, row in enumerate(data):
        bleu_en, meteor_en, chrf_en = en_scores[i]
        bleu_zh, meteor_zh, chrf_zh = zh_scores[i]

        results.append({
            "original_english": row["original_english"].strip(),
            "augmented_chinglish": row["augmented_chinglish"].strip(),
            "original_chinese": row["original_chinese"].strip(),
            "back_translated_english": back_translated_english[i],
            "bleu_en": bleu_en, "meteor_en": meteor_en, "chrf_en": chrf_en,
            "back_translated_chinese": back_translated_chinese[i],
            "bleu_zh": bleu_zh, "meteor_zh": meteor_zh, "chrf_zh": chrf_zh,
        })

        avg_bleu_en += bleu_en
        avg_meteor_en += meteor_en
        avg_chrf_en += chrf_en
        avg_bleu_zh += bleu_zh
        avg_meteor_zh += meteor_zh
        avg_chrf_zh += chrf_zh

    total = len(data)
    if total > 0:
        avg_bleu_en /= total
        avg_meteor_en /= total
        avg_chrf_en /= total
        avg_bleu_zh /= total
        avg_meteor_zh /= total
        avg_chrf_zh /= total

    print(f"\n### Evaluation Summary ###")
    print(f"Avg BLEU (English): {avg_bleu_en:.4f} | Avg METEOR (English): {avg_meteor_en:.4f} | Avg ChrF (English): {avg_chrf_en:.4f}")
    print(f"Avg BLEU (Chinese): {avg_bleu_zh:.4f} | Avg METEOR (Chinese): {avg_meteor_zh:.4f} | Avg ChrF (Chinese): {avg_chrf_zh:.4f}")

    return results

tsv_file_path = "dataset/baseline_augmented/chinglish_health.tsv"
results = evaluate_tsv(tsv_file_path)
