"""
agentic execution across all modules.
"""

import os
from src.load_data import download_blobs
from src.preprocess import clean_text
from src.extract import extract_entities_sentiment
from src.summarize import generate_summary_variants
from src.evaluate import evaluate_summary

BUCKET_NAME = "my-nlp-agent-bucket"
PREFIX = "docs/"
LOCAL_DIR = "downloads_gcp"


os.makedirs(LOCAL_DIR, exist_ok=True)
download_blobs(BUCKET_NAME, PREFIX, LOCAL_DIR)

for file in os.listdir(LOCAL_DIR):
    with open(os.path.join(LOCAL_DIR, file), "r") as f:
        raw_text = f.read()

    print(f"\n=== {file} ===")

    cleaned = clean_text(raw_text)

    # Generate summaries with different prompts
    summary_variants = generate_summary_variants(cleaned)
    reference_summary = cleaned[:300]

    BEST_SCORE = 0
    BEST_SUMMARY = ""
    BEST_PROMPT = ""
    all_summary_scores = []

    for prompt_template, summary in summary_variants:
        scores = evaluate_summary(reference_summary, summary)
        rouge1_f1 = scores["rouge1"].fmeasure
        all_summary_scores.append((prompt_template, rouge1_f1))
        if rouge1_f1 > BEST_SCORE:
            BEST_SCORE = rouge1_f1
            BEST_SUMMARY = summary
            BEST_PROMPT = prompt_template

    # Display all prompt performance
    print("\n--- Summary Evaluation for All Prompts ---")
    for template, score in all_summary_scores:
        print(f"Prompt: {template[:40]}... | ROUGE-1 F1: {score:.4f}")

    print("\n--- Best Summary ---")
    print("Prompt used:", BEST_PROMPT)
    print("Summary:", BEST_SUMMARY)

    # Extract entities and sentiment
    insights = extract_entities_sentiment(cleaned)

    print("\n--- Entities & Sentiment ---")
    print("Entities:", insights["entities"])
    print("Sentiment:", insights["sentiment"])
