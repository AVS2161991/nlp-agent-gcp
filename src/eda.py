"""
exploratory data analysis on the docs.
"""

import os
import pandas as pd
from collections import Counter
from src.preprocess import clean_text


def run_eda(data_dir="docs"):
    all_docs = []
    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file), "r") as f:
            raw_text = f.read()
        cleaned = clean_text(raw_text)
        word_count = len(cleaned.split())
        all_docs.append({"file": file, "word_count": word_count})

    summary_df = pd.DataFrame(all_docs)

    all_words = []
    for doc in all_docs:
        with open(os.path.join(data_dir, doc["file"])) as f:
            all_words.extend(clean_text(f.read()).lower().split())

    word_freq = Counter(all_words)

    if not summary_df.empty:
        print("\nDocument-Level Stats:")
        print(summary_df.describe())
    else:
        print("\nNo documents found.")

    print("\nMost common words:")
    print(word_freq.most_common(20))

    return summary_df, word_freq


if __name__ == "__main__":
    run_eda(data_dir="docs")
