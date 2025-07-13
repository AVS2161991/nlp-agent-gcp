"""
exploratory data analysis on the docs.
"""

import os
from typing import Tuple
from collections import Counter
import pandas as pd
from src.preprocess import clean_text


def run_eda(data_dir: str = "docs") -> Tuple[pd.DataFrame, Counter]:
    """
    Perform exploratory data analysis (EDA) on a directory of text documents.

    This function reads all `.txt` files from the specified directory, applies basic preprocessing
    using `clean_text()`, and computes document-level statistics and word frequency distribution.

    Args:
        data_dir (str): Path to the directory containing text documents. Default is "docs".

    Returns:
        summary_df (pd.DataFrame): A DataFrame with file names and their corresponding word counts.
        word_freq (Counter): A Counter object with word frequency counts across all documents.

    Prints:
        - Summary statistics (count, mean, std, min, max) of word counts per document.
        - Top 20 most frequent words across all documents.
    """
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
