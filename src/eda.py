"""
exploratory data analysis on the docs.
"""

import os
import pandas as pd
from collections import Counter
from preprocess import clean_text

DATA_DIR = "docs"

all_docs = []

for file in os.listdir(DATA_DIR):
    with open(os.path.join(DATA_DIR, file), "r") as f:
        raw_text = f.read()
    cleaned = clean_text(raw_text)
    word_count = len(cleaned.split())
    all_docs.append({"file": file, "word_count": word_count})


summary_df = pd.DataFrame(all_docs)
print("\nDocument-Level Stats:")
print(summary_df.describe())

# Top words
all_words = []
for doc in all_docs:
    with open(os.path.join(DATA_DIR, doc["file"])) as f:
        all_words.extend(clean_text(f.read()).lower().split())

word_freq = Counter(all_words)
print("\nMost common words:")
print(word_freq.most_common(20))
