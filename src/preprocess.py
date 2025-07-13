"""
preprocess module for data cleaning
"""

import re
import nltk

nltk.data.path.append("/tmp/nltk_data")
nltk.download("punkt", download_dir="/tmp/nltk_data", quiet=True)
from nltk.tokenize import sent_tokenize


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()
