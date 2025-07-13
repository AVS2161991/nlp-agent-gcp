"""
preprocess module for data cleaning
"""

import re
import nltk

nltk.data.path.append("/tmp/nltk_data")
nltk.download("punkt", download_dir="/tmp/nltk_data", quiet=True)


def clean_text(text: str) -> str:
    """
    Clean raw text by removing extra whitespace and newlines.

    This function:
    - Replaces multiple whitespace characters (spaces, tabs, newlines) with a single space.
    - Strips leading and trailing whitespace.

    Args:
        text (str): The raw text input.

    Returns:
        str: The cleaned, whitespace-normalized text.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()
