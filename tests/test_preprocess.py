"""
tests for preprocess.py module
"""

from src.preprocess import clean_text


def test_clean_text_removes_extra_whitespace():
    text = "This   is \n a test.\tAnother   one."
    cleaned = clean_text(text)
    assert cleaned == "This is a test. Another one."


def test_clean_text_strips_edges():
    text = "   Leading and trailing whitespace   "
    cleaned = clean_text(text)
    assert cleaned == "Leading and trailing whitespace"


def test_clean_text_empty_string():
    text = "   \n\t  "
    cleaned = clean_text(text)
    assert cleaned == ""
