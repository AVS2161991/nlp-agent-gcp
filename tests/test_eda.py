"""
tests for eda.py module
"""

import pytest
from collections import Counter
from unittest import mock
from src.eda import run_eda


@pytest.fixture
def fake_docs(tmp_path):
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"
    doc1.write_text("Hello world! This is a test.")
    doc2.write_text("Another test document here.")
    return tmp_path


@mock.patch("src.preprocess.clean_text")
def test_word_counts(mock_clean_text, fake_docs):

    mock_clean_text.side_effect = [
        "hello world this is a test",
        "another test document here",
        "hello world this is a test",
        "another test document here",
    ]

    summary_df, word_freq = run_eda(fake_docs)

    assert summary_df.shape[0] == 2
    assert set(summary_df["file"]) == {"doc1.txt", "doc2.txt"}
    assert set(summary_df["word_count"]) == {6, 4}

    expected = Counter(
        {
            "another": 1,
            "test": 1,
            "document": 1,
            "here.": 1,
            "hello": 1,
            "world!": 1,
            "this": 1,
            "is": 1,
            "a": 1,
            "test.": 1,
        }
    )
    assert word_freq == expected
    assert word_freq["test"] == 1
    assert word_freq["hello"] == 1
    assert word_freq["another"] == 1


@mock.patch("src.preprocess.clean_text")
def test_empty_directory(mock_clean_text, tmp_path):
    summary_df, word_freq = run_eda(tmp_path)

    assert summary_df.empty
    assert word_freq == Counter()
