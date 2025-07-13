"""
tests for evaluate.py module
"""

import pytest
from src.evaluate import evaluate_summary


def test_evaluate_summary_typical_case():
    reference = "The cat sat on the mat."
    generated = "The cat is sitting on the mat."

    scores = evaluate_summary(reference, generated)

    assert isinstance(scores, dict)
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores

    for metric in scores.values():
        assert 0.0 <= metric.precision <= 1.0
        assert 0.0 <= metric.recall <= 1.0
        assert 0.0 <= metric.fmeasure <= 1.0


def test_evaluate_summary_identical_text():
    reference = "The quick brown fox jumps over the lazy dog."
    generated = "The quick brown fox jumps over the lazy dog."

    scores = evaluate_summary(reference, generated)

    for metric in scores.values():
        assert metric.fmeasure == pytest.approx(1.0, rel=1e-3)


def test_evaluate_summary_empty_generated():
    reference = "The quick brown fox."
    generated = ""

    scores = evaluate_summary(reference, generated)

    for metric in scores.values():
        assert metric.precision == 0.0
        assert metric.recall == 0.0
        assert metric.fmeasure == 0.0


def test_evaluate_summary_empty_reference():
    reference = ""
    generated = "The quick brown fox."

    scores = evaluate_summary(reference, generated)

    for metric in scores.values():
        assert metric.precision == 0.0
        assert metric.recall == 0.0
        assert metric.fmeasure == 0.0
