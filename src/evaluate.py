"""
summarization evaluation using ROUGE
"""

from typing import Dict
from rouge_score import rouge_scorer, scoring


def evaluate_summary(reference: str, generated: str) -> Dict[str, scoring.Score]:
    """
    Evaluate the quality of a generated summary against a reference summary using ROUGE scores.

    This function uses the `rouge_scorer.RougeScorer` from the `rouge_score` library
    to compute ROUGE-1, ROUGE-2, and ROUGE-L scores between the reference and generated texts.

    Args:
        reference (str): The reference or ground truth summary.
        generated (str): The generated summary to be evaluated.

    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores,
              each as a `Score` object with precision, recall, and fmeasure attributes.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores
