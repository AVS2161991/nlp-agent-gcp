"""
tests for extract.py module
"""

from unittest.mock import patch, MagicMock
from src.extract import extract_entities_sentiment


@patch("src.extract.language_v1.LanguageServiceClient")
def test_extract_entities_sentiment(mock_client_class):

    mock_client = MagicMock()
    mock_entity = MagicMock()
    mock_entity.name = "Python"
    mock_entity.type_ = 1

    mock_sentiment = MagicMock()
    mock_sentiment.score = 0.8
    mock_sentiment.magnitude = 1.2

    mock_client.analyze_entities.return_value.entities = [mock_entity]
    mock_client.analyze_sentiment.return_value.document_sentiment = mock_sentiment
    mock_client_class.return_value = mock_client

    text = "Python is an excellent programming language."
    result = extract_entities_sentiment(text)

    assert result["entities"] == [("Python", "PERSON")]
    assert result["sentiment"] == {"score": 0.8, "magnitude": 1.2}

    assert mock_client.analyze_entities.called
    assert mock_client.analyze_sentiment.called
