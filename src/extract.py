"""
module to extract entities and sentiment 
"""

from typing import Dict
from google.cloud import language_v1


def extract_entities_sentiment(text: str) -> Dict[str, object]:
    """
    Extract named entities and sentiment from a given text using Google Cloud Natural Language API.

    This function:
    - Identifies named entities and their types (e.g., PERSON, LOCATION, ORGANIZATION)
    - Analyzes overall sentiment (score and magnitude) of the input text

    Args:
        text (str): The raw or cleaned text content to analyze.

    Returns:
        Dict[str, object]: A dictionary with:
            - 'entities': A list of tuples (entity name, entity type name)
            - 'sentiment': A dictionary with keys:
                - 'score': Float between -1.0 (negative) and 1.0 (positive)
                - 'magnitude': Float representing overall emotional intensity
    """
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

    entities = client.analyze_entities(request={"document": document}).entities
    sentiment = client.analyze_sentiment(request={"document": document}).document_sentiment

    return {
        "entities": [(e.name, language_v1.Entity.Type(e.type_).name) for e in entities],
        "sentiment": {"score": sentiment.score, "magnitude": sentiment.magnitude},
    }
