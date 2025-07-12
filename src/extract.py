"""
module to extract entities and sentiment 
"""

from google.cloud import language_v1


def extract_entities_sentiment(text):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(
        content=text, type_=language_v1.Document.Type.PLAIN_TEXT
    )

    entities = client.analyze_entities(request={"document": document}).entities
    sentiment = client.analyze_sentiment(
        request={"document": document}
    ).document_sentiment

    return {
        "entities": [(e.name, language_v1.Entity.Type(e.type_).name) for e in entities],
        "sentiment": {"score": sentiment.score, "magnitude": sentiment.magnitude},
    }
