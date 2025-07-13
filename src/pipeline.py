"""
pipeline module for producing a pipeline.json that can be uploaded to VertexAI
"""

from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import Output, Dataset, Input


@dsl.component(
    base_image="us-central1-docker.pkg.dev/storied-destiny-272007/nlp-agent-repo/"
    "nlp-agent-image:latest"
)
def clean_component(text: str, cleaned_output: Output[Dataset]):
    """
    Vertex AI Pipeline component that performs basic text cleaning.

    Args:
        text (str): Raw input text.
        cleaned_output (Output[Dataset]): Output path to store the cleaned text.

    Returns:
        None. The cleaned text is written to the cleaned_output artifact.
    """
    from src.preprocess import clean_text

    cleaned = clean_text(text)
    with open(cleaned_output.path, "w") as f:
        f.write(cleaned)


@dsl.component(
    base_image="us-central1-docker.pkg.dev/storied-destiny-272007/nlp-agent-repo/"
    "nlp-agent-image:latest"
)
def extract_component(cleaned_text: Input[Dataset], extraction_output: Output[Dataset]):
    """
    Vertex AI Pipeline component that extracts entities and sentiment from cleaned text
    using the Google Cloud Natural Language API.

    Args:
        cleaned_text (Input[Dataset]): Path to the cleaned text input.
        extraction_output (Output[Dataset]): Output path to store the extracted insights (JSON).

    Returns:
        None. The results are written as JSON to the extraction_output artifact.
    """
    from src.extract import extract_entities_sentiment
    import json

    with open(cleaned_text.path, "r") as f:
        text = f.read()

    with open(extraction_output.path, "w") as f:
        json.dump(extract_entities_sentiment(text), f)


@dsl.component(
    base_image="us-central1-docker.pkg.dev/storied-destiny-272007/nlp-agent-repo/"
    "nlp-agent-image:latest"
)
def summarize_component(cleaned_text: Input[Dataset], summary_output: Output[Dataset]):
    """
    Vertex AI pipeline component that generates a summary from cleaned text.

    Args:
        cleaned_text (Input[Dataset]): The input artifact containing cleaned text.
        summary_output (Output[Dataset]): The artifact where the summary will be written.

    Returns:
        None: Writes summary text to the output path.
    """
    from src.summarize import generate_summary_variants

    with open(cleaned_text.path, "r") as f:
        text = f.read()

    summaries = generate_summary_variants(text)
    with open(summary_output.path, "w") as f:
        f.write(summaries[0][1])


@dsl.pipeline(name="nlp-agent-pipeline")
def nlp_pipeline(text: str):
    """
    Kubeflow pipeline to process unstructured text using GCP-native NLP tools.

    The pipeline consists of three main stages:
    1. Text cleaning using a custom preprocessing component.
    2. Parallel summarization using Vertex AI generative models.
    3. Parallel extraction of entities and sentiment using Cloud Natural Language API.

    Args:
        text (str): The raw input text to be processed.

    Workflow:
        clean_component → [summarize_component, extract_component]
    """
    clean_task = clean_component(text=text)
    summarize_task = summarize_component(cleaned_text=clean_task.outputs["cleaned_output"])
    extract_task = extract_component(cleaned_text=clean_task.outputs["cleaned_output"])


compiler.Compiler().compile(pipeline_func=nlp_pipeline, package_path="pipeline.json")
