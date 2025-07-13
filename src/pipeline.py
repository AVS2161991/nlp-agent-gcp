"""
pipeline module for producing a pipeline.json that can be uploaded to VertexAI
"""

from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import Output, Dataset, Input


@dsl.component(
    base_image="us-central1-docker.pkg.dev/storied-destiny-272007/nlp-agent-repo/nlp-agent-image:latest"
)
def clean_component(text: str, cleaned_output: Output[Dataset]):
    from preprocess import clean_text

    cleaned = clean_text(text)
    with open(cleaned_output.path, "w") as f:
        f.write(cleaned)


@dsl.component(
    base_image="us-central1-docker.pkg.dev/storied-destiny-272007/nlp-agent-repo/nlp-agent-image:latest"
)
def extract_component(cleaned_text: Input[Dataset], extraction_output: Output[Dataset]):
    from extract import extract_entities_sentiment
    import json

    with open(cleaned_text.path, "r") as f:
        text = f.read()

    with open(extraction_output.path, "w") as f:
        json.dump(extract_entities_sentiment(text), f)


@dsl.component(
    base_image="us-central1-docker.pkg.dev/storied-destiny-272007/nlp-agent-repo/nlp-agent-image:latest"
)
def summarize_component(cleaned_text: Input[Dataset], summary_output: Output[Dataset]):
    from summarize import generate_summary_variants

    with open(cleaned_text.path, "r") as f:
        text = f.read()

    summaries = generate_summary_variants(text)
    with open(summary_output.path, "w") as f:
        f.write(summaries[0][1])


@dsl.pipeline(name="nlp-agent-pipeline")
def nlp_pipeline(text: str):
    clean_task = clean_component(text=text)
    summarize_task = summarize_component(
        cleaned_text=clean_task.outputs["cleaned_output"]
    )
    extract_task = extract_component(cleaned_text=clean_task.outputs["cleaned_output"])


compiler.Compiler().compile(pipeline_func=nlp_pipeline, package_path="pipeline.json")
