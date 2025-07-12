"""
pipeline module for producing a pipeline.json that can be uploaded to VertexAI
"""

from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import Output, Dataset


@dsl.component(
    base_image="us-central1-docker.pkg.dev/storied-destiny-272007/nlp-agent-repo/nlp-agent-image:latest"
)
def extract_component(text: str, extraction_output: Output[Dataset]):
    from extract import extract_entities_sentiment
    import json

    with open(extraction_output, "w") as f:
        json.dump(extract_entities_sentiment(text), f)


@dsl.component(
    base_image="us-central1-docker.pkg.dev/storied-destiny-272007/nlp-agent-repo/nlp-agent-image:latest"
)
def summarize_component(text: str, summary_output: Output[Dataset]):
    from summarize import generate_summary_variants

    summaries = generate_summary_variants(text)
    with open(summary_output, "w") as f:
        f.write(summaries[0][1])


@dsl.pipeline(name="nlp-agent-pipeline")
def nlp_pipeline(text: str):
    summarize_task = summarize_component(text=text)
    extract_task = extract_component(text=text)


compiler.Compiler().compile(pipeline_func=nlp_pipeline, package_path="pipeline.json")
