# nlp-agent-gcp

## Overview
This project demonstrates a GCP-native NLP prototype that extracts entities, sentiment, and
summarizations from unstructured documents. It includes an agentic workflow that uses these tools
to fulfill high-level user queries using modular reasoning across multiple documents.

## Tech Stack and Services
- Vertex AI: For generative summarization using Gemini models.
- Cloud Natural Language API: For entity extraction and sentiment analysis.
- Cloud Storage: To store and retrieve input text files.
- Vertex AI Pipelines: To orchestrate the summarization and extraction processes.
- Artifact Registry: To host custom Docker images for component execution.

## To Reproduce and Run the Project
1. Clone the Repository
```
cd tek_sys
```

2. Install Dependencies
```
cd src/
pip install -r requirements.txt
```

3. Set Up Google Cloud Project
- Enable APIs: Vertex AI, Cloud Storage, Natural Language API, Artifact Registry
- Create service account and assign roles: Vertex AI User, Storage Viewer, Artifact Registry Writer
- Download the service account JSON key

4. Authenticate
```
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your-key.json'
```
or
create a .env file and add the GOOGLE_APPLICATION_CREDENTIALS in it
```
set -a && source .env && set +a
```
5. Upload Text Files
- Upload raw .txt documents to a GCS bucket under docs/ prefix.

6. Run the Agent
```
python agent.py
```
This downloads docs, summarizes them, extracts sentiment/entities, and returns insights.

7. Run Vertex AI Pipeline (Optional)
- Build and push Docker image:
docker build -t us-central1-docker.pkg.dev/PROJECT_ID/nlp-agent-repo/nlp-agent-image .
docker push us-central1-docker.pkg.dev/PROJECT_ID/nlp-agent-repo/nlp-agent-image
- Update pipeline.py to use your Docker image
- Compile and submit the pipeline from Console or CLI

## Why These GCP Services?
- Vertex AI: Provides scalable, state-of-the-art generative models (Gemini, PaLM) with managed
infrastructure.
- Cloud Natural Language API: Fast and reliable for standard entity/sentiment analysis without
training.
- Cloud Storage: Durable, scalable storage for document input.
- Artifact Registry: Modern container image storage to replace deprecated GCR.
- Pipelines: Helps manage multi-step AI workflows with better traceability and reuse.

## Project Structure
- agent.py: Full agentic execution across all modules.
- summarize.py: Gemini summarization logic.
- extract.py: Entity and sentiment extraction via Cloud NLP.
- load_data.py: Downloads files from GCS.
- pipeline.py: Defines KFP pipeline using custom Docker image.
- eda.py: Analyzes word and sentence counts.
- evaluate.py: Compares summaries to references with ROUGE.
