from vertexai.preview.generative_models import GenerativeModel
import vertexai

vertexai.init(project="storied-destiny-272007", location="us-central1")
model = GenerativeModel("gemini-2.0-flash-001")

PROMPTS = [
    "Summarize this:\n{text}",
    "Please write a concise summary of the following document:\n{text}",
    "Extract the key points and main ideas from the following text:\n{text}",
    "Provide a short, human-readable summary of the document below:\n{text}",
    "Summarize the content in 3 sentences:\n{text}",
]


def generate_summary_variants(text, max_tokens=256):
    summaries = []
    for prompt_template in PROMPTS:
        prompt = prompt_template.format(text=text)
        response = model.generate_content(
            [prompt],
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
            },
        )
        summary_text = response.candidates[0].content.parts[0].text
        summaries.append((prompt_template, summary_text))
    return summaries
