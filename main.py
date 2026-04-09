import json
import time
import csv
import os
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
# from openai import AsyncOpenAI
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate

load_dotenv()

############# Configuration ############
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"



SYSTEM_PROMPT = "\n".join([
    "You are a political content classifier. Your task is to label the following text as either 'pro-iran', 'anti-iran' or 'neutral'.",
    "'pro-iran': The text expresses support, sympathy, defense, or positive framing toward Iran, its government, policies, or actions.",
    "'anti-iran': The text expresses opposition, criticism, hostility, or negative framing toward Iran, its government, policies, or actions.",
    "'neutral': The text is factual/balanced with no clear stance, or Iran is only mentioned incidentally.",
    "Respond with ONLY the label: pro-iran, anti-iran or neutral.",
    "",
    "Text: {text}"
])

USER_PROMPT_TEMPLATE = """Classify the following text:
 
\"\"\"{text}\"\"\"
 
Label:"""


MODELS = [
    "anthropic/claude-3-5-sonnet",
    "deepseek/deepseek-v3.2",
    "google/gemma-4-31b-it"
]

VALID_LABELS = {"pro-iran", "anti-iran", "neutral"}

def classify_text(text: str, session: requests.Session, MODEL, retries: int = 3, delay: float =5) -> dict:
    """Send a single text to OpenRouter and return the parsed label dict."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }

    for attempt in range(1, retries + 1):
        try:
            response = session.post(
                OPENROUTER_API_URL,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            label = response.json()["choices"][0]["message"]["content"].strip().lower()
 
            if label not in VALID_LABELS:
                label = "neutral"
            return label
 
        except (requests.RequestException, KeyError) as exc:
            if attempt == retries:
                return "error"
            time.sleep(delay)


def label_dataset(
    texts: list[str],
    output_path: str = "labeled_output.csv",
    progress: bool = True,
    model: str = None) -> list[dict]:

    results = []
 
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
 
    with requests.Session() as session:
        session.headers.update(headers)
 
        for i, text in enumerate(texts, start=1):
            if progress:
                print(f"[{i}/{len(texts)}] Classifying...", end=" ", flush=True)
 
            classification = classify_text(text, session, MODEL=model)
 
            row = {"id": i, "text": text, "label": classification}
            results.append(row)
 
            if progress:
                print(classification)
 
    # Save to CSV
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label"])
        writer.writeheader()
        writer.writerows(results)
 
    print(f"\n✅ Done. Results saved to: {out.resolve()}")
    labels = [r["label"] for r in results]
    return results, labels

df = pd.read_csv("Cleaned_Iran_War_Sentiment.csv", nrows=5)
texts = [row['final_text'] for _, row in df.iterrows()]

for model in MODELS:

    _, labels = label_dataset(texts=texts)

    df[model] = labels

print(df.head())







# response = requests.Session.post(
#                 "https://openrouter.ai/api/v1/chat/completions",
#                 json={
#         "model": "google/gemma-4-31b-it",
#         "messages": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": USER_PROMPT_TEMPLATE},
#         ],
#         "temperature": 0.0,
#         "max_tokens": 120,
#     },
#                 timeout=30,
#             )