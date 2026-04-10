import os
import time
import logging
import pandas as pd
import requests
from typing import Optional

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1"

# Three annotator models (swap freely with any OpenRouter model IDs)
MODELS = {
    "model_1": "meta-llama/llama-3.3-70b-instruct",
    "model_2": "deepseek/deepseek-v3.2",
    "model_3": "mistralai/mistral-nemo",
}

# Rate-limit safety settings
BATCH_SIZE          = 5     # texts per batch
DELAY_BETWEEN_CALLS = 1.5   # seconds between individual API calls
DELAY_BETWEEN_BATCHES = 10  # seconds between batches
MAX_RETRIES         = 3     # retries on transient errors
RETRY_BACKOFF       = 5     # extra seconds added per retry

INPUT_CSV   = None           # e.g. "my_data.csv"
OUTPUT_CSV  = "iran_sentiment_annotations.csv"

# ──────────────────────────────────────────────
# Loading Data
# ──────────────────────────────────────────────

df = pd.read_csv("Cleaned_Iran_War_Sentiment.csv", nrows=200)
texts = [row['final_text'] for _, row in df.iterrows()]

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise text classifier. Your task is to label text snippets
about Iran according to their sentiment or stance toward Iran and its government.

Labels:
  pro-iran   – The text is favorable, supportive, or sympathetic toward Iran,
                its government, or its policies.
  anti-iran  – The text is critical, hostile, or negative toward Iran,
                its government, or its policies.
  neutral    – The text is factual, balanced, or does not express a clear stance.

Rules:
  • Respond with ONLY one of the three labels: pro-iran, anti-iran, or neutral.
  • Do NOT include any explanation, punctuation, or extra words.
  • If the text is ambiguous, prefer neutral."""

def build_user_prompt(text: str) -> str:
    return f'Classify the following text:\n\n"""\n{text}\n"""'

# ---------------------------------------------------------------------------
# HEADERS
# ---------------------------------------------------------------------------
 
def _headers() -> dict:
    """
    NOTE: HTTP-Referer is intentionally omitted.
    Some CDN / proxy rules treat an unrecognised Referer as a browser
    request and serve an HTML redirect page instead of the API response.
    X-Title is enough to identify your app in the OpenRouter dashboard.
    """
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "X-Title":        "Iran Sentiment Annotator",
    }
 
 
# ---------------------------------------------------------------------------
# PREFLIGHT CHECK
# ---------------------------------------------------------------------------
 
def verify_connection() -> None:
    """
    Calls GET /api/v1/models before the annotation loop starts.
    Catches bad keys, network errors, and wrong model IDs early.
    Raises RuntimeError on any problem.
    """
    url = f"{OPENROUTER_URL}/models"
    log.info("Verifying OpenRouter connection (%s) ...", url)
 
    try:
        r = requests.get(url, headers=_headers(), timeout=15)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not reach OpenRouter: {exc}") from exc
 
    # HTML page = auth failure / redirect to login page
    if r.text.strip().startswith("<!DOCTYPE") or r.text.strip().startswith("<html"):
        raise RuntimeError(
            "OpenRouter returned an HTML page instead of JSON.\n"
            "This almost always means the API key is invalid or missing.\n"
            f"  Current key prefix: {repr(OPENROUTER_API_KEY[:16])}\n"
            "  Valid keys look like: sk-or-v1-...\n"
            "  Generate a new key at: https://openrouter.ai/keys"
        )
 
    if r.status_code == 401:
        raise RuntimeError(
            "OpenRouter returned 401 Unauthorized.\n"
            f"  Current key prefix: {repr(OPENROUTER_API_KEY[:16])}\n"
            "  Generate a new key at: https://openrouter.ai/keys"
        )
 
    if not r.ok:
        raise RuntimeError(
            f"OpenRouter preflight failed: HTTP {r.status_code}\n{r.text[:300]}"
        )
 
    models = [m["id"] for m in r.json().get("data", [])]
    log.info("Connection OK. %d models available.", len(models))
 
    for key, model_id in MODELS.items():
        if model_id not in models:
            log.warning(
                "  [!] %s = '%s' not found in model list. "
                "Check the ID or your account's enabled models at openrouter.ai/models",
                key, model_id,
            )
        else:
            log.info("  [OK] %s = '%s'", key, model_id)
 
 
# ---------------------------------------------------------------------------
# API CALL
# ---------------------------------------------------------------------------
 
VALID_LABELS = {"pro-iran", "anti-iran", "neutral"}
 
 
def call_openrouter(text: str, model: str) -> Optional[str]:
    """
    Call OpenRouter chat completions and return a validated label string,
    or None if all retries are exhausted.
    """
    url = f"{OPENROUTER_URL}/chat/completions"
    payload = {
        "model":       model,
        "max_tokens":  10,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(text)},
        ],
    }
 
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url, headers=_headers(), json=payload, timeout=30
            )
 
            # -- Rate limited ------------------------------------------------
            if response.status_code == 429:
                wait = RETRY_BACKOFF * attempt
                log.warning(
                    "Rate limited (429). Waiting %ds before retry %d/%d ...",
                    wait, attempt, MAX_RETRIES,
                )
                time.sleep(wait)
                continue
 
            # -- Capture raw text before any parsing -------------------------
            raw_text = response.text.strip()
 
            # -- HTML page = auth / redirect error ---------------------------
            if raw_text.startswith("<!DOCTYPE") or raw_text.startswith("<html"):
                log.error(
                    "Received an HTML page instead of JSON (HTTP %d). "
                    "The API key is likely invalid. Aborting this call.",
                    response.status_code,
                )
                return None
 
            # -- Empty body --------------------------------------------------
            if not raw_text:
                log.warning(
                    "Empty response body (HTTP %d) on attempt %d/%d for model %s.",
                    response.status_code, attempt, MAX_RETRIES, model,
                )
                time.sleep(RETRY_BACKOFF * attempt)
                continue
 
            # -- Non-2xx (not 429) -------------------------------------------
            if not response.ok:
                log.warning(
                    "HTTP %d on attempt %d/%d for model %s -- body: %.300s",
                    response.status_code, attempt, MAX_RETRIES, model, raw_text,
                )
                if response.status_code < 500:
                    return None  # 4xx won't fix on retry
                time.sleep(RETRY_BACKOFF * attempt)
                continue
 
            # -- Parse JSON --------------------------------------------------
            try:
                data = response.json()
            except ValueError:
                log.warning(
                    "Invalid JSON on attempt %d/%d for model %s -- raw: %.300s",
                    attempt, MAX_RETRIES, model, raw_text,
                )
                time.sleep(RETRY_BACKOFF * attempt)
                continue
 
            # -- Extract content ---------------------------------------------
            try:
                raw_label = data["choices"][0]["message"]["content"].strip().lower()
            except (KeyError, IndexError, TypeError) as exc:
                log.warning(
                    "Unexpected response structure on attempt %d/%d for model %s: %s",
                    attempt, MAX_RETRIES, model, exc,
                )
                time.sleep(RETRY_BACKOFF * attempt)
                continue
 
            # -- Normalise ---------------------------------------------------
            label_map = {
                "pro iran":  "pro-iran",
                "anti iran": "anti-iran",
            }
            label = label_map.get(raw_label, raw_label)
 
            if label not in VALID_LABELS:
                log.warning(
                    "Unexpected label '%s' from %s -- marking as 'neutral'",
                    raw_label, model,
                )
                label = "neutral"
 
            return label
 
        except requests.exceptions.Timeout:
            log.warning(
                "Timeout on attempt %d/%d for model %s", attempt, MAX_RETRIES, model
            )
        except requests.exceptions.ConnectionError as exc:
            log.warning(
                "Connection error on attempt %d/%d for model %s: %s",
                attempt, MAX_RETRIES, model, exc,
            )
        except requests.exceptions.RequestException as exc:
            log.warning(
                "Request error on attempt %d/%d for model %s: %s",
                attempt, MAX_RETRIES, model, exc,
            )
 
        time.sleep(RETRY_BACKOFF * attempt)
 
    log.error("All %d retries exhausted for model %s.", MAX_RETRIES, model)
    return None
 
 
# ---------------------------------------------------------------------------
# BATCH ANNOTATION
# ---------------------------------------------------------------------------
 
def annotate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates every row in df[TEXT_COLUMN] with all three models.
    Returns a new DataFrame with columns:
        text, label_model_1, label_model_2, label_model_3
    """
    results = {
        "text":          [],
        "label_model_1": [],
        "label_model_2": [],
        "label_model_3": [],
    }
 
    total     = len(texts)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
 
    log.info(
        "Starting annotation: %d texts | %d batches | batch_size=%d",
        total, n_batches, BATCH_SIZE,
    )
 
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, total)
        batch = texts[start:end]
 
        log.info("-- Batch %d/%d (rows %d-%d) --", batch_idx + 1, n_batches, start, end - 1)
 
        for i, text in enumerate(batch):
            global_idx = start + i
            log.info("  [%d/%d] %.80s ...", global_idx + 1, total, text)
 
            labels = {}
            for model_key, model_id in MODELS.items():
                label = call_openrouter(text, model_id)
                labels[model_key] = label or "error"
                log.info(
                    "    %-10s (%s) -> %s",
                    model_key, model_id.split("/")[-1], labels[model_key],
                )
                time.sleep(DELAY_BETWEEN_CALLS)
 
            results["text"].append(text)
            results["label_model_1"].append(labels["model_1"])
            results["label_model_2"].append(labels["model_2"])
            results["label_model_3"].append(labels["model_3"])
 
        if batch_idx < n_batches - 1:
            log.info("  Batch done. Sleeping %ds before next batch ...", DELAY_BETWEEN_BATCHES)
            time.sleep(DELAY_BETWEEN_BATCHES)
 
    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():


    # ── Annotate ───────────────────────────────
    df_out = annotate_dataframe(df)

    # ── Save ───────────────────────────────────
    df_out.to_csv(OUTPUT_CSV, index=False)
    log.info("✓ Results saved to %s", OUTPUT_CSV)

    return df_out


if __name__ == "__main__":
    main()