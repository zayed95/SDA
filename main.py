import os
import time
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()


llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

template = "\n".join([
    "You are a political content classifier. Your task is to label the following text as either 'pro-iran', 'anti-iran' or 'neutral'.",
    "'pro-iran': The text expresses support, sympathy, defense, or positive framing toward Iran, its government, policies, or actions.",
    "'anti-iran': The text expresses opposition, criticism, hostility, or negative framing toward Iran, its government, policies, or actions.",
    "'neutral': The text is unbiased and does not express any opinion about the subject matter.",
    "Respond with ONLY the label: pro-iran, anti-iran or neutral.",
    "",
    "Text: {text}"
])

prompt = PromptTemplate.from_template(template)

chain = prompt | llm

# dummy_text = "In Minab, near the Strait of Hormuz, a girls school was bombed, leaving 165 children dead. Yet most Americans cannot name the town where it happened, much less the name of a single one of those girls…That is because modern war depends on a hierarchy of grief, on training people to treat foreign children as an abstraction, a number, an unfortunate ‘incident’ to be denied, obscured or investigated at leisure…the only fact that matters: Children were killed in their classroom, and they were killed with violence made possible, materially and politically, by American power."
# result = chain.invoke({"text": dummy_text})

def classify(text: str, retries: int = 3, delay: float = 0.2):

    for attempt in range(retries):
        try:
            result = chain.invoke({"text": text})

            if result not in ["anti-iran", "pro-iran", "neutral"]:
                return "unclear"
            return result
        
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = delay * (2 ** attempt)
                print(f"Too many requests: waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"Attempt {attempt+1} failed: {e}")
    
    return None

def classify_dataset(df: pd.DataFrame, batch_size: int = 10, delay: float = 0.5) -> pd.DataFrame:

    labels = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]

        posts = [row['final_text'] for _, row in batch.iterrows()]

        try:
            labels.extend([
                classify(text=post) for post in posts
            ])
        except Exception as e:
            print(f"Error in processing batch: {len(batch)}")
            labels.extend([None] * len(batch))
        
        if i + batch_size < len(df):
            time.sleep(delay)

    df['label'] = labels

    out = Path("labled_dataset.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label"])
        writer.writeheader()
        writer.writerows(results)
    return df

df = pd.read_csv("Cleaned_Iran_War_Sentiment.csv", nrows=200)

df = classify_dataset(df=df, batch_size=10)

print(df.head())