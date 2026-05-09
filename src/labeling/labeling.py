import os, time
import requests
import pandas as pd
import logging
from enum import Enum
from collections import Counter
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(name=__name__)

class LLM(Enum):
    
    GPT4o_MINI = "openai/gpt-4o-mini"
    GEMINI_FLASH = "google/gemini-2.5-flash"
    CLAUDE_HAIKU = "anthropic/claude-3.5-haiku"

class Model:
    def __init__(self, llm: LLM):
        self.llm = llm
        pass
    
    def validate_response(self, response) -> str:

        data = response.json()
        raw_label = data["choices"][0]["message"]["content"].strip().lower()
        if raw_label not in ["positive", "negative", "neutral"]:
            return "ERROR"
        return raw_label
        
    def call(self, text: str):

        openrouter_url = os.getenv("OPENROUTER_URL")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        url = f"{openrouter_url}/chat/completions"

        payload = {
            "model": self.llm.value,
            "max_tokens": 10,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": """You are a precise text classifier. Your task is to label text snippets
                according to their overall sentiment.

                Labels:
                positive  – The text expresses a favorable, optimistic, supportive, or hopeful tone.
                negative  – The text expresses a critical, hostile, pessimistic, angry, or fearful tone.
                neutral   – The text is factual, balanced, or does not express a clear sentiment.

                Rules:
                • Respond with ONLY one of the three labels: positive, negative, or neutral.
                • Do NOT include any explanation, punctuation, or extra words.
                • If the text is ambiguous, prefer neutral."""},
                {"role": "user", "content": f'Classify the overall sentiment of the following text:\n\n"""\n{text}\n"""'}
            ]
        }

        headers = {

        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type":  "application/json",
        "X-Title":        "Sentiment Ground Truth Annotator",
        }

        for attempt in range(1, 4):
            try:
                response = requests.post(
                    url=url,
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                if response.status_code == 429:
                    wait = 3 * attempt
                    logger.warning(
                        "Rate limited (429). Waiting %ds before retry %d/3 ...",
                        wait, attempt, 
                    )
                    time.sleep(wait)
                    continue

                return self.validate_response(response)
            
            except Exception as e:
                logger.warning(f"Error calling model {self.llm.name}: {e}")
                if "choices" in str(e):
                    try:
                        logger.warning(f"Response content: {response.text}")
                    except:
                        pass

        
class LabelingController:
    def __init__(self):
        self.models = [Model(llm=llm) for llm in LLM]

    def calculate_fleiss_kappa(self, labels_matrix: list) -> float:
        """
        Calculates Fleiss' Kappa for inter-rater reliability among the models.
        
        :param labels_matrix: list of lists, where each inner list contains labels from models for one text.
        :return: Fleiss' Kappa score.
        """
        N = len(labels_matrix)
        if N == 0:
            return 0.0
        n = len(labels_matrix[0]) # Number of raters (models)
        if n < 2:
            return 1.0

        # Identify all unique labels used
        all_labels = set()
        for row in labels_matrix:
            all_labels.update(row)
        
        categories = sorted(list(all_labels))
        k = len(categories)

        # Create the M matrix: M[i][j] is the number of raters who assigned item i to category j
        M = []
        for row in labels_matrix:
            counts = [row.count(cat) for cat in categories]
            M.append(counts)

        # pj: proportion of all assignments which were to the j-th category
        pj = [0.0] * k
        for j in range(k):
            pj[j] = sum(M[i][j] for i in range(N)) / (N * n)
        
        Pe = sum(p**2 for p in pj)

        # Pi: extent to which raters agree for the i-th item
        Pi = [0.0] * N
        for i in range(N):
            # Sum of squares of counts - n, divided by n(n-1)
            Pi[i] = (sum(M[i][j]**2 for j in range(k)) - n) / (n * (n - 1))
        
        P = sum(Pi) / N
        
        if Pe == 1:
            return 1.0
            
        kappa = (P - Pe) / (1 - Pe)
        return kappa

    def run_pipeline(self, df: pd.DataFrame, checkpoint_path: str = None) -> pd.DataFrame:
        """
        Runs the labeling pipeline by sending each text to 3 different LLMs.
        Settles on a final label via majority voting and calculates Fleiss' Kappa.
        """
        all_results = []
        agreement_matrix = []

        start_row = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint_df = pd.read_csv(checkpoint_path)
                if not checkpoint_df.empty:
                    start_row = len(checkpoint_df)
                    logger.info(f"Checkpoint found. Resuming from row {start_row}")
                    
                    model_names = [m.llm.name for m in self.models]
                    if all(col in checkpoint_df.columns for col in model_names + ['label']):
                        all_results = checkpoint_df[model_names + ['label']].to_dict('records')
                        agreement_matrix = checkpoint_df[model_names].values.tolist()
                    else:
                        logger.warning("Checkpoint columns mismatch. Restarting.")
                        start_row = 0
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
                start_row = 0

        logger.info("Starting multi-model labeling pipeline...")

        for idx in range(start_row, len(df), 5):
            end = min(idx + 5, len(df))
            batch = df.iloc[idx : end]
            
            for _, row in batch.iterrows():
                text = row['processed_text']
                model_labels = {}
                
                # Call all 3 models synchronously
                for model in self.models:
                    label = model.call(text=text)
                    model_labels[model.llm.name] = label
                
                labels_list = list(model_labels.values())
                agreement_matrix.append(labels_list)
                
                # Voting system: Settle on one label
                counts = Counter(labels_list)
                most_common_label, count = counts.most_common(1)[0]
                
                if count >= 2:
                    # Majority reached
                    final_label = most_common_label
                else:
                    # No majority (3-way tie): Settle using GPT4o_MINI as primary or default to first
                    final_label = model_labels.get(LLM.GPT4o_MINI.name, labels_list[0])
                
                model_labels['label'] = final_label
                all_results.append(model_labels)

            if checkpoint_path:
                temp_results_df = pd.DataFrame(all_results)
                progress_df = pd.concat([df.iloc[:len(all_results)].reset_index(drop=True), temp_results_df], axis=1)
                progress_df.to_csv(checkpoint_path, index=False)
                logger.info(f"Checkpoint saved to {checkpoint_path}")

            logger.info(f"Processed batch {idx//5 + 1}/{(len(df)-1)//5 + 1}")
            time.sleep(10) # Delay to respect rate limits

        # Combine results with original dataframe
        results_df = pd.DataFrame(all_results)
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Calculate and log inter-rater reliability
        kappa = self.calculate_fleiss_kappa(agreement_matrix)
        logger.info(f"Fleiss' Kappa (Inter-model agreement): {kappa:.4f}")
        
        # Store kappa in dataframe attributes for reference
        output_df.attrs['fleiss_kappa'] = kappa
        
        return output_df