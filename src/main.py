import argparse, logging
import pandas as pd
from preprocessing import Preprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(name="__name__")

def get_args():
    parser = argparse.ArgumentParser(description="NLP Preprocessing Pipeline")
        
    # Input settings
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--column_name", default="content", help="Column to clean")    
    # Toggles (Booleans)
    parser.add_argument("--translate", action="store_true", help="Translate non-English text")
    parser.add_argument("--no_urls", action="store_true")
    parser.add_argument("--no_emoji", action="store_true")
    parser.add_argument("--do_spelling", action="store_true")
    parser.add_argument("--no_punctuation", action="store_true")
    parser.add_argument("--no_stopwords", action="store_true")
    parser.add_argument("--do_lemma", action="store_true")
    
    return parser.parse_args()

def run_pipeline(args) -> pd.DataFrame:
        
        pipeline = Preprocessing(args=args)
        df = pd.read_csv(args.input)
        logger.info(f"Starting pipeline on column: {args.column_name}...")
        
        df['processed_text'] = df[args.column_name].apply(pipeline.process_text)
        
        df.to_csv(args.output, index=False, encoding='utf-8')

        logger.info("Pipeline complete!")
        return df

args = get_args()

run_pipeline(args=args)



