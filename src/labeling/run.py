import pandas as pd
import logging
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.labeling.labeling import LabelingController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the labeling pipeline with checkpointing.")
    parser.add_argument("--input", default="data/processed/processed-data.csv", help="Input processed CSV")
    parser.add_argument("--output", default="data/labeled/labeled-data.csv", help="Output labeled CSV / Checkpoint")
    parser.add_argument("--limit", type=int, help="Limit number of rows for testing")
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return
        
    logging.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    if args.limit:
        logging.info(f"Limiting to first {args.limit} rows...")
        df = df.head(args.limit)
    
    controller = LabelingController()
    logging.info("Starting labeling pipeline...")
    labeled_df = controller.run_pipeline(df, checkpoint_path=output_path)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logging.info(f"Saving labeled data to {output_path}...")
    labeled_df.to_csv(output_path, index=False)
    
    logging.info("Labeling process complete.")
    kappa = labeled_df.attrs.get('fleiss_kappa', 0.0)
    logging.info(f"Final Fleiss' Kappa: {kappa:.4f}")

if __name__ == "__main__":
    main()
