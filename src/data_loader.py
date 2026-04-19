from datasets import load_dataset
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_DIR, WI_LOCNESS_DATASET

CSV_FILENAME = "wi_locness.csv"

def load_wi_locness(max_examples=10000):
    os.makedirs(RAW_DIR, exist_ok=True)
    csv_path = os.path.join(RAW_DIR, CSV_FILENAME)

    ds = load_dataset(WI_LOCNESS_DATASET, split="train")

    data = []
    for example in ds:
        data.append({
            "original_text": example["input"],
            "corrected_text": example["output"],
        })
        if len(data) >= max_examples:
            break

    if not data:
        raise RuntimeError("No examples found in W&I+LOCNESS dataset!")

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} examples to {csv_path}")
    return df


if __name__ == "__main__":
    load_wi_locness()
