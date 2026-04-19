from src.data_loader import load_wi_locness
from src.build_dataset import build_dataset
from src.model.sentence_classifier import train_sentence_classifier
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.config import RAW_DIR, PROCESSED_DIR

if __name__ == "__main__":
    # Step 1: download dataset
    load_wi_locness()

    # Step 2: build labeled classification dataset
    raw_path = os.path.join(RAW_DIR, "wi_locness.csv")
    processed_path = os.path.join(PROCESSED_DIR, "classification_dataset.csv")
    build_dataset(raw_path, processed_path)

    # Step 3: train
    train_sentence_classifier(processed_path)
