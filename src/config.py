import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "results", "checkpoint-3000")

WI_LOCNESS_DATASET = "512duncanl/wi_locness_detokenized"
