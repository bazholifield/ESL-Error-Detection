import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CHECKPOINT_PATH

LABEL_LIST = ["OK", "ERR"]

def load_model(checkpoint=CHECKPOINT_PATH):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    model.eval()
    return tokenizer, model

def detect_errors(text: str, tokenizer=None, model=None):
    if tokenizer is None or model is None:
        tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    return LABEL_LIST[pred]
