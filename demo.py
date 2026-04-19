import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import CHECKPOINT_PATH

LABEL_LIST = ["OK", "ERR"]

def load_model(checkpoint=CHECKPOINT_PATH):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    return LABEL_LIST[pred]

def main():
    parser = argparse.ArgumentParser(description="ESL Error Detector")
    parser.add_argument("--text", type=str, help="Sentence to check")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    tokenizer, model = load_model(args.checkpoint)

    if args.text:
        result = predict(args.text, tokenizer, model)
        print(f"Input:      {args.text}")
        print(f"Prediction: {result}")
    else:
        # interactive mode
        print("ESL Error Detector — type a sentence to check, or 'quit' to exit.")
        while True:
            text = input("\n> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                print(f"Prediction: {predict(text, tokenizer, model)}")

if __name__ == "__main__":
    main()
