import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import CHECKPOINT_PATH
from error_classifier import classify_errors
from article_checker import check_articles

THRESHOLD = 0.25  # low threshold = high recall; rule-based classifier handles false positives

def load_model(checkpoint=CHECKPOINT_PATH):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    # article checker runs first — catches common ESL article errors before the model
    article_errors = check_articles(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    prob_err = F.softmax(logits, dim=-1)[0, 1].item()

    if prob_err < THRESHOLD and not article_errors:
        return "No error detected", []

    error_types = article_errors + [e for e in classify_errors(text) if e not in article_errors]
    return "Error detected", error_types

def main():
    parser = argparse.ArgumentParser(description="ESL Error Detector")
    parser.add_argument("--text", type=str, help="Sentence to check")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    tokenizer, model = load_model(args.checkpoint)

    if args.text:
        verdict, errors = predict(args.text, tokenizer, model)
        print(f"Input:   {args.text}")
        print(f"Result:  {verdict}")
        if errors:
            print(f"Type(s): {', '.join(errors)}")
    else:
        print("ESL Error Detector — type a sentence to check, or 'quit' to exit.")
        while True:
            text = input("\n> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                verdict, errors = predict(text, tokenizer, model)
                print(f"Result:  {verdict}")
                if errors:
                    print(f"Type(s): {', '.join(errors)}")

if __name__ == "__main__":
    main()
