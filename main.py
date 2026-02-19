from src.model.sentence_classifier import train_sentence_classifier

if __name__ == "__main__":
    csv_path = "data/processed/wikiedits_clean.csv"
    train_sentence_classifier(csv_path)
