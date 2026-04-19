# ESL Error Detector

A tool that automatically detects errors in ESL (English as a Second Language) writing using a fine-tuned DistilBERT model. It classifies sentences as correct or erroneous (error categorization and feedback are planned for later).

I started this while teaching English in Spain to work on my NLP skills, learn about language correction, and hopefully build something useful for my students.

## How to use it

```bash
python demo.py --text "She go to school yesterday."
```

Or run it interactively:

```bash
python demo.py
```

## Setup

```bash
pip install -r requirements.txt
```

To train the model yourself, run `python main.py` or use the Colab notebook at `notebook/train_colab.ipynb` (recommended — GPU makes it much faster). The trained checkpoint is not included in the repo due to file size.

## How it works

1. Loads learner English sentence pairs (original → corrected) from the W&I+LOCNESS dataset
2. Fine-tunes DistilBERT to classify sentences as OK or ERR
3. Runs inference on new input via the trained checkpoint

## Dataset

**W&I+LOCNESS** — Write & Improve + LOCNESS corpus of learner English, annotated with corrections (~38k sentence pairs). 

## Model performance

Evaluated on a held-out 20% split after 3 epochs:

| Metric | Score |
|--------|-------|
| Accuracy | 64.4% |
| F1 | 0.627 |
| Precision | 0.658 |
| Recall | 0.600 |
