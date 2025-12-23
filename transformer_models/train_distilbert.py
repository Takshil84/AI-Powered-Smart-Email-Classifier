import os
import json
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


## CONFIGURATION
DEVICE = torch.device("cpu")  # Force CPU for Windows stability
MODEL_NAME = "distilbert-base-uncased"
DATASET_PATH = r"C:\Users\Taksh\Desktop\Infosys\Datset\Classification_Dataset\cleaned_Dataset\merged_cleaned_dataset.csv"
OUTPUT_DIR = "./final_bert_model"

BATCH_SIZE = 8
EPOCHS = 3
MAX_LENGTH = 128
LEARNING_RATE = 5e-5
SEED = 42

LABELS = ["complaint", "promotion", "request", "social_media", "spam"]
LABEL_MAP = {label: idx for idx, label in enumerate(LABELS)}

print(f"Using device: {DEVICE}")


## REPRODUCIBILITY

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)


# DATASET CLASS

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["Message", "Category"])

    df["label"] = df["Category"].map(LABEL_MAP)

    # Stratified split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["Message"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"]
    )

    print("Tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    train_dataset = EmailDataset(train_encodings, train_labels)
    val_dataset = EmailDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS)
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_macro_f1 = 0.0

    print("Starting training...\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # Training
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE)
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        preds, true_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE)
                )
                predictions = torch.argmax(outputs.logits, dim=-1)
                preds.extend(predictions.cpu().numpy())
                true_labels.extend(batch["labels"].numpy())

        acc = accuracy_score(true_labels, preds)
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            true_labels, preds, average="weighted", zero_division=0
        )
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            true_labels, preds, average="macro", zero_division=0
        )

        print(
            f"Validation | Acc: {acc:.4f} | "
            f"Weighted F1: {f1_w:.4f} | Macro F1: {f1_m:.4f}"
        )

        # Save Best Model
        if f1_m > best_macro_f1:
            best_macro_f1 = f1_m
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
                json.dump(LABEL_MAP, f)
            print("Saved best model.\n")

    print("Training complete.")

if __name__ == "__main__":
    main()
