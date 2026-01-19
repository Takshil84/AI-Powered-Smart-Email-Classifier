import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# 1. Configuration
DATASET_PATH = r"C:\Users\mahan\Desktop\Infosys\Datset\Urgency_Dataset\Cleaned_Dataset\merged_3class_urgency.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./final_urgency_model"
NUM_LABELS = 3

# 2. Data Loading & Preparation
def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    # Basic cleaning: drop NaNs, ensure string
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    
    # Check distribution
    print("Label Distribution:\n", df['label'].value_counts())
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42
    )
    return train_texts, val_texts, train_labels, val_labels

# 3. Tokenization
print("Initializing Tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize_function(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=128)

# 4. Dataset Class
class UrgencyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 5. Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# Main Execution Flow
if __name__ == "__main__":
    # Load
    train_texts, val_texts, train_labels, val_labels = load_data()
    
    # Tokenize
    print("Tokenizing data...")
    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)
    
    # Create Datasets
    train_dataset = UrgencyDataset(train_encodings, train_labels)
    val_dataset = UrgencyDataset(val_encodings, val_labels)
    
    # Model
    print("Loading Model...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    # Training Config
    try:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=4,  # Reduced batch size
            per_device_eval_batch_size=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=5,
            eval_strategy="epoch", # Updated from evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train
        print("Starting Training...")
        trainer.train()
        
        # Save Final Model
        print(f"Saving model to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print("Training Complete!")
    except Exception as e:
        print(f"CRITICAL ERROR DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()
