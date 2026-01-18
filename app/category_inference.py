import torch
import json
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import sys

# Configuration
MODEL_PATH = "./final_bert_model"
CONFIDENCE_THRESHOLD = 0.6  # Adjust as needed

class CategoryClassifier:
    def __init__(self):
        print("Loading Category Classifier...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for Category: {self.device}")
        
        self.model = None
        self.tokenizer = None
        self.label_map = {}

        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Category model path {MODEL_PATH} does not exist.")
            return

        try:
            # Load Label Map
            with open(os.path.join(MODEL_PATH, "label_map.json"), "r") as f:
                self.label_map = {v: k for k, v in json.load(f).items()} # Invert map: idx -> label
            
            # Load Model & Tokenizer
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
            self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
            self.model.to(self.device)
            self.model.eval()
            print("Category ML Model loaded successfully.")
        except Exception as e:
            print(f"Error loading Category ML model: {e}")
            self.model = None

    def predict(self, text):
        if not self.model:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "source": "ML (Model Not Loaded)"
            }

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        conf, pred_idx = torch.max(probabilities, dim=1)
        
        label_str = self.label_map.get(pred_idx.item(), "Unknown")
        
        return {
            "label": label_str,
            "confidence": conf.item(),
            "source": "ML"
        }

if __name__ == "__main__":
    classifier = CategoryClassifier()
    test_text = "I want to return my order."
    print(f"Test Input: {test_text}")
    print(f"Result: {classifier.predict(test_text)}")
