# AI-Powered-Smart-Email-Classifier
AI Powered Smart Email Classifier for Enterprises
## Milestone 1: Data Collection & Preprocessing

---

## Overview

This project builds an AI-powered email classification system for enterprises. Milestone 1 focuses on collecting and cleaning email datasets for training machine learning models.

---

## What I Did

### 1. Collected Email Datasets
- **Classification Dataset**: Collected emails for 5 different categories
- **Urgency Dataset**: Collected emails with urgency priority levels

### 2. Cleaned the Data
Removed noise and normalized text:
- Removed HTML tags
- Removed URLs and email addresses
- Removed email signatures
- Converted to lowercase
- Removed extra whitespace
- Removed duplicates and empty messages

### 3. Organized Datasets
- Created separate cleaning scripts for each category
- Merged all classification data into one dataset
- Split urgency data into train/test/validation sets

---


---

## Data Cleaning

Applied the following cleaning steps to all emails:

```
1. Remove HTML tags and entities
2. Remove URLs (http://, www.)
3. Remove email addresses
4. Remove email signatures
5. Remove phone numbers
6. Remove special characters
7. Lowercase all text
8. Normalize whitespace
9. Remove duplicates
10. Remove empty/very short messages
```

---

## Project Structure

```
Infosys/
├── Dataset/
│   ├── Classification_Dataset/
│   │   ├── Raw_Dataset/              # Original data files
│   │   ├── Cleaning code/            # Cleaning scripts
│   │   └── cleaned_Dataset/          # merged_cleaned_dataset.csv
│   │
│   └── Urgency_Dataset/
│       ├── Raw_Dataset/              # train.csv, test.csv, validation.csv
│       ├── Cleaned_Dataset/          # Cleaned versions
│       └── data_cleaning.py          # Cleaning script
│
└── README.md
```


## Usage

**Classification Dataset Cleaning:**
```bash
cd Datset/Classification_Dataset/Cleaning\ code/
python clean_complaint.py
python clean_request.py
python clean_promotion.py
python clean_social_media.py
python clean_spam.py
python merge_cleaned_datasets.py
```

**Urgency Dataset Cleaning:**
```bash
cd Datset/Urgency_Dataset/
python data_cleaning.py
```

---

## Results

✅ **Classification Dataset**: 3,315 cleaned emails across 5 categories  
✅ **Urgency Dataset**: 2,810+ cleaned emails with 4 urgency levels  
✅ **Data Quality**: No duplicates, no missing values, all text normalized  
✅ **Code**: Modular cleaning scripts for reproducibility

---  

## Milestone 2: Email Categorization Engine

---

## Overview

Milestone 2 focused on developing an NLP-based classification system to categorize emails into **Complaint**, **Request**, **Social Media**, **Spam**, and **Promotion**.

---

## What I Did

### 1. Baseline Classifiers
Implemented traditional machine learning models to establish a performance benchmark.
- **Models**: Logistic Regression, Multinomial Naive Bayes.
- **Preprocessing**: TF-IDF Vectorization (Top 5000 features).
- **Validation**:
    - **Stratified 5-Fold Cross-Validation**: Applied to ensure the model's performance is consistent across different subsets of data.
    - **Reason for Stratified CV**: Ensures each fold preserves the percentage of samples for each class, providing a statistically robust evaluation.

**Baseline Results:**
- **Logistic Regression**: ~98.0% Accuracy
- **Naive Bayes**: ~97.7% Accuracy

### 2. Transformer Fine-Tuning (DistilBERT)
Fine-tuned a pre-trained **DistilBERT** model for state-of-the-art performance.
- **Model**: `distilbert-base-uncased`
- **Method**: Fine-tuned using PyTorch with `AdamW` optimizer (Manual Loop for Windows stability).
- **Configuration**:
    - Epochs: 3
    - Batch Size: 8
    - Max Sequence Length: 128
    - Optimizer: AdamW (lr=5e-5)

**DistilBERT Results (Best Epoch):**
- **Accuracy**: **98.79%**
- **Weighted F1-Score**: **98.80%**
- **Macro F1-Score**: **98.85%**

DistilBERT outperformed the baselines, achieving nearly 99% performance across all metrics.

---

## Files Created (Milestone 2)

- `baseline_classification.py`: Script for training and evaluating Logistic Regression and Naive Bayes.
- `bert_finetuning.py`: Script for fine-tuning the DistilBERT model.
- `final_bert_model/`: Directory containing the saved fine-tuned model and tokenizer.

---

## Usage (Milestone 2)

**Run Baseline Models:**
```bash
python baseline_classification.py
```

**Run DistilBERT Training:**
```bash
python bert_finetuning.py
```




## Milestone 3: Urgency Detection & Scoring

---

## Overview

Milestone 3 focused on implementing a **Hybrid Urgency Detection System** to classify emails into priority levels (Low, Medium, High). The system prioritizes "Safety" using strict rules while leveraging an ML model for general context, ensuring critical issues are never missed.

## What I Did

### 1. Hybrid Architecture Implementation
Developed a "Confidence-Aware" Hybrid System:
- **Rule-Based Engine (`rule_based_urgency.py`)**: Detects high-risk keywords (e.g., "system down", "security breach") to force **High Urgency**.
- **ML Model (`urgency_model_training.py`)**: Fine-tuned **DistilBERT** on a 3-class dataset to handle general context.
- **Inference Logic (`hybrid_inference.py`)**:
    1.  **Critical Override**: Rule matches -> High.
    2.  **High Confidence ML**: Model > 85% confidence -> Trust Model.
    3.  **Safety Fallback**: Model uncertain & Medium keyword match -> Medium.

### 2. Model Training
- **Algorithm**: DistilBERT (`distilbert-base-uncased`)
- **Dataset**: `merged_3class_urgency.csv` (Mapped: 0=Low, 1=Medium, 2=High)
- **Results**:
    - **Accuracy**: **92.31%**
    - **F1 Score (Weighted)**: **92.31%**
    - **Validation Loss**: **0.32**

### 3. Verification
Validated specific test cases:
- "System is down" -> **High** (Rule Override) ✅
- "Newsletter subscription" -> **Low** (ML Prediction) ✅
- "Help with issue" -> **Medium** (Rule Fallback) ✅

## Files Created (Milestone 3)
- `urgency_model_training.py`: Script to fine-tune DistilBERT for urgency.
- `rule_based_urgency.py`: Module defining critical/medium regex rules.
- `hybrid_inference.py`: Production-ready inference script combining ML + Rules.
- `final_urgency_model/`: Directory containing the saved model.

## Usage (Milestone 3)

**Train the Model:**
```bash
python urgency_model_training.py
```

**Run Hybrid Inference:**
```bash
python hybrid_inference.py
```

---
