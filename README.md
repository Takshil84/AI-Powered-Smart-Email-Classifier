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
