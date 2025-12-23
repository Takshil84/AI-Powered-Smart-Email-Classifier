import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

# Path to the dataset
dataset_path = r'C:\Users\Taksh\Desktop\Dataset\Classification_Dataset\cleaned_Dataset\merged_cleaned_dataset.csv'

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters and numbers (keeping only alphabet for simpler baseline)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    
    # Stopword removal
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
    except Exception:
        pass
    
    # Lemmatization
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    except Exception:
        pass
    
    return " ".join(tokens)

def main():
    results = []
    
    def log(msg):
        print(msg)
        results.append(msg)

    log("Loading dataset...")
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding='latin1')
    
    # Drop rows with missing values in Message or Category
    df = df.dropna(subset=['Message', 'Category'])
    
    log(f"Dataset loaded. Total rows: {len(df)}")
    
    log("Preprocessing messages (Tokenization, Stopword removal, Lemmatization)...")
    df['Processed_Message'] = df['Message'].apply(preprocess_text)
    
    log("Splitting data into training and testing sets...")
    X = df['Processed_Message']
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    log("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # --- Logistic Regression ---
    log("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    
    log("\nLogistic Regression Results:")
    log(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    log("Classification Report:")
    log(classification_report(y_test, y_pred_lr))
    
    # --- Naive Bayes ---
    log("\nTraining Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)
    
    log("\nNaive Bayes Results:")
    log(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    log("Classification Report:")
    log(classification_report(y_test, y_pred_nb))

    with open('baseline_results.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print("\nResults saved to baseline_results.txt")

if __name__ == "__main__":
    main()
