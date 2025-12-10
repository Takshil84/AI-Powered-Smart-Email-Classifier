import pandas as pd
import re


# 1. Load Dataset

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['Category', 'Message']]


# 2. Keep Only Spam Rows

df['Category'] = df['Category'].str.strip().str.lower()

df = df[df['Category'] == 'spam']   # KEEP ONLY SPAM


# 3. Drop empty or duplicate messages

df = df.dropna(subset=['Message'])
df = df.drop_duplicates(subset=['Message'])


# 4. Cleaning Function

def clean_text(text):
    text = text.lower()                                            # lowercase
    text = re.sub(r"\S+@\S+\.\S+", "", text)                       # remove email addresses
    text = re.sub(r"http\S+|www\S+", "", text)                     # remove URLs
    text = re.sub(r"<.*?>", "", text)                              # remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)                    # remove punctuation/special chars
    text = re.sub(r"\s+", " ", text).strip()                       # remove extra spaces
    return text


# 5. Apply Cleaning

df['Message'] = df['Message'].astype(str).apply(clean_text)


# 6. Save Cleaned Spam File

df.to_csv("spam_cleaned.csv", index=False)

print("Done! Cleaned spam-only dataset shape:", df.shape)
df.head()
