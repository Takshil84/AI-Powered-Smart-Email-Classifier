import pandas as pd
import re

# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_csv("marketing_email_samples.csv")

# ------------------------------
# 2. Extract & clean raw text from marketing_email
# ------------------------------
def preprocess_raw(text):
    text = str(text)

    # Remove "Subject line:" prefix
    text = re.sub(r"subject line:\s*", "", text, flags=re.IGNORECASE)

    # Remove greetings (optional)
    text = re.sub(r"^(dear\s+[a-zA-Z ]+,)", "", text, flags=re.IGNORECASE)

    # Remove signature placeholders like [Your Name]
    text = re.sub(r"\[[^\]]*\]", " ", text)

    return text.strip()

df['Message'] = df['marketing_email'].apply(preprocess_raw)

# ------------------------------
# 3. Assign category
# ------------------------------
df['Category'] = "promotion"

# ------------------------------
# 4. Keep only required columns
# ------------------------------
df = df[['Category', 'Message']]

# ------------------------------
# 5. Drop empty / short / duplicate messages
# ------------------------------
df = df.dropna(subset=['Message'])
df['Message'] = df['Message'].astype(str).str.strip()
df = df[df['Message'].str.len() > 5]
df = df.drop_duplicates(subset=['Message'])

# ------------------------------
# 6. Deep text cleaning
# ------------------------------
def clean_text(text):
    text = text.lower()

    # remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove html tags
    text = re.sub(r"<.*?>", " ", text)

    # remove excessive line breaks or tabs
    text = re.sub(r"\n|\r|\t", " ", text)

    # remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

df['Message'] = df['Message'].apply(clean_text)

# ------------------------------
# 7. Limit to 700 rows
# ------------------------------
df = df.head(700)

# ------------------------------
# 8. Save final cleaned file
# ------------------------------
df.to_csv("promotion_cleaned.csv", index=False)

print("Done! promotion_cleaned.csv saved. Shape:", df.shape)
df.head()
