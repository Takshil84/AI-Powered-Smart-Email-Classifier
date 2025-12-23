import pandas as pd
import re

# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_csv("social_media_emails.csv")

# ------------------------------
# 2. Create unified message column
# ------------------------------
def combine_text(row):
    # Prefer 'text' if available
    if isinstance(row['text'], str) and row['text'].strip() != "":
        return row['text']
    else:
        subject = row['subject'] if isinstance(row['subject'], str) else ""
        body = row['body'] if isinstance(row['body'], str) else ""
        return subject + " " + body

df['Message'] = df.apply(combine_text, axis=1)


# 3. Keep only needed columns & clean category
df['Category'] = df['category'].str.strip().str.lower()

df = df[['Category', 'Message']]

# Drop empty & duplicate messages
df = df.dropna(subset=['Message'])
df = df.drop_duplicates(subset=['Message'])

# 4. Cleaning function

def clean_text(text):
    text = text.lower()                                        # lowercase
    text = re.sub(r"\S+@\S+\.\S+", "", text)                   # remove emails
    text = re.sub(r"http\S+|www\S+", "", text)                 # remove URLs
    text = re.sub(r"<.*?>", "", text)                          # remove HTML
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)                # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()                   # extra space cleanup
    return text

df['Message'] = df['Message'].astype(str).apply(clean_text)


# 5. Keep only first 700 rows
df = df.head(700)

# 6. Save output
df.to_csv("social_cleaned.csv", index=False)

print("Done! Final dataset shape:", df.shape)
df.head()
