import pandas as pd
import re

# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_csv("complaint.csv")

# ------------------------------
# 2. Build Message = Subject + Description
# ------------------------------
def build_message(row):
    subj = row['Ticket Subject'] if isinstance(row['Ticket Subject'], str) else ""
    desc = row['Ticket Description'] if isinstance(row['Ticket Description'], str) else ""
    return (subj + " " + desc).strip()

df['Message'] = df.apply(build_message, axis=1)

# ------------------------------
# 3. Assign Category
# ------------------------------
df['Category'] = "complaint"

# ------------------------------
# 4. Keep only needed columns
# ------------------------------
df = df[['Category', 'Message']]

# ------------------------------
# 5. Drop empty / short / duplicate messages
# ------------------------------
df = df.dropna(subset=['Message'])
df['Message'] = df['Message'].astype(str).str.strip()  # Add .str before .strip()
df = df[df['Message'].str.len() > 5]     # remove extremely short text
df = df.drop_duplicates(subset=['Message'])

# ------------------------------
# 6. Cleaning function
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\S+@\S+\.\S+", " ", text)         # remove emails
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"<.*?>", " ", text)                # remove HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()          # remove extra spaces
    return text

df['Message'] = df['Message'].apply(clean_text)

# ------------------------------
# 7. Limit to 700 rows
# ------------------------------
df = df.head(700)

# ------------------------------
# 8. Save cleaned file
# ------------------------------
df.to_csv("complaint_cleaned.csv", index=False)

print("Done! complaint_cleaned.csv saved. Shape:", df.shape)
df.head()
