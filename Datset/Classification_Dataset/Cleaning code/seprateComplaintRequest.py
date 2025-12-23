# split_complaint_request.py
import pandas as pd
import re
from rapidfuzz import fuzz

# -------------------------
# Config
# -------------------------
INPUT_CSV = "Complaint&Request.csv"   # change if needed
OUT_COMPLAINT = "complaint_550.csv"
OUT_REQUEST = "request_550.csv"
SAMPLE_N = 7000

# -------------------------
# Load
# -------------------------
df = pd.read_csv(INPUT_CSV)
print("Columns found:", list(df.columns))

# Helpful: show unique Ticket Type values if present
if "Ticket Type" in df.columns:
    print("\nUnique values in 'Ticket Type':")
    print(df["Ticket Type"].dropna().unique()[:50])

# -------------------------
# Normalize text fields
# -------------------------
# Map possible column names to subject/description
subject_col = "Ticket Subject"
desc_col = "Ticket Description"

df[subject_col] = df[subject_col].fillna("").astype(str)
df[desc_col] = df[desc_col].fillna("").astype(str)
df["text"] = (df[subject_col] + " " + df[desc_col]).str.lower()

# -------------------------
# Enterprise keyword lists
# -------------------------
complaint_keywords = [
    "not working", "stopped working", "doesn't work", "doesnt work",
    "broken", "damaged", "defective", "faulty", "malfunction",
    "error", "failed", "failure", "glitch", "bug",
    "unacceptable", "pathetic", "horrible", "terrible", "worst",
    "poor service", "bad experience", "very disappointed", "not satisfied",
    "refund", "return request", "replacement", "refund not received",
    "refund pending", "overcharged", "charged twice", "billing issue",
    "not delivered", "never arrived", "missing package", "wrong item",
    "delay", "delayed", "cannot access", "can't access", "unauthorized",
    "account blocked", "login problem", "cannot login", "password not working"
]

request_keywords = [
    "please help", "need help", "kindly assist", "assist me",
    "support needed", "help needed", "please assist",
    "i want to know", "need information", "provide details",
    "need details", "want to check", "please clarify",
    "update address", "change address", "update phone", "reset password",
    "forgot password", "modify account", "account update",
    "order status", "track my order", "tracking", "current status",
    "update status", "progress update",
    "invoice request", "send invoice", "need invoice", "need receipt",
    "download statement",
    "how to", "guide me", "procedure for", "process for",
    "how can i", "steps needed",
    "product details", "availability", "pricing", "quote request",
    "schedule appointment"
]

# -------------------------
# Matching utilities
# -------------------------
def keyword_match(text, keywords, fuzzy_threshold=85):
    # quick substring first, then fuzzy partial match
    for kw in keywords:
        if kw in text:
            return True
        # fuzzy on shorter keywords can be noisy; use partial_ratio
        if fuzz.partial_ratio(text, kw) >= fuzzy_threshold:
            return True
    return False

# -------------------------
# Labeling strategy
# -------------------------
# If Ticket Type exists and clearly identifies complaint/request, prefer it.
df["complaint"] = 0
df["request"] = 0

# Normalizing Ticket Type (if present)
if "Ticket Type" in df.columns:
    def map_ticket_type(tt):
        if pd.isna(tt): 
            return None
        t = str(tt).strip().lower()
        # common variants that indicate complaint
        if any(x in t for x in ["complaint", "issue", "problem", "grievance", "bug", "fault"]):
            return "complaint"
        # common variants that indicate request
        if any(x in t for x in ["request", "query", "question", "inquiry", "support", "help"]):
            return "request"
        return None
    df["ticket_type_mapped"] = df["Ticket Type"].apply(map_ticket_type)
    # apply mapped ticket type
    df.loc[df["ticket_type_mapped"] == "complaint", "complaint"] = 1
    df.loc[df["ticket_type_mapped"] == "request", "request"] = 1
    print("\nTicket Type mapped counts:")
    print(df["ticket_type_mapped"].value_counts(dropna=False))

# For rows still unlabeled by ticket_type, use keyword/fuzzy matching
mask_unlabeled = (df["complaint"] == 0) & (df["request"] == 0)
print(f"\nRows unlabeled by Ticket Type: {mask_unlabeled.sum()}")

# Apply keyword matching only to unlabeled rows
df.loc[mask_unlabeled, "complaint"] = df.loc[mask_unlabeled, "text"].apply(
    lambda t: 1 if keyword_match(t, complaint_keywords) else 0
)
df.loc[mask_unlabeled, "request"] = df.loc[mask_unlabeled, "text"].apply(
    lambda t: 1 if keyword_match(t, request_keywords) else 0
)

# Conflict resolution: if both flagged, prefer complaint
df.loc[(df["complaint"] == 1) & (df["request"] == 1), "request"] = 0

# If still unlabeled & text ends with question mark → mark as request
still_unlabeled = (df["complaint"] == 0) & (df["request"] == 0)
df.loc[still_unlabeled & df["text"].str.strip().str.endswith("?"), "request"] = 1

# Final counts
print("\nFinal counts:")
print("Complaints:", int(df["complaint"].sum()))
print("Requests:", int(df["request"].sum()))
print("Unlabeled rows remaining:", int(((df["complaint"]==0)&(df["request"]==0)).sum()))

# -------------------------
# Extract DataFrames
# -------------------------
df_complaint = df[df["complaint"] == 1].reset_index(drop=True)
df_request = df[df["request"] == 1].reset_index(drop=True)

# -------------------------
# Sample up to SAMPLE_N each
# -------------------------
complaint_sample = df_complaint.sample(n=SAMPLE_N, random_state=42) if len(df_complaint) >= SAMPLE_N else df_complaint
request_sample = df_request.sample(n=SAMPLE_N, random_state=42) if len(df_request) >= SAMPLE_N else df_request

# -------------------------
# Save CSVs
# -------------------------
complaint_sample.to_csv(OUT_COMPLAINT, index=False)
request_sample.to_csv(OUT_REQUEST, index=False)

print(f"\nSaved: {OUT_COMPLAINT} ({len(complaint_sample)} rows)")
print(f"Saved: {OUT_REQUEST} ({len(request_sample)} rows)")

# -------------------------
# Quick preview for manual review
# -------------------------
print("\nSample complaint rows (5):")
print(complaint_sample[[subject_col, desc_col]].head(5).to_string(index=False))
print("\nSample request rows (5):")
print(request_sample[[subject_col, desc_col]].head(5).to_string(index=False))
