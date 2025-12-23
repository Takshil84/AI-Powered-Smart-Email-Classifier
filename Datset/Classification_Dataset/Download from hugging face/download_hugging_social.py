from datasets import load_dataset
import pandas as pd
from huggingface_hub import login

# Authenticate with HuggingFace
print("Authenticating with HuggingFace...")
login(token="remomved for security") 

# Load the full dataset from HuggingFace
print("Downloading dataset...")
ds = load_dataset("jason23322/high-accuracy-email-classifier")

# Filter data where category is 'social_media'
print("Filtering social media data...")
social_media_data = ds.filter(lambda x: x['category'] == 'social_media')

# Convert to pandas DataFrame for easier handling
social_media_df = social_media_data['train'].to_pandas()

print(f"Found {len(social_media_df)} social media entries")
print(social_media_df.head())

# Save to CSV if needed
social_media_df.to_csv('social_media_emails.csv', index=False)
print("Social media data saved to 'social_media_emails.csv'")

