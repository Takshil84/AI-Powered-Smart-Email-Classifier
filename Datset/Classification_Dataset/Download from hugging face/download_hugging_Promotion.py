
from datasets import load_dataset
import pandas as pd
import os

dataset_name = "Isotonic/marketing_email_samples"   
split_name = "test"                                # 'train', 'test', or any split
output_csv = "marketing_email_samples.csv"          


print(f"Downloading dataset: {dataset_name} ...")

# Load the dataset from HuggingFace
ds = load_dataset(dataset_name)

# Convert the split to Pandas DataFrame
df = ds[split_name].to_pandas()

# Save to CSV (in the same folder as your script)
df.to_csv(output_csv, index=False)

print(f"Dataset saved successfully as: {output_csv}")
print(f"Rows: {len(df)}")
print("Preview:")
print(df.head())
