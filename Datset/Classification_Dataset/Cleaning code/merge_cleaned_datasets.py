import pandas as pd
import os
import random
from pathlib import Path

def merge_cleaned_datasets_randomly():
    """
    Merge all CSV files from cleaned_Dataset folder randomly
    """
    # Define the path to the cleaned dataset folder
    cleaned_dataset_path = Path("cleaned_Dataset")
    
    # Get all CSV files in the folder
    csv_files = list(cleaned_dataset_path.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in cleaned_Dataset folder")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Read and combine all CSV files
    all_dataframes = []
    total_rows = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_dataframes.append(df)
            print(f"Loaded {csv_file.name}: {len(df)} rows")
            total_rows += len(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not all_dataframes:
        print("No valid CSV files could be loaded")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nTotal rows before shuffling: {len(combined_df)}")
    
    # Randomly shuffle the combined dataset
    shuffled_df = combined_df.sample(frac=1, random_state=None).reset_index(drop=True)
    print(f"Dataset shuffled randomly")
    
    # Display category distribution
    print(f"\nCategory distribution in merged dataset:")
    category_counts = shuffled_df['Category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} rows")
    
    # Save the merged and shuffled dataset
    output_file = "merged_cleaned_dataset.csv"
    shuffled_df.to_csv(output_file, index=False)
    print(f"\nMerged dataset saved as '{output_file}'")
    print(f"Total rows in final dataset: {len(shuffled_df)}")
    
    return shuffled_df

if __name__ == "__main__":
    merged_data = merge_cleaned_datasets_randomly()