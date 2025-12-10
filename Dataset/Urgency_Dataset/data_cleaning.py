import pandas as pd
from pathlib import Path
import re
import warnings

warnings.filterwarnings('ignore')

class EmailUrgencyDataCleaner:
    """Class to handle data cleaning for email urgency datasets"""
    
    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)
        self.train_df = None
        self.test_df = None
        self.val_df = None
        
    def load_data(self):
        """Load the three datasets"""
        print("=" * 70)
        print("LOADING DATASETS")
        print("=" * 70)
        
        try:
            self.train_df = pd.read_csv(self.data_dir / 'train.csv')
            self.test_df = pd.read_csv(self.data_dir / 'test.csv')
            self.val_df = pd.read_csv(self.data_dir / 'validation.csv')
            
            print(f"✓ Train dataset loaded: {len(self.train_df)} samples")
            print(f"✓ Test dataset loaded: {len(self.test_df)} samples")
            print(f"✓ Validation dataset loaded: {len(self.val_df)} samples")
            print(f"\nTotal samples: {len(self.train_df) + len(self.test_df) + len(self.val_df)}")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def detect_duplicates(self):
        """Detect duplicate entries"""
        print("\n" + "=" * 70)
        print("DUPLICATE DETECTION")
        print("=" * 70)
        
        for name, df in [('Train', self.train_df), ('Test', self.test_df), ('Validation', self.val_df)]:
            duplicates = df.duplicated(subset=['text']).sum()
            exact_duplicates = df.duplicated().sum()
            
            print(f"\n{name} Dataset:")
            print(f"  Duplicate texts: {duplicates}")
            print(f"  Exact duplicates (text + label): {exact_duplicates}")
            
            if duplicates > 0:
                print(f"  ⚠ Warning: {duplicates} duplicate texts found!")
    
    def clean_text(self, text):
        """
        Clean individual email text entry
        - Removes HTML tags and entities
        - Removes URLs
        - Removes email signatures
        - Removes extra noise (special characters, excessive punctuation)
        - Lowercases text
        - Normalizes whitespace
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # 1. Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 2. Remove HTML entities (e.g., &nbsp;, &amp;, &#123;)
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#?\w+;', ' ', text)
        
        # 3. Remove URLs (http, https, www)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # 4. Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
        
        # 5. Remove common email signatures markers
        signature_patterns = [
            r'--+\s*',  # Signature separator (-- or ----)
            r'sent from my (?:iphone|ipad|android|blackberry|mobile)',
            r'regards,?',
            r'best regards,?',
            r'sincerely,?',
            r'thanks,?',
            r'thank you,?',
            r'cheers,?',
            r'kind regards,?',
            r'warm regards,?',
            r'get outlook for (?:ios|android)',
        ]
        for pattern in signature_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # 6. Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' ', text)
        text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', ' ', text)
        
        # 7. Remove excessive special characters and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 8. Normalize whitespace (multiple spaces, tabs, newlines to single space)
        text = re.sub(r'\s+', ' ', text)
        
        # 9. Lowercase the text
        text = text.lower()
        
        # 10. Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_datasets(self):
        """Clean all datasets"""
        print("\n" + "=" * 70)
        print("DATA CLEANING")
        print("=" * 70)
        
        datasets = [
            ('Train', self.train_df),
            ('Test', self.test_df),
            ('Validation', self.val_df)
        ]
        
        for name, df in datasets:
            print(f"\nCleaning {name} Dataset...")
            original_size = len(df)
            
            # 1. Handle missing values
            df_cleaned = df.dropna()
            missing_removed = original_size - len(df_cleaned)
            if missing_removed > 0:
                print(f"  ✓ Removed {missing_removed} rows with missing values")
            
            # 2. Clean text
            df_cleaned['text'] = df_cleaned['text'].apply(self.clean_text)
            print(f"  ✓ Text cleaned (whitespace normalized)")
            
            # 3. Remove empty texts
            empty_before = len(df_cleaned)
            df_cleaned = df_cleaned[df_cleaned['text'].str.len() > 0]
            empty_removed = empty_before - len(df_cleaned)
            if empty_removed > 0:
                print(f"  ✓ Removed {empty_removed} rows with empty text")
            
            # 4. Validate labels
            valid_labels = [0, 1, 2, 3]
            invalid_labels = df_cleaned[~df_cleaned['label'].isin(valid_labels)]
            if len(invalid_labels) > 0:
                print(f"  ⚠ Warning: {len(invalid_labels)} rows with invalid labels")
                df_cleaned = df_cleaned[df_cleaned['label'].isin(valid_labels)]
                print(f"  ✓ Removed rows with invalid labels")
            
            # 5. Remove exact duplicates
            duplicates_before = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            duplicates_removed = duplicates_before - len(df_cleaned)
            if duplicates_removed > 0:
                print(f"  ✓ Removed {duplicates_removed} exact duplicate rows")
            
            # 6. Reset index
            df_cleaned = df_cleaned.reset_index(drop=True)
            
            # Update the dataframe
            if name == 'Train':
                self.train_df = df_cleaned
            elif name == 'Test':
                self.test_df = df_cleaned
            else:
                self.val_df = df_cleaned
            
            final_size = len(df_cleaned)
            total_removed = original_size - final_size
            
            print(f"  Summary: {original_size} → {final_size} rows "
                  f"({total_removed} removed, {(total_removed/original_size)*100:.2f}%)")
    
    def save_cleaned_data(self):
        """Save cleaned datasets"""
        print("\n" + "=" * 70)
        print("SAVING CLEANED DATA")
        print("=" * 70)
        
        # Create cleaned data directory
        cleaned_dir = self.data_dir / 'cleaned'
        cleaned_dir.mkdir(exist_ok=True)
        
        # Save cleaned datasets
        self.train_df.to_csv(cleaned_dir / 'train_cleaned.csv', index=False)
        print(f"✓ Saved: {cleaned_dir / 'train_cleaned.csv'} ({len(self.train_df)} rows)")
        
        self.test_df.to_csv(cleaned_dir / 'test_cleaned.csv', index=False)
        print(f"✓ Saved: {cleaned_dir / 'test_cleaned.csv'} ({len(self.test_df)} rows)")
        
        self.val_df.to_csv(cleaned_dir / 'validation_cleaned.csv', index=False)
        print(f"✓ Saved: {cleaned_dir / 'validation_cleaned.csv'} ({len(self.val_df)} rows)")
    
    def run_cleaning(self):
        """Run the complete cleaning pipeline"""
        print("\n" + "🧹 " * 20)
        print("EMAIL URGENCY DETECTION - DATA CLEANING")
        print("🧹 " * 20)
        
        try:
            self.load_data()
            self.detect_duplicates()
            self.clean_datasets()
            self.save_cleaned_data()
            
            print("\n✅ All cleaning operations completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error during cleaning: {e}")
            raise


if __name__ == "__main__":
    cleaner = EmailUrgencyDataCleaner()
    cleaner.run_cleaning()
