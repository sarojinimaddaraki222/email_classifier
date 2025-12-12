import pandas as pd
import os

def combine_cleaned_datasets():
    """Combine all cleaned datasets into one"""
    print("🔄 Combining cleaned datasets...\n")
    
    datasets = []
    
    # Load spam dataset
    try:
        spam_df = pd.read_csv("../data/cleaned/spam_cleaned.csv")
        spam_df['source'] = 'spam'
        datasets.append(spam_df)
        print(f"✅ Loaded spam dataset: {len(spam_df)} emails")
    except FileNotFoundError:
        print("❌ spam_cleaned.csv not found")
    
    # Load enron dataset
    try:
        enron_df = pd.read_csv("../data/cleaned/enron_cleaned.csv")
        enron_df['source'] = 'enron'
        datasets.append(enron_df)
        print(f"✅ Loaded enron dataset: {len(enron_df)} emails")
    except FileNotFoundError:
        print("❌ enron_cleaned.csv not found")
    
    # Load support dataset
    try:
        support_df = pd.read_csv("../data/cleaned/support_cleaned.csv")
        support_df['source'] = 'support'
        datasets.append(support_df)
        print(f"✅ Loaded support dataset: {len(support_df)} emails")
    except FileNotFoundError:
        print("❌ enterprise_cleaned.csv not found")
    
    if not datasets:
        print("\n❌ No datasets found! Please run preprocessing scripts first.")
        return
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Ensure required columns exist
    required_cols = ['cleaned_text']
    for col in required_cols:
        if col not in combined_df.columns:
            # Try alternative column names
            alt_names = ['clean_text', 'text', 'message']
            for alt in alt_names:
                if alt in combined_df.columns:
                    combined_df['cleaned_text'] = combined_df[alt]
                    break
    
    # Remove duplicates
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['cleaned_text'], keep='first')
    print(f"\n🗑️  Removed {initial_count - len(combined_df)} duplicate emails")
    
    # Remove empty texts
    combined_df = combined_df[combined_df['cleaned_text'].str.strip() != '']
    
    # Save combined dataset
    output_path = "../data/cleaned/combined_emails.csv"
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved combined dataset: {output_path}")
    print(f"\n📊 Combined Dataset Statistics:")
    print(f"   Total emails: {len(combined_df)}")
    print(f"\n   Source distribution:")
    print(combined_df['source'].value_counts())
    
    if 'label' in combined_df.columns:
        print(f"\n   Label distribution:")
        print(combined_df['label'].value_counts())
    
    return combined_df

if __name__ == "__main__":
    combine_cleaned_datasets()