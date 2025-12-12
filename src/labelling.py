import pandas as pd

def assign_category(text):
    """Assign category based on keywords"""
    text = str(text).lower()
    
    categories = {
        'spam': ['free', 'winner', 'prize', 'click', 'buy', 'offer', 'deal', 'congratulations', 'urgent'],
        'work': ['meeting', 'project', 'deadline', 'report', 'presentation', 'schedule', 'agenda', 'task', 'team'],
        'personal': ['family', 'friend', 'party', 'vacation', 'dinner', 'birthday', 'weekend'],
        'support': ['help', 'issue', 'problem', 'support', 'ticket', 'error', 'bug', 'fix', 'resolve'],
        'finance': ['invoice', 'payment', 'bank', 'account', 'credit', 'bill', 'transaction', 'purchase']
    }
    
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in text)
        scores[category] = score
    
    if max(scores.values()) == 0:
        return 'general'
    return max(scores, key=scores.get)

def assign_urgency(text):
    """Assign urgency level based on keywords"""
    text = str(text).lower()
    
    urgency_keywords = {
        'high': ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'important', 'deadline today', 'rush'],
        'medium': ['soon', 'this week', 'reminder', 'follow up', 'please review', 'when possible'],
        'low': ['fyi', 'for your information', 'no rush', 'when you can', 'whenever']
    }
    
    for level in ['high', 'medium', 'low']:
        for keyword in urgency_keywords[level]:
            if keyword in text:
                return level
    
    return 'medium'

def label_emails():
    """Add category and urgency labels to combined dataset"""
    print("🏷️  Starting labelling process...\n")
    
    # Load combined dataset
    input_path = "../data/cleaned/combined_emails.csv"
    output_path = "../data/cleaned/labeled_emails.csv"
    
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Loaded combined dataset: {len(df)} emails")
    except FileNotFoundError:
        print(f"❌ Error: {input_path} not found!")
        print("   Please run combine_datasets.py first")
        return
    
    # Identify text column
    text_col = None
    for col in ['cleaned_text', 'clean_text', 'text']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print("❌ Error: Could not find text column!")
        return
    
    print(f"   Using text column: {text_col}")
    
    # Add category labels
    print("\n📂 Assigning categories...")
    df['category'] = df[text_col].apply(assign_category)
    
    # Add urgency labels
    print("⚡ Assigning urgency levels...")
    df['urgency'] = df[text_col].apply(assign_urgency)
    
    # Save labeled dataset
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved labeled dataset: {output_path}")
    print(f"\n📊 Labelling Statistics:")
    print(f"   Total emails: {len(df)}")
    
    print(f"\n   Category distribution:")
    for cat, count in df['category'].value_counts().items():
        print(f"      {cat}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n   Urgency distribution:")
    for urg, count in df['urgency'].value_counts().items():
        print(f"      {urg}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    label_emails()