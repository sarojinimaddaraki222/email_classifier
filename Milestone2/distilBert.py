import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

print("="*60)
print("DISTILBERT - ULTRA FAST VERSION")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using: {device}")

# Load dataset
df = pd.read_csv("../data/cleaned/labeled_emails.csv", low_memory=False)
df = df.dropna(subset=['cleaned_text'])
print(f"\n✅ Loaded: {len(df)} emails")

# Use only 5000 samples for ultra-fast training
print(f"⚡ Sampling 5,000 emails for ultra-fast training...")
df = df.sample(n=5000, random_state=42)
print(f"✅ Using: {len(df)} emails")

# Map to 4 categories
def map_category(cat):
    cat = str(cat).lower()
    if 'spam' in cat:
        return 'spam'
    elif 'support' in cat or 'work' in cat:
        return 'complaints'
    elif 'finance' in cat or 'general' in cat:
        return 'requests'
    else:
        return 'feedback'

df['category_4'] = df['category'].apply(map_category)

print(f"\n📊 Category Distribution:")
print(df['category_4'].value_counts())

# Prepare data
texts = df['cleaned_text'].tolist()
labels = df['category_4'].tolist()

# Label mapping
label_map = {"complaints": 0, "requests": 1, "spam": 2, "feedback": 3}
numeric_labels = [label_map[label] for label in labels]

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, numeric_labels, test_size=0.2, random_state=42
)

print(f"   Training: {len(train_texts)} | Testing: {len(test_texts)}")

# Load tokenizer
print("\n🔧 Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize with shorter max_length
print("🔧 Tokenizing...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)

# Dataset class
class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, train_labels)
test_dataset = EmailDataset(test_encodings, test_labels)

# Smaller batch size
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load model
print("\n🔧 Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
).to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training function
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return predictions, true_labels

# Train for only 1 epoch (ultra-fast)
print("\n🚀 Training DistilBERT (1 epoch only)...")
num_epochs = 1

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"   Training Loss: {train_loss:.4f}")

# Evaluate
print("\n📊 Evaluating on test set...")
predictions, true_labels = evaluate(model, test_loader, device)

accuracy = accuracy_score(true_labels, predictions)
print(f"\n✅ Accuracy: {accuracy:.4f}")

# Classification report
reverse_label_map = {v: k for k, v in label_map.items()}
pred_names = [reverse_label_map[idx] for idx in predictions]
test_names = [reverse_label_map[idx] for idx in true_labels]

print("\n📊 Classification Report:")
print(classification_report(test_names, pred_names))

print("\n" + "="*60)
print("DISTILBERT COMPLETE (Ultra-Fast Mode)")
print("="*60)
print("\n⚡ Note: Used only 5,000 samples and 1 epoch for speed")
print("   For better accuracy, increase sample size and epochs")