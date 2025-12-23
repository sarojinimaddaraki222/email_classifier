import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ==================================================
# DISTILBERT – 50K DATA | 3 EPOCHS
# ==================================================
print("=" * 60)
print("DISTILBERT – FINAL (50K DATA, 3 EPOCHS)")
print("=" * 60)

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# --------------------------------------------------
# Dataset Path (CHANGE IF NEEDED)
# --------------------------------------------------
DATASET_PATH = "../data/cleaned/labeled_emails.csv"

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv(DATASET_PATH, low_memory=False)
df = df.dropna(subset=["cleaned_text"])
print(f"\n✅ Total usable rows: {len(df)}")

# Safe sampling to 50,000
sample_size = min(50000, len(df))
df = df.sample(n=sample_size, random_state=42)
print(f"✅ Using samples: {len(df)}")

# --------------------------------------------------
# Map to 4 Categories
# --------------------------------------------------
def map_category(cat):
    cat = str(cat).lower()
    if "spam" in cat:
        return "spam"
    elif "support" in cat or "work" in cat:
        return "complaints"
    elif "finance" in cat or "general" in cat:
        return "requests"
    else:
        return "feedback"

df["final_label"] = df["category"].apply(map_category)

print("\n📊 Label Distribution:")
print(df["final_label"].value_counts())

# --------------------------------------------------
# Label Encoding
# --------------------------------------------------
label_map = {
    "complaints": 0,
    "requests": 1,
    "spam": 2,
    "feedback": 3
}

texts = df["cleaned_text"].tolist()
labels = [label_map[label] for label in df["final_label"]]

# --------------------------------------------------
# Train-Test Split (Stratified)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"\n🧪 Training: {len(X_train)} | Testing: {len(X_test)}")

# --------------------------------------------------
# Tokenizer
# --------------------------------------------------
print("\n🔧 Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_encodings = tokenizer(
    X_train, truncation=True, padding=True, max_length=128
)
test_encodings = tokenizer(
    X_test, truncation=True, padding=True, max_length=128
)

# --------------------------------------------------
# Dataset Class
# --------------------------------------------------
class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, y_train)
test_dataset = EmailDataset(test_encodings, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# --------------------------------------------------
# Model
# --------------------------------------------------
print("\n🔧 Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# --------------------------------------------------
# Training Function
# --------------------------------------------------
def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------
def evaluate(model, loader):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            true.extend(labels.cpu().numpy())

    return preds, true

# --------------------------------------------------
# Training (3 Epochs)
# --------------------------------------------------
EPOCHS = 3
print("\n🚀 Training started...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    loss = train_epoch(model, train_loader)
    print(f"   Training Loss: {loss:.4f}")

# --------------------------------------------------
# Final Evaluation
# --------------------------------------------------
print("\n📊 Evaluating on test set...")
preds, true = evaluate(model, test_loader)

accuracy = accuracy_score(true, preds)
print(f"\n✅ Accuracy: {accuracy:.4f}")

reverse_map = {v: k for k, v in label_map.items()}
pred_names = [reverse_map[p] for p in preds]
true_names = [reverse_map[t] for t in true]

print("\n📊 Classification Report:")
print(classification_report(true_names, pred_names))

print("\n" + "=" * 60)
print("✅ DISTILBERT TRAINING COMPLETE (50K, 3 EPOCHS)")
print("=" * 60)
