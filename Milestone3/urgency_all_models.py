import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np

print("="*60)
print("MILESTONE 3: URGENCY DETECTION - ALL MODELS")
print("="*60)

# Load dataset
df = pd.read_csv("../Email_Datasets/cleaned/labeled_emails.csv", low_memory=False)
df = df.dropna(subset=['cleaned_text', 'urgency'])
print(f"\n✅ Loaded: {len(df)} emails")

print(f"\n📊 Urgency Distribution:")
print(df['urgency'].value_counts())

# Prepare data
X = df['cleaned_text']
y = df['urgency']

# TF-IDF Vectorization
print("\n🔧 TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
X_tfidf = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

# ============================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================

print("\n" + "="*60)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*60)

lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
print("\n🚀 Training Logistic Regression...")
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')

print(f"\n✅ Accuracy: {lr_accuracy:.4f}")
print(f"📊 F1-Score (Weighted): {lr_f1:.4f}")

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_lr))

# ============================================
# MODEL 2: NAIVE BAYES
# ============================================

print("\n" + "="*60)
print("MODEL 2: NAIVE BAYES")
print("="*60)

nb_model = MultinomialNB(alpha=0.1)
print("\n🚀 Training Naive Bayes...")
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted')

print(f"\n✅ Accuracy: {nb_accuracy:.4f}")
print(f"📊 F1-Score (Weighted): {nb_f1:.4f}")

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_nb))

# ============================================
# MODEL 3: RULE-BASED + HYBRID
# ============================================

print("\n" + "="*60)
print("MODEL 3: RULE-BASED + HYBRID")
print("="*60)

# Define urgency keywords
high_urgency_keywords = [
    "urgent", "asap", "immediately", "critical", "emergency", 
    "not working", "down", "failed", "broken", "deadline today"
]

medium_urgency_keywords = [
    "soon", "please", "request", "help", "issue", "problem"
]

def rule_based_urgency(text):
    text = str(text).lower()
    for word in high_urgency_keywords:
        if word in text:
            return "high"
    for word in medium_urgency_keywords:
        if word in text:
            return "medium"
    return "medium"

def hybrid_urgency(text, vectorizer, model):
    rule_result = rule_based_urgency(text)
    if rule_result == "high":
        return "high"
    text_vec = vectorizer.transform([str(text)])
    return model.predict(text_vec)[0]

# Get test texts
_, X_test_df, _, y_test_hybrid = train_test_split(
    df[['cleaned_text']], df['urgency'], test_size=0.2, random_state=42, stratify=df['urgency']
)

print("\n🚀 Applying Hybrid Approach (Rule + LR)...")
y_pred_hybrid = [hybrid_urgency(text, vectorizer, lr_model) for text in X_test_df['cleaned_text'].values]

hybrid_accuracy = accuracy_score(y_test_hybrid, y_pred_hybrid)
hybrid_f1 = f1_score(y_test_hybrid, y_pred_hybrid, average='weighted')

print(f"\n✅ Accuracy: {hybrid_accuracy:.4f}")
print(f"📊 F1-Score (Weighted): {hybrid_f1:.4f}")

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test_hybrid, y_pred_hybrid))

print("\n📊 Classification Report:")
print(classification_report(y_test_hybrid, y_pred_hybrid))

# ============================================
# COMPARISON
# ============================================

print("\n" + "="*60)
print("MODEL COMPARISON - URGENCY DETECTION")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Hybrid (Rule + LR)'],
    'Accuracy': [lr_accuracy, nb_accuracy, hybrid_accuracy],
    'F1-Score': [lr_f1, nb_f1, hybrid_f1]
})

print("\n📊 Performance Comparison:")
print(comparison.to_string(index=False))

best_idx = comparison['Accuracy'].idxmax()
print(f"\n🏆 Best Model: {comparison.loc[best_idx, 'Model']}")
print(f"   Accuracy: {comparison.loc[best_idx, 'Accuracy']:.4f}")
print(f"   F1-Score: {comparison.loc[best_idx, 'F1-Score']:.4f}")

print("\n" + "="*60)
print("MILESTONE 3 COMPLETE!")
print("="*60)
print("\n✅ Trained 3 models for urgency detection")
print("✅ Validated with confusion matrix & F1 score")
print("✅ Compared all approaches")