import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np

print("="*60)
print("MILESTONE 3: URGENCY DETECTION & SCORING")
print("="*60)

# Load dataset
df = pd.read_csv("../Email_Datasets/cleaned/labeled_emails.csv", low_memory=False)
df = df.dropna(subset=['cleaned_text', 'urgency'])
print(f"\n✅ Loaded: {len(df)} emails")

# Check urgency distribution
print(f"\n📊 Urgency Distribution:")
print(df['urgency'].value_counts())

# ============================================
# STEP 1: RULE-BASED URGENCY DETECTION
# ============================================

print("\n" + "="*60)
print("STEP 1: RULE-BASED URGENCY DETECTION")
print("="*60)

# Define urgency keywords
high_urgency_keywords = [
    "urgent", "asap", "immediately", "critical", "emergency", 
    "not working", "down", "failed", "broken", "deadline today",
    "rush", "important", "serious", "crisis"
]

medium_urgency_keywords = [
    "soon", "please", "request", "help", "issue", "problem",
    "this week", "follow up", "reminder", "when possible"
]

low_urgency_keywords = [
    "fyi", "information", "no rush", "when you can", "whenever",
    "just checking", "heads up", "for your reference"
]

def rule_based_urgency(text):
    """Rule-based urgency detection using keywords"""
    text = str(text).lower()
    
    # Check for high urgency keywords
    for word in high_urgency_keywords:
        if word in text:
            return "high"
    
    # Check for medium urgency keywords
    for word in medium_urgency_keywords:
        if word in text:
            return "medium"
    
    # Check for low urgency keywords
    for word in low_urgency_keywords:
        if word in text:
            return "low"
    
    # Default to medium if no keywords found
    return "medium"

# Test rule-based detection
print("\n🔍 Testing Rule-Based Detection:")
test_samples = [
    "System is down, please fix immediately!",
    "Can you help with my refund request?",
    "FYI - here's the update you requested"
]

for sample in test_samples:
    urgency = rule_based_urgency(sample)
    print(f"   Text: '{sample}'")
    print(f"   Predicted Urgency: {urgency}\n")

# Apply rule-based detection to dataset
print("📊 Applying rule-based detection to dataset...")
df['rule_based_urgency'] = df['cleaned_text'].apply(rule_based_urgency)

# Evaluate rule-based approach
rule_accuracy = accuracy_score(df['urgency'], df['rule_based_urgency'])
print(f"\n✅ Rule-Based Accuracy: {rule_accuracy:.4f}")

print("\n📊 Rule-Based Confusion Matrix:")
print(confusion_matrix(df['urgency'], df['rule_based_urgency']))

print("\n📊 Rule-Based Classification Report:")
print(classification_report(df['urgency'], df['rule_based_urgency']))

# ============================================
# STEP 2: ML-BASED URGENCY CLASSIFICATION
# ============================================

print("\n" + "="*60)
print("STEP 2: ML-BASED URGENCY CLASSIFICATION")
print("="*60)

# Prepare data for ML
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

# Train Logistic Regression for Urgency
print("\n🚀 Training Logistic Regression for Urgency...")
ml_urgency_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
ml_urgency_model.fit(X_train, y_train)

# Predict
y_pred_ml = ml_urgency_model.predict(X_test)

# Evaluate ML model
ml_accuracy = accuracy_score(y_test, y_pred_ml)
print(f"\n✅ ML-Based Accuracy: {accuracy_score(y_test, y_pred_ml):.4f}")

print("\n📊 ML-Based Confusion Matrix:")
cm_ml = confusion_matrix(y_test, y_pred_ml)
print(cm_ml)

print("\n📊 ML-Based Classification Report:")
print(classification_report(y_test, y_pred_ml))

# Calculate F1 Score
f1_ml = f1_score(y_test, y_pred_ml, average='weighted')
print(f"\n📊 ML-Based F1 Score (Weighted): {f1_ml:.4f}")

# ============================================
# STEP 3: HYBRID APPROACH (RULE + ML)
# ============================================

print("\n" + "="*60)
print("STEP 3: HYBRID URGENCY DETECTION (RULE + ML)")
print("="*60)

def hybrid_urgency_detection(text, vectorizer, ml_model):
    """
    Hybrid approach: Use rule-based first, fall back to ML
    """
    # Step 1: Rule-based check
    rule_result = rule_based_urgency(text)
    
    # If rule-based finds high urgency, trust it
    if rule_result == "high":
        return "high"
    
    # Otherwise, use ML prediction
    text_vec = vectorizer.transform([str(text)])
    ml_pred = ml_model.predict(text_vec)[0]
    
    return ml_pred

# Test hybrid approach
print("\n🔍 Testing Hybrid Detection:")
for sample in test_samples:
    urgency = hybrid_urgency_detection(sample, vectorizer, ml_urgency_model)
    print(f"   Text: '{sample}'")
    print(f"   Predicted Urgency: {urgency}\n")

# Apply hybrid approach to test set
print("📊 Applying hybrid approach to test set...")

# Get the actual test data by splitting again (same random state)
_, X_test_df, _, y_test_actual = train_test_split(
    df[['cleaned_text']], df['urgency'], test_size=0.2, random_state=42, stratify=df['urgency']
)

X_test_texts = X_test_df['cleaned_text'].values
print(f"   Processing {len(X_test_texts)} test samples...")

y_pred_hybrid = []
for text in X_test_texts:
    pred = hybrid_urgency_detection(text, vectorizer, ml_urgency_model)
    y_pred_hybrid.append(pred)

# Evaluate hybrid approach
hybrid_accuracy = accuracy_score(y_test_actual, y_pred_hybrid)
print(f"\n✅ Hybrid Approach Accuracy: {hybrid_accuracy:.4f}")

print("\n📊 Hybrid Confusion Matrix:")
cm_hybrid = confusion_matrix(y_test_actual, y_pred_hybrid)
print(cm_hybrid)

print("\n📊 Hybrid Classification Report:")
print(classification_report(y_test_actual, y_pred_hybrid))

# Calculate F1 Score
f1_hybrid = f1_score(y_test_actual, y_pred_hybrid, average='weighted')
print(f"\n📊 Hybrid F1 Score (Weighted): {f1_hybrid:.4f}")

# ============================================
# STEP 4: COMPREHENSIVE COMPARISON
# ============================================

print("\n" + "="*60)
print("APPROACH COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Approach': ['Rule-Based', 'ML-Based', 'Hybrid (Rule + ML)'],
    'Accuracy': [rule_accuracy, ml_accuracy, hybrid_accuracy],
    'F1-Score': [
        f1_score(df['urgency'], df['rule_based_urgency'], average='weighted'),
        f1_ml,
        f1_hybrid
    ]
})

print("\n📊 Performance Comparison:")
print(comparison.to_string(index=False))

# Determine best approach
best_idx = comparison['Accuracy'].idxmax()
best_approach = comparison.loc[best_idx, 'Approach']
best_accuracy = comparison.loc[best_idx, 'Accuracy']

print(f"\n🏆 Best Approach: {best_approach}")
print(f"   Accuracy: {best_accuracy:.4f}")

# ============================================
# STEP 5: DETAILED ANALYSIS BY URGENCY LEVEL
# ============================================

print("\n" + "="*60)
print("PER-CLASS PERFORMANCE ANALYSIS")
print("="*60)

# ML-Based per-class metrics
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred_ml, labels=['high', 'medium', 'low']
)

per_class_df = pd.DataFrame({
    'Urgency Level': ['High', 'Medium', 'Low'],
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

print("\n📊 ML-Based Performance by Urgency Level:")
print(per_class_df.to_string(index=False))

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("MILESTONE 3 COMPLETE!")
print("="*60)

print("\n✅ Completed Tasks:")
print("   1. ✅ Implemented rule-based urgency detection")
print("   2. ✅ Trained ML urgency classification model")
print("   3. ✅ Combined ML + keyword-based detection (Hybrid)")
print("   4. ✅ Validated with confusion matrix & F1 score")

print(f"\n📊 Final Results:")
print(f"   Rule-Based: {rule_accuracy:.4f} accuracy")
print(f"   ML-Based: {ml_accuracy:.4f} accuracy")
print(f"   Hybrid: {hybrid_accuracy:.4f} accuracy")
print(f"   Best F1-Score: {max(comparison['F1-Score']):.4f}")

print("\n" + "="*60)