import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print("="*60)
print("NAIVE BAYES - BASELINE CLASSIFIER")
print("="*60)

# Load dataset
df = pd.read_csv("../data/cleaned/labeled_emails.csv")
df = df.dropna(subset=['cleaned_text'])
print(f"\n✅ Loaded: {len(df)} emails")

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
X = df['cleaned_text']
y = df['category_4']

# TF-IDF Vectorization
print("\n🔧 TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print(f"   Training: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

# Train Naive Bayes
print("\n🚀 Training Naive Bayes...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n" + "="*60)
print("NAIVE BAYES COMPLETE")
print("="*60)