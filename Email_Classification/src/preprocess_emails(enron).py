import pandas as pd
import re
import os

RAW_PATH = "../data/raw/emails.csv"
SAVE_PATH = "../data/cleaned/enron_cleaned.csv"

def clean_text(text):
    if pd.isna(text):
        return ""

    # Remove metadata headers
    text = re.sub(r"(Message-ID:.*|Date:.*|From:.*|To:.*|Subject:.*)", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove non-alphabet chars
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess_enron():
    print("🔵 Loading Enron dataset...")

    df = pd.read_csv(RAW_PATH)
    print("Loaded:", df.shape)

    print("🔵 Cleaning...")
    df["clean_text"] = df["message"].apply(clean_text)

    os.makedirs("../data/cleaned", exist_ok=True)
    df[["clean_text"]].to_csv(SAVE_PATH, index=False)

    print("✅ Saved cleaned Enron dataset at:", SAVE_PATH)


if __name__ == "__main__":
    preprocess_enron()
