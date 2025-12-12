import pandas as pd
import re
import os

RAW_PATH = "../data/raw/spam.csv"
SAVE_PATH = "../data/cleaned/spam_cleaned.csv"

def clean_text(t):
    t = str(t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def preprocess_spam():
    print("🟣 Loading spam dataset...")

    df = pd.read_csv(RAW_PATH, encoding="latin1")

    # v1 = label (ham/spam), v2 = message
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    df["clean_text"] = df["text"].apply(clean_text)

    os.makedirs("../data/cleaned", exist_ok=True)
    df[["clean_text", "label"]].to_csv(SAVE_PATH, index=False)

    print("✅ Saved cleaned spam dataset at:", SAVE_PATH)


if __name__ == "__main__":
    preprocess_spam()
