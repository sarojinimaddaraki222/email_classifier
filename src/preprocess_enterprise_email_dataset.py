import pandas as pd
import re
import os

RAW_PATH = "../data/raw/enterprise_email_dataset.csv"
SAVE_PATH = "../data/cleaned/enterprise_cleaned.csv"

def clean_text(t):
    if pd.isna(t):
        return ""
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def preprocess_enterprise():
    print("🟢 Loading enterprise dataset...")
    
    df = pd.read_csv(RAW_PATH)
    df["clean_text"] = df["email_text"].apply(clean_text)

    os.makedirs("../data/cleaned", exist_ok=True)
    df[["clean_text", "category", "urgency"]].to_csv(SAVE_PATH, index=False)

    print("✅ Saved at:", SAVE_PATH)


if __name__ == "__main__":
    preprocess_enterprise()
