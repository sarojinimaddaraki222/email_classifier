import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import os

# ---------------- FILE STORAGE ----------------
FILE_NAME = "emails.json"

def load_emails():
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, "r") as f:
            data = json.load(f)
            for e in data:
                e['time'] = datetime.fromisoformat(e['time'])
            return data
    return []

def save_emails(emails):
    data = []
    for e in emails:
        temp = e.copy()
        temp['time'] = temp['time'].isoformat()
        data.append(temp)
    with open(FILE_NAME, "w") as f:
        json.dump(data, f)

# ---------------- INIT ----------------
if 'emails' not in st.session_state:
    st.session_state.emails = load_emails()

# ---------------- KEYWORDS ----------------
CATEGORY_KEYWORDS = {
    'spam': ['win','prize','free','lottery','offer'],
    'complaint': ['error','issue','problem','failed','not working'],
    'feedback': ['thank','great','excellent','love'],
    'request': ['please','need','help','request']
}

URGENCY_KEYWORDS = {
    'high': ['urgent','asap','critical','down'],
    'medium': ['soon','issue','problem'],
    'low': ['fyi','no rush']
}

# ---------------- FUNCTIONS ----------------
def classify_email(subject, body):
    text = (subject + " " + body).lower()

    scores = {k:0 for k in CATEGORY_KEYWORDS}

    for cat, words in CATEGORY_KEYWORDS.items():
        for w in words:
            if w in text:
                scores[cat] += 2 if w in subject.lower() else 1

    category = max(scores, key=scores.get)

    confidence = 0.4 if max(scores.values()) == 0 else min(0.5 + scores[category]*0.1, 0.95)

    urgency = "medium"
    for w in URGENCY_KEYWORDS['high']:
        if w in text:
            urgency = "high"
            break
    for w in URGENCY_KEYWORDS['low']:
        if w in text:
            urgency = "low"

    return category, confidence, urgency

def get_sentiment(text):
    pos = ['good','great','happy','love']
    neg = ['bad','error','issue','problem']

    score = sum([1 for w in pos if w in text]) - sum([1 for w in neg if w in text])

    if score > 0: return "Positive"
    if score < 0: return "Negative"
    return "Neutral"

def generate_reply(category):
    replies = {
        "complaint": "We are sorry. Our team is working on it.",
        "request": "Thanks for your request. We’ll respond soon.",
        "feedback": "Thanks for your feedback!",
        "spam": "This is marked as spam."
    }
    return replies.get(category, "Thanks!")

# ---------------- UI ----------------
st.title("📧 AI Email Classification System")

# Sidebar
st.sidebar.header("Add Email")
subject = st.sidebar.text_input("Subject")
body = st.sidebar.text_area("Body")

if st.sidebar.button("Classify"):
    if subject and body:
        cat, conf, urg = classify_email(subject, body)
        sentiment = get_sentiment(body)

        new_email = {
            "subject": subject,
            "body": body,
            "category": cat,
            "confidence": conf,
            "urgency": urg,
            "sentiment": sentiment,
            "time": datetime.now(),
            "status": "Pending"
        }

        st.session_state.emails.append(new_email)
        save_emails(st.session_state.emails)
        st.sidebar.success(f"{cat.upper()} | {urg.upper()}")

# ---------------- METRICS ----------------
emails = st.session_state.emails

col1, col2, col3 = st.columns(3)
col1.metric("Total Emails", len(emails))
col2.metric("High Priority", len([e for e in emails if e['urgency']=="high"]))
col3.metric("Complaints", len([e for e in emails if e['category']=="complaint"]))

# ---------------- CHART ----------------
if emails:
    df = pd.DataFrame(emails)
    fig = px.pie(df, names='category', title="Category Distribution")
    st.plotly_chart(fig)

# ---------------- SEARCH ----------------
search = st.text_input("🔍 Search")

if search:
    emails = [e for e in emails if search.lower() in e['subject'].lower()]

# ---------------- EMAIL LIST ----------------
for i, e in enumerate(emails):
    st.subheader(e['subject'])
    st.write(e['body'])

    st.write(f"Category: {e['category']} | Urgency: {e['urgency']}")
    st.write(f"Sentiment: {e['sentiment']}")

    st.progress(e['confidence'])

    st.info(generate_reply(e['category']))

    st.caption(f"Status: {e['status']}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(f"Resolve {i}"):
            st.session_state.emails[i]['status'] = "Resolved"
            save_emails(st.session_state.emails)
            st.rerun()

    with col2:
        if st.button(f"Delete {i}"):
            st.session_state.emails.pop(i)
            save_emails(st.session_state.emails)
            st.rerun()

# ---------------- DOWNLOAD ----------------
if st.button("Download CSV"):
    df = pd.DataFrame(st.session_state.emails)
    st.download_button("Download", df.to_csv(index=False), "emails.csv")
