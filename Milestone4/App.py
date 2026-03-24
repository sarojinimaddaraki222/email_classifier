import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import random

# Page config
st.set_page_config(page_title="AI Email Classification", layout="wide")

# -------------------- KEYWORDS --------------------
CATEGORY_KEYWORDS = {
    'spam': ['win','prize','click','free','lottery','offer'],
    'complaint': ['not working','error','issue','problem','failed','angry'],
    'feedback': ['thank','great','excellent','awesome','love'],
    'request': ['please','need','help','request','can you']
}

URGENCY_KEYWORDS = {
    'high': ['urgent','asap','immediately','critical','down','failed'],
    'medium': ['soon','help','issue','problem'],
    'low': ['fyi','no rush','whenever']
}

# -------------------- SESSION --------------------
if 'emails' not in st.session_state:
    st.session_state.emails = []

# -------------------- CLASSIFICATION --------------------
def classify_email(subject, body):
    text = f"{subject} {body}".lower()

    scores = {k:0 for k in CATEGORY_KEYWORDS}

    for category, words in CATEGORY_KEYWORDS.items():
        for word in words:
            if word in text:
                scores[category] += 2 if word in subject.lower() else 1

    category = max(scores, key=scores.get)

    if max(scores.values()) == 0:
        category = "request"
        confidence = 0.4
    else:
        confidence = min(0.5 + scores[category]*0.1, 0.95)

    # urgency
    urgency = "medium"
    urgency_conf = 0.5

    for w in URGENCY_KEYWORDS['high']:
        if w in text:
            urgency = "high"
            urgency_conf = 0.9
            break

    if urgency != "high":
        for w in URGENCY_KEYWORDS['low']:
            if w in text:
                urgency = "low"
                urgency_conf = 0.7
                break

    return category, confidence, urgency, urgency_conf

# -------------------- UI --------------------
st.title("📧 AI Email Classification System")

# Sidebar input
st.sidebar.header("Add Email")
subject = st.sidebar.text_input("Subject")
body = st.sidebar.text_area("Body")

if st.sidebar.button("Classify"):
    if subject and body:
        cat, conf, urg, urg_conf = classify_email(subject, body)

        st.session_state.emails.append({
            "subject": subject,
            "body": body,
            "category": cat,
            "confidence": conf,
            "urgency": urg,
            "urg_conf": urg_conf,
            "time": datetime.now()
        })

        st.sidebar.success(f"{cat.upper()} | {urg.upper()}")

# -------------------- SEARCH --------------------
search = st.text_input("🔍 Search emails")

emails = st.session_state.emails

if search:
    emails = [e for e in emails if search.lower() in e['subject'].lower()]

# -------------------- SORT --------------------
emails = sorted(emails, key=lambda x: (x['urgency']=="high", x['time']), reverse=True)

# -------------------- METRICS --------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Emails", len(emails))
col2.metric("High Priority", len([e for e in emails if e['urgency']=="high"]))
col3.metric("Complaints", len([e for e in emails if e['category']=="complaint"]))

# -------------------- CHART --------------------
if emails:
    df = pd.DataFrame(emails)
    fig = px.pie(df, names='category')
    st.plotly_chart(fig)

# -------------------- EMAIL LIST --------------------
for i, e in enumerate(emails):
    st.subheader(e['subject'])
    st.write(e['body'])

    st.progress(e['confidence'])
    st.caption(f"{e['category']} | {e['urgency']}")

    if st.button(f"Delete {i}"):
        st.session_state.emails.pop(i)
        st.rerun()

# -------------------- DOWNLOAD --------------------
if emails:
    df = pd.DataFrame(emails)
    st.download_button("Download CSV", df.to_csv(index=False), "emails.csv")
