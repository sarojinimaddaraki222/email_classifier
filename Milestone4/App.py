import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="AI Email Classification System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .badge-complaint {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
    }
    .badge-request {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
    }
    .badge-feedback {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
    }
    .badge-spam {
        background-color: #f3f4f6;
        color: #374151;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
    }
    .urgency-high {
        background-color: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
    }
    .urgency-medium {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
    }
    .urgency-low {
        background-color: #22c55e;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
    }
    .success-box {
        background: linear-gradient(to right, #eff6ff, #f0fdf4);
        border: 2px solid #bfdbfe;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Classification keywords
CATEGORY_KEYWORDS = {
    'spam': ['win', 'prize', 'click here', 'congratulations', 'free money', 'viagra', 'lottery', 'act now'],
    'complaint': ['not working', 'error', 'issue', 'problem', 'broken', 'failed', 'disappointed', 'angry', 'frustrated', 'terrible', 'worst'],
    'feedback': ['thank', 'great', 'excellent', 'awesome', 'love', 'appreciate', 'wonderful', 'amazing', 'happy'],
    'request': ['please', 'could you', 'can you', 'would like', 'need', 'want', 'help', 'request', 'feature']
}

URGENCY_KEYWORDS = {
    'high': ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'not working', 'down', 'failed', 'broken', 'deadline today', 'rush', 'important', 'serious', 'crisis'],
    'medium': ['soon', 'please', 'request', 'help', 'issue', 'problem', 'this week', 'follow up', 'reminder', 'when possible'],
    'low': ['fyi', 'information', 'no rush', 'when you can', 'whenever', 'just checking', 'heads up', 'for your reference']
}

# Initialize session state
if 'emails' not in st.session_state:
    st.session_state.emails = [
        {
            'id': 1,
            'subject': 'URGENT: System Down',
            'body': 'Our payment system is not working...',
            'category': 'complaint',
            'urgency': 'high',
            'confidence_category': 0.85,
            'confidence_urgency': 0.90,
            'timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'id': 2,
            'subject': 'Feature Request',
            'body': 'Would love to see dark mode...',
            'category': 'request',
            'urgency': 'low',
            'confidence_category': 0.75,
            'confidence_urgency': 0.80,
            'timestamp': datetime.now() - timedelta(hours=5)
        },
        {
            'id': 3,
            'subject': 'Great Service!',
            'body': 'Thank you for the excellent support...',
            'category': 'feedback',
            'urgency': 'low',
            'confidence_category': 0.88,
            'confidence_urgency': 0.75,
            'timestamp': datetime.now() - timedelta(days=1)
        }
    ]

if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

# Classification function
def classify_email(subject, body):
    text = f"{subject} {body}".lower()
    
    # Classify Category
    category_scores = {
        'spam': 0,
        'complaint': 0,
        'feedback': 0,
        'request': 0
    }

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                category_scores[category] += 1

    category = max(category_scores, key=category_scores.get)
    category_confidence = min(0.5 + (category_scores[category] * 0.15), 0.95) if category_scores[category] > 0 else 0.4

    # Classify Urgency
    urgency = 'medium'
    urgency_confidence = 0.5

    for keyword in URGENCY_KEYWORDS['high']:
        if keyword in text:
            urgency = 'high'
            urgency_confidence = 0.9
            break

    if urgency != 'high':
        for keyword in URGENCY_KEYWORDS['low']:
            if keyword in text:
                urgency = 'low'
                urgency_confidence = 0.75
                break

    if urgency == 'medium':
        for keyword in URGENCY_KEYWORDS['medium']:
            if keyword in text:
                urgency_confidence = 0.7
                break

    if category_scores[category] == 0:
        category = 'request'

    return {
        'category': category,
        'urgency': urgency,
        'confidence_category': category_confidence,
        'confidence_urgency': urgency_confidence
    }

# Helper functions
def format_timestamp(timestamp):
    now = datetime.now()
    diff = now - timestamp
    
    if diff.total_seconds() < 3600:
        return f"{int(diff.total_seconds() / 60)}m ago"
    elif diff.total_seconds() < 86400:
        return f"{int(diff.total_seconds() / 3600)}h ago"
    else:
        return timestamp.strftime("%b %d, %Y")

def get_category_badge(category):
    badges = {
        'complaint': 'badge-complaint',
        'request': 'badge-request',
        'feedback': 'badge-feedback',
        'spam': 'badge-spam'
    }
    return f'<span class="{badges.get(category, "badge-spam")}">{category.capitalize()}</span>'

def get_urgency_badge(urgency):
    badges = {
        'high': 'urgency-high',
        'medium': 'urgency-medium',
        'low': 'urgency-low'
    }
    return f'<span class="{badges.get(urgency, "urgency-medium")}">{urgency.capitalize()} Priority</span>'

# Header
st.markdown('<div class="main-header">📧 AI Email Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise Customer Support Dashboard</div>', unsafe_allow_html=True)

# Sidebar - Classify New Email
with st.sidebar:
    st.header("📨 Classify New Email")
    
    subject = st.text_input("Email Subject", placeholder="Enter email subject...")
    body = st.text_area("Email Body", placeholder="Enter email body...", height=150)
    
    if st.button("🚀 Classify Email"):
        if subject and body:
            with st.spinner("Classifying..."):
                result = classify_email(subject, body)
                
                # Add to emails
                new_email = {
                    'id': len(st.session_state.emails) + 1,
                    'subject': subject,
                    'body': body,
                    'category': result['category'],
                    'urgency': result['urgency'],
                    'confidence_category': result['confidence_category'],
                    'confidence_urgency': result['confidence_urgency'],
                    'timestamp': datetime.now()
                }
                st.session_state.emails.insert(0, new_email)
                st.session_state.classification_result = result
                st.rerun()
        else:
            st.error("Please enter both subject and body!")
    
    # Show classification result
    if st.session_state.classification_result:
        result = st.session_state.classification_result
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success("✅ Classification Complete!")
        st.markdown(f"**Category:** {result['category'].capitalize()}")
        st.markdown(f"**Confidence:** {result['confidence_category']*100:.1f}%")
        st.markdown(f"**Urgency:** {result['urgency'].capitalize()}")
        st.markdown(f"**Confidence:** {result['confidence_urgency']*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Clear Result"):
            st.session_state.classification_result = None
            st.rerun()
    
    st.divider()
    
    # Filters
    st.header("🔍 Filters")
    category_filter = st.selectbox(
        "Category",
        ["All", "Complaint", "Request", "Feedback", "Spam"]
    )
    
    urgency_filter = st.selectbox(
        "Urgency",
        ["All", "High", "Medium", "Low"]
    )

# Main content
# Stats Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📥 Total Emails",
        value=len(st.session_state.emails)
    )

with col2:
    high_priority = len([e for e in st.session_state.emails if e['urgency'] == 'high'])
    st.metric(
        label="🔴 High Priority",
        value=high_priority
    )

with col3:
    complaints = len([e for e in st.session_state.emails if e['category'] == 'complaint'])
    st.metric(
        label="⚠️ Complaints",
        value=complaints
    )

with col4:
    st.metric(
        label="⏱️ Avg Response Time",
        value="2.4h"
    )

st.divider()

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Category Distribution")
    
    category_counts = pd.DataFrame([
        {'Category': 'Complaints', 'Count': len([e for e in st.session_state.emails if e['category'] == 'complaint'])},
        {'Category': 'Requests', 'Count': len([e for e in st.session_state.emails if e['category'] == 'request'])},
        {'Category': 'Feedback', 'Count': len([e for e in st.session_state.emails if e['category'] == 'feedback'])},
        {'Category': 'Spam', 'Count': len([e for e in st.session_state.emails if e['category'] == 'spam'])}
    ])
    
    fig_pie = px.pie(
        category_counts,
        values='Count',
        names='Category',
        color='Category',
        color_discrete_map={
            'Complaints': '#ef4444',
            'Requests': '#3b82f6',
            'Feedback': '#10b981',
            'Spam': '#6b7280'
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("📈 Urgency Levels")
    
    urgency_counts = pd.DataFrame([
        {'Urgency': 'High', 'Count': len([e for e in st.session_state.emails if e['urgency'] == 'high'])},
        {'Urgency': 'Medium', 'Count': len([e for e in st.session_state.emails if e['urgency'] == 'medium'])},
        {'Urgency': 'Low', 'Count': len([e for e in st.session_state.emails if e['urgency'] == 'low'])}
    ])
    
    fig_bar = px.bar(
        urgency_counts,
        x='Urgency',
        y='Count',
        color='Urgency',
        color_discrete_map={
            'High': '#dc2626',
            'Medium': '#f59e0b',
            'Low': '#10b981'
        }
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# Email List
st.subheader(f"📬 Classified Emails ({len(st.session_state.emails)})")

# Apply filters
filtered_emails = st.session_state.emails.copy()

if category_filter != "All":
    filtered_emails = [e for e in filtered_emails if e['category'] == category_filter.lower()]

if urgency_filter != "All":
    filtered_emails = [e for e in filtered_emails if e['urgency'] == urgency_filter.lower()]

# Display emails
for email in filtered_emails:
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"**{email['subject']}**")
            st.caption(email['body'][:100] + "..." if len(email['body']) > 100 else email['body'])
            
            st.markdown(
                f"{get_category_badge(email['category'])} "
                f"{get_urgency_badge(email['urgency'])} "
                f"<span style='color: #6b7280; font-size: 0.75rem;'>{email['confidence_category']*100:.0f}% confidence</span>",
                unsafe_allow_html=True
            )
        
        with col2:
            st.caption(format_timestamp(email['timestamp']))
        
        st.divider()

# Footer
st.markdown("---")
st.caption("🤖 Powered by Rule-Based Classification | Built with Streamlit")
