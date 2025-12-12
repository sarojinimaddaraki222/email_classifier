AI-Powered Smart Email Classifier
Infosys Springboard Internship - Milestone 1
Sarojini Maddaraki - Batch 8/9/10
Overview:
Email classification system that automatically categorizes emails and assigns urgency levels.

Project Structure:
email_classifier/
├── data/
│   ├── raw/
│   │   ├── enron.csv
│   │   ├── enterprice_email_dataset.csv
│   │   └── spam.csv
│   └── cleaned/
│       ├── spam_cleaned.csv
│       ├── enron_cleaned.csv
│       ├── support_cleaned.csv
│       ├── combined_emails.csv
│       └── labeled_emails.csv
├── src/
│   ├── preprocess_spam.py
│   ├── preprocess_enron.py
│   ├── preprocess_support.py
│   ├── combine_datasets.py
│   ├── labelling.py
│
├── requirements.txt
├── LICENSE
└── README.md

Installation:
pip install pandas

Output:
Final dataset: data/cleaned/labeled_emails.csv
Contains:
cleaned_text - Cleaned email content
category - Email category (spam, work, personal, support, finance, general)
urgency - Urgency level (high, medium, low)
source - DATASET_SOURCE.md(It has drive link to download dataset)
