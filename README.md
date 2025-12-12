# AI-Powered Smart Email Classifier

**Infosys Springboard Internship - Milestone 1**

## Description
This project automatically categorizes emails and assigns urgency levels using keyword-based text classification.

## Installation
```bash
pip install pandas
```

## How to Run

Run the preprocessing scripts in order:

```bash
cd src
python preprocess_spam.py
python preprocess_enron.py
python preprocess_support.py
python combine_datasets.py
python labelling.py
```

## Features
- Data cleaning and preprocessing
- Removes noise from email text
- Categorizes emails into: spam, work, personal, support, finance, general
- Assigns urgency levels: high, medium, low

## Datasets
- Enron emails
- Support emails
- Spam emails

## Output
Final labeled dataset with categories and urgency levels

## Milestone 1 Tasks Completed
 Data collection  
 Text cleaning and noise removal  
 Email categorization  
 Urgency level labeling

## Author
Sarojini Maddaraki - Batch 8/9/10

## License
MIT License
