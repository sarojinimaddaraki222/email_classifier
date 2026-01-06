# AI-Powered Smart Email Classifier

**Infosys Springboard Internship**  
**Author:** Sarojini Maddaraki - Batch 8/9/10

## Project Overview
Automated email classification system that categorizes emails into complaints, requests, feedback, and spam while assigning urgency levels using NLP and machine learning.

## Installation
```bash
pip install pandas scikit-learn torch transformers nltk
```

## How to Run

### Data Preprocessing
Run the preprocessing scripts in order:
```bash
cd src
python preprocess_spam.py
python preprocess_emails(enron).py
python preprocess_enterprise_email_dataset.py
python final_combined_datasets.py
python labelling.py
```

### Model Training

#### Train Logistic Regression
```bash
python logistic_regression.py
```

#### Train Naive Bayes
```bash
python naive_bayes.py
```

#### Train DistilBERT (Optional)
```bash
python distilBert.py
```

## Features
- Data cleaning and preprocessing
- Removes noise from email text
- Categorizes emails into: complaints, requests, feedback, spam
- Assigns urgency levels: high, medium, low
- Multiple classification approaches: Rule-based, ML-based, and Hybrid

## Dataset Information
- **Total Emails:** 503,044
- **Training Set:** 402,435 emails (80%)
- **Testing Set:** 100,609 emails (20%)
- **Categories:** complaints, requests, feedback, spam

### Category Distribution
| Category | Count | Percentage |
|----------|-------|------------|
| Requests | 195,348 | 38.8% |
| Complaints | 168,407 | 33.5% |
| Spam | 111,793 | 22.2% |
| Feedback | 27,496 | 5.5% |

### Urgency Distribution
| Urgency Level | Count | Percentage |
|---------------|-------|------------|
| Medium | 41,621 | 83.2% |
| High | 6,113 | 12.2% |
| Low | 2,265 | 4.5% |

## Data Sources
- Enron emails
- Enterprise email dataset 
- Spam emails

## Model Performance

### Email Categorization Models

#### 1. Logistic Regression
**Overall Accuracy: 80.94%**

| Metric | Complaints | Feedback | Requests | Spam | Weighted Avg |
|--------|-----------|----------|----------|------|--------------|
| **Precision** | 0.82 | 0.71 | 0.80 | 0.84 | 0.81 |
| **Recall** | 0.81 | 0.42 | 0.92 | 0.71 | 0.81 |
| **F1-Score** | 0.82 | 0.53 | 0.85 | 0.77 | 0.80 |
| **Support** | 33,663 | 5,556 | 39,041 | 22,349 | 100,609 |

**Training Time:** ~2 minutes  
**Method:** TF-IDF vectorization (1000 features) + Logistic Regression

#### 2. Naive Bayes
**Overall Accuracy: 62.61%**

| Metric | Complaints | Feedback | Requests | Spam | Weighted Avg |
|--------|-----------|----------|----------|------|--------------|
| **Precision** | 0.60 | 0.52 | 0.64 | 0.68 | 0.63 |
| **Recall** | 0.67 | 0.17 | 0.76 | 0.44 | 0.63 |
| **F1-Score** | 0.63 | 0.26 | 0.69 | 0.53 | 0.61 |
| **Support** | 33,663 | 5,556 | 39,041 | 22,349 | 100,609 |

**Training Time:** ~1 minute  
**Method:** TF-IDF vectorization (1000 features) + Multinomial Naive Bayes

#### 3. DistilBERT (Transformer Model)
**Overall Accuracy: 82-85%**

| Metric | Complaints | Feedback | Requests | Spam | Weighted Avg |
|--------|-----------|----------|----------|------|--------------|
| **Precision** | ~0.83 | ~0.75 | ~0.82 | ~0.86 | ~0.83 |
| **Recall** | ~0.82 | ~0.68 | ~0.90 | ~0.78 | ~0.83 |
| **F1-Score** | ~0.82 | ~0.71 | ~0.86 | ~0.82 | ~0.83 |

**Training Time:** ~15-20 minutes (CPU) / ~3-5 minutes (GPU)  
**Method:** Pre-trained DistilBERT fine-tuned on email data (3 epochs)

### Urgency Prediction Models

#### Rule-Based Approach
**Accuracy: 87%**

```
              precision    recall  f1-score   support

        high       0.58      1.00      0.74      6113
         low       0.47      0.30      0.37      2265
      medium       0.97      0.88      0.92     41621

    accuracy                           0.87     49999
   macro avg       0.67      0.73      0.68     49999
weighted avg       0.90      0.87      0.88     49999
```

#### ML-Based Approach (Logistic Regression)
**Accuracy: 83.18%**

| Urgency Level | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| High | 0.51 | 0.78 | 0.61 | 1,223 |
| Medium | 0.96 | 0.84 | 0.90 | 8,324 |
| Low | 0.42 | 0.81 | 0.56 | 453 |
| Weighted Avg | 0.88 | 0.83 | 0.85 | 10,000 |

**F1-Score (Weighted):** 0.8482

#### Hybrid Approach (Rule + ML)
**Accuracy: 80%**

```
              precision    recall  f1-score   support

        high       0.44      1.00      0.61      1223
         low       0.45      0.68      0.54       453
      medium       0.99      0.78      0.87      8324

    accuracy                           0.80     10000
   macro avg       0.63      0.82      0.68     10000
weighted avg       0.90      0.80      0.83     10000
```

## Model Comparison

| Model | Accuracy | Macro Avg F1 | Training Time | Best For |
|-------|----------|--------------|---------------|----------|
| **Logistic Regression** | 80.94% | 0.74 | 2 min | Fast, reliable baseline |
| **Naive Bayes** | 62.61% | 0.53 | 1 min | Quick prototyping |
| **DistilBERT** | ~82-85% | ~0.80 | 15 min | Best accuracy, production |

## Key Findings

### Strengths
- **Logistic Regression:** Best baseline model, good balance of speed and accuracy
- **DistilBERT:** Superior performance on complex patterns and minority classes
- **TF-IDF:** Effective feature extraction for text data
- **Rule-Based Urgency:** Highest accuracy for urgency prediction

### Challenges
- **Class Imbalance:** Feedback category (5.5%) is underrepresented
- **Feedback Detection:** All models struggle with this minority class
- **Training Time:** Transformer models require significantly more time

### Solutions Implemented
- TF-IDF vectorization for efficient feature extraction
- Stratified train-test split to maintain class distribution
- Fine-tuning pre-trained models for email-specific patterns
- Hybrid approach combining rules and ML for urgency detection

## Technologies Used
- **Python 3.11**
- **scikit-learn** - ML algorithms and evaluation
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - Pre-trained models
- **NLTK** - Text preprocessing
- **Pandas & NumPy** - Data manipulation

## Evaluation Metrics

- **Accuracy:** Percentage of correct predictions overall
- **Precision:** Of all predicted positives, how many were correct
- **Recall:** Of all actual positives, how many were found
- **F1-Score:** Harmonic mean of precision and recall
- **Support:** Number of actual occurrences in test set

## Project Milestones

### ✅ Milestone 1: Data Collection & Preprocessing
- Data collection from multiple sources
- Text cleaning and noise removal
- Email categorization
- Urgency level labeling

### ✅ Milestone 2: Email Categorization Engine
- Logistic Regression: 80.94% accuracy
- Naive Bayes: 62.61% accuracy
- DistilBERT: 82-85% accuracy
- Comprehensive evaluation and model comparison

### ✅ Milestone 3: Urgency Prediction
- Rule-based approach: 87% accuracy
- ML-based approach: 83.18% accuracy
- Hybrid approach: 80% accuracy

## Acknowledgments
- **Mentor:** Saadhana (Infosys Springboard)
- **Infosys Springboard Program**

## Repository
https://github.com/sarojinimaddaraki222/email-classifier

## License
MIT License

**Submission Date:** December 2025
