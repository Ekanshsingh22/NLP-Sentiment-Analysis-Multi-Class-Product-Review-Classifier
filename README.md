# 🧠 NLP Sentiment Analysis: Multi-Class Product Review Classifier

> **An end-to-end NLP pipeline that automatically classifies e-commerce product reviews into Positive, Negative, or Neutral sentiments — comparing 5 machine learning algorithms with full evaluation and deployment-ready output.**

---

## 📌 Problem Statement

E-commerce platforms receive **millions of product reviews daily**. Manual analysis is impossible at scale. This project builds an automated sentiment classifier that enables businesses to:

- 📊 Monitor brand reputation in real-time
- 🚨 Detect product quality issues early
- 📬 Prioritize customer support responses
- 📈 Build automated review insight dashboards

---

## 🏗️ Project Architecture

```
Raw Text Reviews
       │
       ▼
┌──────────────────┐
│  Text Cleaning   │  Lowercase, remove punctuation, normalize whitespace
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  TF-IDF (1+2gram)│  10,000 features, sublinear TF scaling
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  5 ML Models Trained & Compared         │
│  ├── Logistic Regression                │
│  ├── Naive Bayes                        │
│  ├── Linear SVM                         │
│  ├── Random Forest                      │
│  └── Gradient Boosting                  │
└────────┬────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Evaluation      │  Accuracy, F1-Score, Confusion Matrix, Feature Importance
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Saved Pipeline  │  best_model.pkl — ready for API deployment
└──────────────────┘
```

---

## 📁 Repository Structure

```
sentiment-analysis-nlp/
│
├── 📓 notebooks/
│   └── Sentiment_Analysis_NLP.ipynb   ← Main notebook (fully documented)
│
├── 📊 data/
│   └── product_reviews.csv            ← 1,500 labeled reviews dataset
│
├── 📈 outputs/
│   ├── eda_analysis.png               ← EDA visualizations
│   ├── model_evaluation.png           ← Model comparison charts
│   ├── feature_importance.png         ← Top predictive features per class
│   ├── model_results.csv              ← Tabular model comparison
│   └── best_model.pkl                 ← Serialized deployment-ready model
│
├── 🐍 pipeline.py                     ← Full Python script (runs everything)
├── requirements.txt                   ← All dependencies
└── README.md                          ← This file
```

---

## 📊 Dataset

| Field | Description |
|---|---|
| `review_id` | Unique identifier |
| `review_text` | Raw review text |
| `sentiment` | Label: positive / negative / neutral |
| `rating` | Star rating 1–5 |
| `product_category` | Electronics, Books, Clothing, etc. |

**1,500 reviews** across **8 product categories** with balanced class distribution.

---

## 🤖 Models & Results

| Model | Accuracy | F1 (Weighted) | Notes |
|---|---|---|---|
| **Logistic Regression** ⭐ | 1.0000 | 1.0000 | Best: fast + interpretable |
| Naive Bayes | 1.0000 | 1.0000 | Excellent baseline |
| Linear SVM | 1.0000 | 1.0000 | Robust for text |
| Random Forest | 1.0000 | 1.0000 | Handles non-linearity |
| Gradient Boosting | 1.0000 | 1.0000 | Sequential learning |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python pipeline.py
```

### 3. Open the Notebook

```bash
jupyter notebook notebooks/Sentiment_Analysis_NLP.ipynb
```

### 4. Use the Model Directly

```python
import pickle, re

# Load trained model
with open('outputs/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(text):
    text = re.sub(r'[^a-z\s]', '', text.lower()).strip()
    return model.predict([text])[0]

print(predict("This product is absolutely amazing!"))   # → positive
print(predict("Terrible quality, broke immediately."))  # → negative
print(predict("Works fine, nothing special."))          # → neutral
```

---

## 🔑 Key Technical Concepts

### TF-IDF with Bigrams
TF-IDF converts text to numerical vectors. Bigrams capture multi-word expressions:
- "not good" ≠ "not" + "good" separately
- "highly recommend" is a strong positive signal as a phrase

### Why Logistic Regression Wins for Text
- Linear models excel when features >> samples (10K features, 1.5K samples)
- Interpretable via coefficient weights
- Scales well, trains in milliseconds
- Regularization (C parameter) prevents overfitting

### Sklearn Pipelines
Using `Pipeline` ensures the same preprocessing is applied at both train and inference time — preventing data leakage and making deployment trivial.

---

## 📈 Visualizations

| Chart | Description |
|---|---|
| `eda_analysis.png` | Class distribution, rating histogram, review lengths by sentiment, per-category breakdown |
| `model_evaluation.png` | Confusion matrix (best model) + multi-model accuracy/F1 comparison |
| `feature_importance.png` | Top 20 TF-IDF feature coefficients for each sentiment class |

---

## 🔮 Future Work

- [ ] Fine-tune **DistilBERT** / **RoBERTa** for contextual embeddings
- [ ] Implement **aspect-based sentiment analysis** (e.g., "battery is terrible but display is great")
- [ ] Build **Flask REST API** with `/predict` endpoint
- [ ] Deploy on **Hugging Face Spaces** with Gradio UI
- [ ] Add **real-time stream processing** with Kafka integration

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---



