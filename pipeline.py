"""
Sentiment Analysis Pipeline
Runs all ML models and saves outputs/plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re, os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

os.makedirs('/home/claude/sentiment_project/outputs', exist_ok=True)

# ── 1. Load & preview ──────────────────────────────────────────────────────
df = pd.read_csv('/home/claude/sentiment_project/data/product_reviews.csv')
print("Dataset Shape:", df.shape)
print(df.head())
print("\nClass Distribution:\n", df['sentiment'].value_counts())

# ── 2. Text Preprocessing ─────────────────────────────────────────────────
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['review_text'].apply(preprocess)
df['review_length'] = df['review_text'].apply(lambda x: len(str(x).split()))

# ── 3. EDA Plots ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Exploratory Data Analysis – Product Reviews Sentiment', fontsize=16, fontweight='bold')

# 3a. Sentiment distribution
colors = ['#2ECC71', '#E74C3C', '#3498DB']
counts = df['sentiment'].value_counts()
axes[0,0].bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.2)
axes[0,0].set_title('Sentiment Class Distribution', fontweight='bold')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0,0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# 3b. Rating distribution
df['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0,1], color='#9B59B6', edgecolor='white')
axes[0,1].set_title('Star Rating Distribution', fontweight='bold')
axes[0,1].set_xlabel('Rating'); axes[0,1].set_ylabel('Count')
axes[0,1].tick_params(axis='x', rotation=0)

# 3c. Review length by sentiment
for i, (sent, color) in enumerate(zip(['positive','negative','neutral'], colors)):
    data = df[df['sentiment']==sent]['review_length']
    axes[1,0].hist(data, bins=15, alpha=0.6, label=sent, color=color)
axes[1,0].set_title('Review Length Distribution by Sentiment', fontweight='bold')
axes[1,0].set_xlabel('Word Count'); axes[1,0].legend()

# 3d. Category breakdown
cat_sent = df.groupby(['product_category','sentiment']).size().unstack(fill_value=0)
cat_sent.plot(kind='bar', ax=axes[1,1], color=colors, edgecolor='white')
axes[1,1].set_title('Sentiment by Product Category', fontweight='bold')
axes[1,1].set_xlabel(''); axes[1,1].legend(title='Sentiment')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/home/claude/sentiment_project/outputs/eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ EDA plot saved")

# ── 4. Model Training ─────────────────────────────────────────────────────
X = df['clean_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('clf', MultinomialNB(alpha=0.1))
    ]),
    'Linear SVM': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf', LinearSVC(max_iter=2000, C=0.5, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    'Gradient Boosting': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
}

results = {}
print("\n── Model Training Results ──")
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'accuracy': acc, 'f1_weighted': f1, 'pipeline': pipeline, 'y_pred': y_pred}
    print(f"  {name:25s} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

# ── 5. Best model detailed report ─────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1_weighted'])
best = results[best_name]
print(f"\n🏆 Best Model: {best_name}")
print(classification_report(y_test, best['y_pred']))

# ── 6. Confusion Matrix ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Best Model: {best_name} – Detailed Evaluation', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, best['y_pred'], labels=['positive','neutral','negative'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['positive','neutral','negative'],
            yticklabels=['positive','neutral','negative'], ax=axes[0])
axes[0].set_title('Confusion Matrix'); axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

# Model comparison bar chart
model_names = list(results.keys())
accuracies  = [results[m]['accuracy'] for m in model_names]
f1_scores   = [results[m]['f1_weighted'] for m in model_names]
x = np.arange(len(model_names))
w = 0.35
axes[1].bar(x - w/2, accuracies, w, label='Accuracy', color='#3498DB', alpha=0.8)
axes[1].bar(x + w/2, f1_scores,  w, label='F1-Score',  color='#E74C3C', alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
axes[1].set_ylim(0, 1.1); axes[1].legend()
axes[1].set_title('Model Comparison'); axes[1].set_ylabel('Score')
for i, (a, f) in enumerate(zip(accuracies, f1_scores)):
    axes[1].text(i-w/2, a+0.01, f'{a:.3f}', ha='center', fontsize=7)
    axes[1].text(i+w/2, f+0.01, f'{f:.3f}', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig('/home/claude/sentiment_project/outputs/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Evaluation plot saved")

# ── 7. Feature Importance (TF-IDF top terms) ──────────────────────────────
lr_pipeline = results['Logistic Regression']['pipeline']
vectorizer  = lr_pipeline.named_steps['tfidf']
clf         = lr_pipeline.named_steps['clf']
feature_names = vectorizer.get_feature_names_out()
classes = clf.classes_

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Top 20 Predictive Features per Sentiment Class (Logistic Regression)', fontsize=13, fontweight='bold')

palette = {'positive': '#2ECC71', 'neutral': '#3498DB', 'negative': '#E74C3C'}
for ax, cls in zip(axes, ['positive', 'neutral', 'negative']):
    idx = list(classes).index(cls)
    coefs = clf.coef_[idx]
    top20_idx = np.argsort(coefs)[-20:]
    top20_words = [feature_names[i] for i in top20_idx]
    top20_vals  = coefs[top20_idx]
    ax.barh(top20_words, top20_vals, color=palette[cls], alpha=0.85)
    ax.set_title(f'{cls.title()} Class', fontweight='bold')
    ax.set_xlabel('Coefficient Weight')

plt.tight_layout()
plt.savefig('/home/claude/sentiment_project/outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Feature importance plot saved")

# ── 8. Save results CSV ────────────────────────────────────────────────────
results_df = pd.DataFrame([
    {'Model': m, 'Accuracy': results[m]['accuracy'], 'F1_Weighted': results[m]['f1_weighted']}
    for m in results
]).sort_values('F1_Weighted', ascending=False)
results_df.to_csv('/home/claude/sentiment_project/outputs/model_results.csv', index=False)
print("✅ Results CSV saved")
print("\n── Final Leaderboard ──")
print(results_df.to_string(index=False))

# ── 9. Inference demo ─────────────────────────────────────────────────────
best_pipeline = results[best_name]['pipeline']
test_reviews = [
    "This product is absolutely fantastic, best purchase ever!",
    "Terrible quality, stopped working after two days. Very disappointed.",
    "It arrived on time and works okay, nothing special.",
    "Incredible value for money, I am extremely happy!",
    "Waste of money, completely broken out of the box.",
]
preds = best_pipeline.predict([preprocess(t) for t in test_reviews])
print("\n── Live Inference Demo ──")
for rev, pred in zip(test_reviews, preds):
    emoji = '😊' if pred=='positive' else ('😞' if pred=='negative' else '😐')
    print(f"  {emoji} [{pred.upper():8s}] {rev[:60]}...")
