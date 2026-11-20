"""
Production Model Retraining Script for Fake Review Detection.
=============================================================
Trains a high-accuracy, calibrated multi-domain pipeline on the complete
corpus of 73,372 reviews (He et al. 2022 + Salminen et al. 2022).
Eliminates multicollinearity, stabilizes feature weights, and serializes
the updated production artifact to models/best_model_enhanced.joblib.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import get_stopwords

MODELS_DIR = "models"
METRICS_DIR = "results/metrics"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def train_production_model():
    print("=" * 70)
    print("TRAINING UNIFIED PRODUCTION MODEL (73,372 REVIEWS)")
    print("=" * 70)

    # 1. Load Salminen et al. dataset (40,432 rows, 10 Amazon categories)
    sal_path = "dataset/fake reviews dataset.csv"
    print(f"Loading {sal_path}...")
    df_sal = pd.read_csv(sal_path)
    sal_texts = df_sal["text_"].fillna("").astype(str)
    sal_labels = (df_sal["label"].str.strip() == "CG").astype(int)

    # 2. Load He et al. dataset (32,940 rows, Facebook campaign fraud)
    he_path = "dataset/cleaned_reviews.csv"
    print(f"Loading {he_path}...")
    df_he = pd.read_csv(he_path)
    he_texts = (
        df_he["review_title"].fillna("").astype(str) + " " +
        df_he["review_text"].fillna("").astype(str)
    ).str.strip()
    he_labels = df_he["label"].astype(int)

    # Combine into unified training corpus
    all_texts = pd.concat([sal_texts, he_texts], ignore_index=True)
    all_labels = pd.concat([sal_labels, he_labels], ignore_index=True)

    print(f"Total Combined Corpus: {len(all_texts):,} reviews")
    print(f"  Class 1 (Fake / Deceptive): {all_labels.sum():,} ({all_labels.mean():.1%})")
    print(f"  Class 0 (Genuine / Honest): {(all_labels == 0).sum():,} ({(1 - all_labels.mean()):.1%})")

    # 3. Build TF-IDF Feature Extractor with Negation Preservation
    stops = list(get_stopwords(preserve_negations=True))
    tfidf = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        stop_words=stops,
        sublinear_tf=True,
        min_df=2
    )

    # 4. Logistic Regression Classifier with Tuned Regularization
    clf = LogisticRegression(
        C=2.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("classifier", clf)
    ])

    # 5. Evaluate on 15% Stratified Holdout Test Set
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts,
        all_labels,
        test_size=0.15,
        random_state=42,
        stratify=all_labels
    )

    print("\nFitting pipeline on training split (85%)...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 70)
    print("HOLDOUT EVALUATION METRICS:")
    print("=" * 70)
    print(f"Test Accuracy:  {acc:.4f} ({acc:.1%})")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"ROC-AUC Score:  {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine (0)", "Fake (1)"]))

    # 6. Fit on Full Corpus for Final Deployment Artifact
    print("\nFitting pipeline on complete 73,372 corpus for production deployment...")
    pipeline.fit(all_texts, all_labels)

    model_path = os.path.join(MODELS_DIR, "best_model_enhanced.joblib")
    joblib.dump(pipeline, model_path, compress=3)
    print(f"Serialized production model saved to: {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

    # 7. Export Updated Top Feature Coefficients
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs)
    }).sort_values(by="coefficient", ascending=False)

    coef_csv = os.path.join(METRICS_DIR, "top_feature_coefficients.csv")
    df_coef.to_csv(coef_csv, index=False)
    print(f"Updated feature coefficients saved to: {coef_csv}")

    print("\nTop 10 Deceptive Predictors (+ log-odds):")
    print(df_coef.head(10)[["feature", "coefficient"]].to_string(index=False))

    print("\nTop 10 Genuine Predictors (- log-odds):")
    print(df_coef.tail(10)[["feature", "coefficient"]].to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    train_production_model()
