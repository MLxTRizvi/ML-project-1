"""
Model Training, Cross-Validation, and Feature Ablation Experiment Runner.
========================================================================
Implements leak-free training across 5 classical Machine Learning algorithms,
evaluates Pipeline A (Baseline TF-IDF) vs Pipeline B (Enhanced Multi-Modal),
performs threshold optimization, and exports publication artifacts.
"""

import os
import sys
import time
import json
import joblib
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, precision_recall_curve

from src.data_loader import load_cleaned_primary_dataset
from src.feature_engineering import (
    build_baseline_feature_pipeline,
    build_enhanced_feature_pipeline
)
from src.evaluate import evaluate_classifier, compute_threshold_metrics

# Configure headless plotting
plt.switch_backend("Agg")
sns.set_theme(style="whitegrid")

MODELS_DIR = "models"
FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def load_and_split_data(test_size: float = 0.20, random_state: int = 42):
    """
    Loads primary dataset and performs reviewer-group splitting to prevent data leakage.
    Ensures exact comparability with the historical baseline.
    """
    df = load_cleaned_primary_dataset()

    # Combine review title and review text
    df["combined_text"] = (
        df["review_title"].fillna("").astype(str) + " " +
        df["review_text"].fillna("").astype(str)
    ).str.strip()

    X = df["combined_text"]
    y = df["label"]
    groups = df["reviewer_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_reviewers = set(groups.iloc[train_idx])
    test_reviewers = set(groups.iloc[test_idx])
    assert train_reviewers.isdisjoint(test_reviewers), "Data leakage! Reviewer overlap detected."

    print("=" * 60)
    print("DATA SPLIT SUMMARY (Leak-Free GroupShuffleSplit)")
    print("=" * 60)
    print(f"Total Dataset:    {len(df):,} reviews")
    print(f"Training Set:     {len(X_train):,} reviews (Reviewers: {len(train_reviewers):,})")
    print(f"Testing Set:      {len(X_test):,} reviews (Reviewers: {len(test_reviewers):,})")
    print(f"Reviewer Overlap: {len(train_reviewers & test_reviewers)} (Guaranteed 0)")
    print(f"Train Class Distribution: Fake {y_train.mean():.1%}, Honest {(1 - y_train.mean()):.1%}")
    print(f"Test Class Distribution:  Fake {y_test.mean():.1%}, Honest {(1 - y_test.mean()):.1%}")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


def get_candidate_models() -> Dict[str, Any]:
    """Defines the 5 classical ML algorithms for experimental comparison."""
    return {
        "Multinomial Naive Bayes (Smoothed)": MultinomialNB(alpha=0.5, fit_prior=True),
        "Logistic Regression (Balanced)": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Linear SVM (Balanced)": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest (Depth 25)": RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "SGD Log-Loss Classifier": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            random_state=42
        )
    }


def run_pipeline_a_benchmark(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates 5 candidate algorithms using Pipeline A (Baseline TF-IDF).
    """
    print("\n>>> RUNNING PIPELINE A: BASELINE TF-IDF (5 ALGORITHMS) <<<")
    models = get_candidate_models()
    results = []
    trained_pipelines = {}

    for name, clf in models.items():
        print(f"\n[Pipeline A] Training: {name}...")
        pipeline = Pipeline([
            ("features", build_baseline_feature_pipeline(max_features=50000, ngram_range=(1, 2))),
            ("classifier", clf)
        ])

        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - start_time

        eval_res = evaluate_classifier(pipeline, X_test, y_test, model_name=f"PipeA_{name}", train_time_sec=train_time)
        results.append(eval_res)
        trained_pipelines[name] = pipeline

        print(f"   Accuracy: {eval_res['accuracy']:.4f} | Macro F1: {eval_res['macro_f1']:.4f} | "
              f"Fake Precision: {eval_res['precision_fake']:.4f} | Fake Recall: {eval_res['recall_fake']:.4f} | "
              f"Train Time: {train_time:.2f}s")

    return results, trained_pipelines


def run_pipeline_b_benchmark(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates the top models using Pipeline B (Enhanced Multi-Modal).
    """
    print("\n>>> RUNNING PIPELINE B: ENHANCED MULTI-MODAL (TF-IDF + LINGUISTIC + VADER) <<<")
    # Evaluate top linear and probabilistic classifiers with multi-modal features
    models = {
        "Logistic Regression (Balanced)": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Linear SVM (Balanced)": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42
        ),
        "SGD Log-Loss Classifier": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            random_state=42
        )
    }

    results = []
    trained_pipelines = {}

    for name, clf in models.items():
        print(f"\n[Pipeline B] Training: {name} (Multi-Modal)...")
        pipeline = Pipeline([
            ("features", build_enhanced_feature_pipeline(
                word_max_features=40000,
                char_max_features=15000,
                word_ngram_range=(1, 2),
                char_ngram_range=(2, 5)
            )),
            ("classifier", clf)
        ])

        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - start_time

        eval_res = evaluate_classifier(pipeline, X_test, y_test, model_name=f"PipeB_{name}", train_time_sec=train_time)
        results.append(eval_res)
        trained_pipelines[name] = pipeline

        print(f"   Accuracy: {eval_res['accuracy']:.4f} | Macro F1: {eval_res['macro_f1']:.4f} | "
              f"Fake Precision: {eval_res['precision_fake']:.4f} | Fake Recall: {eval_res['recall_fake']:.4f} | "
              f"Train Time: {train_time:.2f}s")

    return results, trained_pipelines


def plot_roc_and_pr_curves(eval_results, y_test):
    """Figure 9: Generates ROC Curves and Precision-Recall Curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#2b5c8f", "#d95f02", "#2ca02c", "#7570b3", "#e7298a"]
    c_idx = 0

    for res in eval_results:
        y_scores = res.get("y_scores")
        if y_scores is None:
            continue
        name = res["model_name"].replace("PipeA_", "A: ").replace("PipeB_", "B: ")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc_val = res["roc_auc"]
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val})", color=colors[c_idx % len(colors)], lw=2)

        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_test, y_scores)
        pr_val = res["pr_auc"]
        axes[1].plot(rec, prec, label=f"{name} (AP = {pr_val})", color=colors[c_idx % len(colors)], lw=2)
        c_idx += 1

    # Format ROC Plot
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (0.50)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right", fontsize=9)

    # Format PR Plot
    baseline_fake_ratio = y_test.mean()
    axes[1].axhline(y=baseline_fake_ratio, color="k", linestyle="--", lw=1.5, label=f"Baseline ({baseline_fake_ratio:.2f})")
    axes[1].set_title("Precision-Recall (PR) Curves", fontweight="bold")
    axes[1].set_xlabel("Recall (Fake Class)")
    axes[1].set_ylabel("Precision (Fake Class)")
    axes[1].legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "09_roc_pr_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\nSaved ROC/PR Curves: {out_path}")


def plot_confusion_matrices(eval_results):
    """Figure 10: Confusion Matrix Grid for candidate models."""
    n_models = min(6, len(eval_results))
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, res in enumerate(eval_results[:n_models]):
        cm_dict = res["confusion_matrix"]
        cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
        name = res["model_name"].replace("PipeA_", "A: ").replace("PipeB_", "B: ")

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Honest (0)", "Fake (1)"],
            yticklabels=["Honest (0)", "Fake (1)"],
            cbar=False, ax=axes[i]
        )
        axes[i].set_title(f"{name}\nAcc: {res['accuracy']:.1%} | Macro F1: {res['macro_f1']:.1%}", fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    # Hide unused subplots
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "10_confusion_matrices_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrices Grid: {out_path}")


def plot_threshold_curve(th_df: pd.DataFrame, model_name: str = "Logistic Regression"):
    """Figure 11: Threshold tuning curve showing Precision vs Recall trade-offs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(th_df["threshold"], th_df["precision_fake"], marker="o", lw=2, label="Precision (Fake)", color="#2b5c8f")
    ax.plot(th_df["threshold"], th_df["recall_fake"], marker="s", lw=2, label="Recall (Fake)", color="#d95f02")
    ax.plot(th_df["threshold"], th_df["macro_f1"], marker="^", lw=2, label="Macro F1", color="#2ca02c")
    ax.plot(th_df["threshold"], th_df["accuracy"], marker="d", lw=2, label="Accuracy", color="#7570b3")

    ax.axvline(x=0.50, color="gray", linestyle="--", label="Default Threshold (0.50)")
    ax.set_title(f"Classification Metrics vs Decision Threshold ({model_name})", fontweight="bold")
    ax.set_xlabel("Decision Threshold (Probability Cutoff for Fake Class)")
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower left")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "11_threshold_tuning_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Threshold Tuning Curve: {out_path}")


def main():
    print("=" * 70)
    print("FAIR EXPERIMENTAL COMPARISON & LEAK-FREE MODEL TRAINING")
    print("=" * 70)

    # 1. Load Data with Group Splitting
    X_train, X_test, y_train, y_test = load_and_split_data()

    # 2. Benchmark Pipeline A (Baseline TF-IDF)
    results_a, pipes_a = run_pipeline_a_benchmark(X_train, X_test, y_train, y_test)

    # 3. Benchmark Pipeline B (Enhanced Multi-Modal)
    results_b, pipes_b = run_pipeline_b_benchmark(X_train, X_test, y_train, y_test)

    all_results = results_a + results_b

    # 4. Export Consolidated Comparison Table
    table_rows = []
    for r in all_results:
        table_rows.append({
            "Pipeline": "Enhanced (B)" if "PipeB_" in r["model_name"] else "Baseline (A)",
            "Model": r["model_name"].replace("PipeA_", "").replace("PipeB_", ""),
            "Accuracy": r["accuracy"],
            "Macro_F1": r["macro_f1"],
            "Precision_Fake": r["precision_fake"],
            "Recall_Fake": r["recall_fake"],
            "F1_Fake": r["f1_fake"],
            "Precision_Honest": r["precision_honest"],
            "Recall_Honest": r["recall_honest"],
            "F1_Honest": r["f1_honest"],
            "ROC_AUC": r["roc_auc"],
            "PR_AUC": r["pr_auc"],
            "Train_Time_Sec": r["train_time_sec"],
            "Latency_ms_per_100": r["pred_latency_ms_per_100"],
            "Model_Size_MB": r["model_size_mb"]
        })

    comparison_df = pd.DataFrame(table_rows).sort_values(by="Macro_F1", ascending=False)
    comp_csv_path = os.path.join(METRICS_DIR, "model_comparison_comprehensive.csv")
    comparison_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved Comprehensive Comparison Table: {comp_csv_path}")

    # 5. Direct Feature Ablation Table (Comparing Pipeline A vs Pipeline B for Logistic Regression)
    ablation_rows = [
        r for r in table_rows if "Logistic Regression" in r["Model"]
    ]
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_csv_path = os.path.join(METRICS_DIR, "feature_ablation_comparison.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"Saved Feature Ablation Table: {ablation_csv_path}")

    # 6. Generate Figures
    plot_roc_and_pr_curves(all_results, y_test)
    plot_confusion_matrices(all_results)

    # 7. Threshold Tuning for the Best Calibrated Model
    best_pipe = pipes_b.get("Logistic Regression (Balanced)") or pipes_a.get("Logistic Regression (Balanced)")
    best_eval = [r for r in all_results if "Logistic Regression" in r["model_name"] and "PipeB_" in r["model_name"]][0]
    
    if best_eval["y_scores"] is not None:
        th_df = compute_threshold_metrics(y_test, best_eval["y_scores"])
        th_csv_path = os.path.join(METRICS_DIR, "threshold_tuning.csv")
        th_df.to_csv(th_csv_path, index=False)
        plot_threshold_curve(th_df, "Enhanced Logistic Regression")

    # 8. Save the Best Retrained Model Artifact
    best_model_path = os.path.join(MODELS_DIR, "best_model_enhanced.joblib")
    joblib.dump(best_pipe, best_model_path, compress=3)
    print(f"\nSaved Best Enhanced Model Artifact: {best_model_path}")

    print("\n" + "=" * 70)
    print("CONSOLIDATED EXPERIMENTAL RESULTS:")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
