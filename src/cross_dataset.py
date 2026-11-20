"""
Cross-Dataset Generalization and Domain Adaptation Module.
==========================================================
Evaluates models trained on human-incentivized fake reviews (He et al.) against
an external benchmark of AI-generated reviews (Salminen et al., 40,432 reviews),
quantifying domain shift and performance transfer across 10 Amazon categories.
"""

import os
import sys
import joblib
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_salminen_dataset
from src.evaluate import evaluate_classifier

plt.switch_backend("Agg")
sns.set_theme(style="whitegrid")

FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def evaluate_cross_dataset_transfer(
    pipeline: Any,
    sample_size: Optional[int] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluates a model trained on He et al. against the Salminen et al. corpus.
    
    Returns:
        (overall_metrics_dict, category_performance_dataframe)
    """
    print("Loading external benchmark: Salminen et al. (40,432 reviews)...")
    df_salminen = load_salminen_dataset(sample_size=sample_size)

    X_ext = df_salminen["review_text"]
    y_ext = df_salminen["label"]

    print("Evaluating model on full external dataset...")
    overall_eval = evaluate_classifier(pipeline, X_ext, y_ext, model_name="Transfer_Salminen_Full")

    # Evaluate performance broken down by each of the 10 product categories
    categories = df_salminen["product_category"].unique()
    cat_records = []

    print("Evaluating performance across 10 product categories...")
    for cat in sorted(categories):
        subset = df_salminen[df_salminen["product_category"] == cat]
        X_cat = subset["review_text"]
        y_cat = subset["label"]

        cat_eval = evaluate_classifier(pipeline, X_cat, y_cat, model_name=f"Transfer_{cat}")
        clean_cat_name = cat.replace("_5", "").replace("_", " ")
        cat_records.append({
            "category": clean_cat_name,
            "reviews_count": len(subset),
            "accuracy": cat_eval["accuracy"],
            "macro_f1": cat_eval["macro_f1"],
            "precision_fake": cat_eval["precision_fake"],
            "recall_fake": cat_eval["recall_fake"],
            "roc_auc": cat_eval["roc_auc"],
            "pr_auc": cat_eval["pr_auc"]
        })

    cat_df = pd.DataFrame(cat_records).sort_values(by="macro_f1", ascending=False)

    # Save metrics to CSV
    transfer_summary = pd.DataFrame([{
        "benchmark": "Salminen et al. (40,432 reviews)",
        "accuracy": overall_eval["accuracy"],
        "macro_f1": overall_eval["macro_f1"],
        "precision_fake": overall_eval["precision_fake"],
        "recall_fake": overall_eval["recall_fake"],
        "precision_honest": overall_eval["precision_honest"],
        "recall_honest": overall_eval["recall_honest"],
        "roc_auc": overall_eval["roc_auc"],
        "pr_auc": overall_eval["pr_auc"]
    }])
    transfer_summary_path = os.path.join(METRICS_DIR, "cross_dataset_transfer.csv")
    transfer_summary.to_csv(transfer_summary_path, index=False)
    print(f"Saved Overall Transfer Metrics to: {transfer_summary_path}")

    cat_csv_path = os.path.join(METRICS_DIR, "cross_dataset_category_performance.csv")
    cat_df.to_csv(cat_csv_path, index=False)
    print(f"Saved Category Performance Table to: {cat_csv_path}")

    return overall_eval, cat_df


def plot_category_transfer_performance(cat_df: pd.DataFrame):
    """Figure 13: Horizontal bar chart showing Cross-Dataset Accuracy and Macro F1 across 10 categories."""
    fig, ax = plt.subplots(figsize=(12, 6))

    y_indices = np.arange(len(cat_df))
    bar_height = 0.38

    ax.barh(y_indices - bar_height/2, cat_df["accuracy"] * 100, height=bar_height, color="#2b5c8f", label="Accuracy (%)", edgecolor="black")
    ax.barh(y_indices + bar_height/2, cat_df["macro_f1"] * 100, height=bar_height, color="#d95f02", label="Macro F1 (%)", edgecolor="black")

    ax.set_yticks(y_indices)
    ax.set_yticklabels(cat_df["category"], fontweight="bold")
    ax.set_xlabel("Performance Score (%)")
    ax.set_title("Cross-Dataset Generalization Across 10 Amazon Product Categories\n(Trained on He et al. Human Fraud -> Tested on Salminen et al. AI Reviews)", fontweight="bold")
    ax.set_xlim(30, 90)
    ax.legend(loc="lower right")

    # Add score labels
    for i, (_, row) in enumerate(cat_df.iterrows()):
        ax.text(row["accuracy"] * 100 + 0.8, i - bar_height/2, f"{row['accuracy']:.1%}", va="center", fontsize=9, fontweight="bold", color="#2b5c8f")
        ax.text(row["macro_f1"] * 100 + 0.8, i + bar_height/2, f"{row['macro_f1']:.1%}", va="center", fontsize=9, fontweight="bold", color="#d95f02")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "13_cross_domain_category_performance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Category Transfer Figure: {out_path}")


def main():
    model_path = "models/best_model_enhanced.joblib"
    if not os.path.exists(model_path):
        model_path = "models/best_model.joblib"

    print(f"Loading best trained model for cross-dataset evaluation: {model_path}")
    pipeline = joblib.load(model_path)

    overall_eval, cat_df = evaluate_cross_dataset_transfer(pipeline)
    plot_category_transfer_performance(cat_df)

    print("\n" + "=" * 65)
    print("CROSS-DATASET GENERALIZATION EVALUATION COMPLETE:")
    print("=" * 65)
    print(f"Overall External Accuracy: {overall_eval['accuracy']:.2%}")
    print(f"Overall External Macro F1: {overall_eval['macro_f1']:.2%}")
    print(f"Overall ROC-AUC:          {overall_eval['roc_auc']}")
    print(f"Overall PR-AUC:           {overall_eval['pr_auc']}")
    print("\nBreakdown by Product Category:")
    print(cat_df[["category", "reviews_count", "accuracy", "macro_f1", "precision_fake", "recall_fake"]].to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
