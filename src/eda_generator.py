"""
Exploratory Data Analysis (EDA) Generator.
==========================================
Generates publication-quality charts and quantitative metrics comparing
genuine vs fake reviews across both the primary (He et al.) and
secondary (Salminen et al.) datasets.

Output plots are saved under results/figures/ (300 DPI).
Summary metrics are saved under results/metrics/eda_summary.json.
"""

import os
import sys
import json
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from src.data_loader import load_cleaned_primary_dataset, load_salminen_dataset
from src.preprocessing import clean_text_standard, extract_linguistic_features

# Configure Matplotlib for Headless High-Resolution Output
plt.switch_backend("Agg")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 14
})

FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def plot_class_distributions(df_primary: pd.DataFrame, df_salminen: pd.DataFrame):
    """Figure 1: Class distribution comparison across primary and secondary datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Primary Dataset
    primary_counts = df_primary["label"].value_counts().sort_index()
    primary_labels = ["Genuine (0)", "Fake (1)"]
    colors = ["#2b5c8f", "#d95f02"]

    bars1 = axes[0].bar(primary_labels, primary_counts.values, color=colors, width=0.55, edgecolor="black")
    axes[0].set_title("Primary Dataset (He et al., 2022)\nFacebook Campaign Enriched", fontweight="bold")
    axes[0].set_ylabel("Number of Reviews")
    axes[0].set_ylim(0, max(primary_counts.values) * 1.15)
    for bar in bars1:
        yval = bar.get_height()
        pct = (yval / len(df_primary)) * 100
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 500, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")

    # Secondary Dataset
    salminen_counts = df_salminen["label"].value_counts().sort_index()
    bars2 = axes[1].bar(primary_labels, salminen_counts.values, color=colors, width=0.55, edgecolor="black")
    axes[1].set_title("Secondary Dataset (Salminen et al., 2022)\nGPT-2 Synthetic vs Human", fontweight="bold")
    axes[1].set_ylabel("Number of Reviews")
    axes[1].set_ylim(0, max(salminen_counts.values) * 1.15)
    for bar in bars2:
        yval = bar.get_height()
        pct = (yval / len(df_salminen)) * 100
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 500, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "01_class_distribution_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_review_length_distribution(df: pd.DataFrame):
    """Figure 2: Review length (word and character count) distributions by authenticity."""
    df["word_count"] = df["review_text"].apply(lambda t: len(str(t).split()))
    df["char_count"] = df["review_text"].apply(lambda t: len(str(t)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Word Count KDE / Histogram (capped at 99th percentile for clean display)
    cap_words = np.percentile(df["word_count"], 98)
    filtered = df[df["word_count"] <= cap_words]

    sns.histplot(
        data=filtered,
        x="word_count",
        hue="label",
        element="step",
        stat="density",
        common_norm=False,
        palette={0: "#2b5c8f", 1: "#d95f02"},
        ax=axes[0],
        bins=40,
        alpha=0.3
    )
    axes[0].set_title("Word Count Distribution by Class", fontweight="bold")
    axes[0].set_xlabel("Word Count per Review")
    axes[0].legend(title="Label", labels=["Fake (1)", "Genuine (0)"])

    # Character Count KDE
    cap_chars = np.percentile(df["char_count"], 98)
    filtered_chars = df[df["char_count"] <= cap_chars]
    sns.kdeplot(
        data=filtered_chars,
        x="char_count",
        hue="label",
        common_norm=False,
        palette={0: "#2b5c8f", 1: "#d95f02"},
        ax=axes[1],
        fill=True,
        alpha=0.25
    )
    axes[1].set_title("Character Count Density by Class", fontweight="bold")
    axes[1].set_xlabel("Character Count per Review")
    axes[1].legend(title="Label", labels=["Fake (1)", "Genuine (0)"])

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "02_review_length_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_review_length_boxplots(df: pd.DataFrame):
    """Figure 3: Review length boxplots with log-scale representation."""
    df["word_count"] = df["review_text"].apply(lambda t: len(str(t).split()))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x="label",
        y="word_count",
        hue="label",
        palette=["#2b5c8f", "#d95f02"],
        showfliers=False,
        width=0.45,
        legend=False,
        ax=ax
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Genuine Reviews (0)", "Fake Reviews (1)"], fontweight="bold")
    ax.set_ylabel("Word Count (Outliers Omitted for Clarity)")
    ax.set_title("Review Length Comparison: Genuine vs Fake", fontweight="bold")

    # Annotate Median and IQR values
    for i, label_val in enumerate([0, 1]):
        subset = df[df["label"] == label_val]["word_count"]
        median_val = subset.median()
        q25 = subset.quantile(0.25)
        q75 = subset.quantile(0.75)
        ax.text(
            i, median_val + 2,
            f"Median: {median_val:.0f}\nIQR: [{q25:.0f}, {q75:.0f}]",
            ha="center", va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
        )

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "03_review_length_boxplots.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_rating_distribution_by_label(df_salminen: pd.DataFrame):
    """Figure 4: Rating distribution across genuine vs fake reviews (from Salminen dataset)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    cross = pd.crosstab(
        df_salminen["rating"],
        df_salminen["label"],
        normalize="columns"
    ) * 100
    cross.columns = ["Genuine (OR)", "Fake (CG)"]

    cross.plot(kind="bar", ax=ax, color=["#2b5c8f", "#d95f02"], edgecolor="black", width=0.7)
    ax.set_title("Star Rating Distribution by Authenticity (Salminen et al.)", fontweight="bold")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Percentage within Class (%)")
    ax.set_xticklabels([f"{int(r)} Star" for r in cross.index], rotation=0)
    ax.legend(title="Review Class")

    for p in ax.patches:
        height = p.get_height()
        if height > 1.0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2., height + 0.8),
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    ax.set_ylim(0, 75)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "04_rating_distribution_by_label.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_top_ngrams_genuine_vs_fake(df: pd.DataFrame, top_n: int = 15):
    """Figure 5: Side-by-side comparison of top bigrams in genuine vs fake reviews."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, (label_val, title, color) in enumerate([
        (0, "Genuine Reviews (Honest)", "#2b5c8f"),
        (1, "Fake Reviews (Deceptive)", "#d95f02")
    ]):
        subset_texts = df[df["label"] == label_val]["review_text"].apply(
            lambda t: clean_text_standard(str(t), preserve_negations=True)
        )

        vec = CountVectorizer(ngram_range=(2, 2), max_features=top_n)
        matrix = vec.fit_transform(subset_texts)
        counts = matrix.sum(axis=0).A1
        words = vec.get_feature_names_out()

        ngram_df = pd.DataFrame({"bigram": words, "frequency": counts}).sort_values("frequency", ascending=True)

        axes[idx].barh(ngram_df["bigram"], ngram_df["frequency"], color=color, edgecolor="black")
        axes[idx].set_title(f"Top Bigrams in {title}", fontweight="bold")
        axes[idx].set_xlabel("Occurrences")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "05_top_words_genuine_vs_fake.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_category_distribution(df_salminen: pd.DataFrame):
    """Figure 6: Product Category Breakdown (Salminen dataset)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    order = df_salminen["product_category"].value_counts().index
    clean_labels = [c.replace("_5", "").replace("_", " ") for c in order]

    sns.countplot(
        data=df_salminen,
        y="product_category",
        order=order,
        palette="crest",
        edgecolor="black",
        ax=ax
    )
    ax.set_yticklabels(clean_labels, fontweight="bold")
    ax.set_title("Distribution of Product Categories (Salminen et al., 40,432 Reviews)", fontweight="bold")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Product Category")

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{int(width):,}",
            (width + 80, p.get_y() + p.get_height() / 2.),
            va="center", fontsize=10, fontweight="bold"
        )

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "06_category_distribution_by_label.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_linguistic_markers(df: pd.DataFrame):
    """Figure 7: Linguistic markers comparison (exclamations, questions, uppercase, diversity)."""
    sample_df = df.sample(min(3000, len(df)), random_state=42).copy()
    features = sample_df["review_text"].apply(extract_linguistic_features).tolist()
    feat_df = pd.DataFrame(features)
    feat_df["label"] = sample_df["label"].values

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = ["#2b5c8f", "#d95f02"]

    # Exclamation Marks
    sns.barplot(
        data=feat_df, x="label", y="exclamation_count", hue="label",
        palette=colors, ax=axes[0, 0], errorbar=None, edgecolor="black", legend=False
    )
    axes[0, 0].set_title("Mean Exclamation Marks per Review", fontweight="bold")
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(["Genuine (0)", "Fake (1)"])
    axes[0, 0].set_ylabel("Count")

    # First Person Pronouns
    sns.barplot(
        data=feat_df, x="label", y="first_person_pronouns", hue="label",
        palette=colors, ax=axes[0, 1], errorbar=None, edgecolor="black", legend=False
    )
    axes[0, 1].set_title("Mean 1st-Person Pronouns ('I', 'me', 'my')", fontweight="bold")
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["Genuine (0)", "Fake (1)"])
    axes[0, 1].set_ylabel("Count")

    # Lexical Diversity
    sns.barplot(
        data=feat_df, x="label", y="lexical_diversity", hue="label",
        palette=colors, ax=axes[1, 0], errorbar=None, edgecolor="black", legend=False
    )
    axes[1, 0].set_title("Mean Lexical Diversity (Type-Token Ratio)", fontweight="bold")
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(["Genuine (0)", "Fake (1)"])
    axes[1, 0].set_ylabel("Ratio (0.0 - 1.0)")

    # Uppercase Ratio
    sns.barplot(
        data=feat_df, x="label", y="uppercase_ratio", hue="label",
        palette=colors, ax=axes[1, 1], errorbar=None, edgecolor="black", legend=False
    )
    axes[1, 1].set_title("Mean Uppercase Ratio (Capitalization)", fontweight="bold")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(["Genuine (0)", "Fake (1)"])
    axes[1, 1].set_ylabel("Ratio")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "07_linguistic_markers_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def run_full_eda() -> Dict[str, Any]:
    """Executes full EDA pipeline, saves all figures and quantitative metrics."""
    print("Loading Primary Dataset (He et al.)...")
    df_primary = load_cleaned_primary_dataset()

    print("Loading Secondary Dataset (Salminen et al.)...")
    df_salminen = load_salminen_dataset()

    print("Computing metrics and generating figures...")
    plot_class_distributions(df_primary, df_salminen)
    plot_review_length_distribution(df_primary)
    plot_review_length_boxplots(df_primary)
    plot_rating_distribution_by_label(df_salminen)
    plot_top_ngrams_genuine_vs_fake(df_primary)
    plot_category_distribution(df_salminen)
    plot_linguistic_markers(df_primary)

    # Compute Quantitative Summary Metrics
    df_primary["word_count"] = df_primary["review_text"].apply(lambda t: len(str(t).split()))
    df_primary["char_count"] = df_primary["review_text"].apply(lambda t: len(str(t)))

    genuine_sub = df_primary[df_primary["label"] == 0]
    fake_sub = df_primary[df_primary["label"] == 1]

    summary = {
        "primary_dataset": {
            "name": "He_et_al_Amazon",
            "total_reviews": len(df_primary),
            "fake_reviews": int((df_primary["label"] == 1).sum()),
            "genuine_reviews": int((df_primary["label"] == 0).sum()),
            "fake_ratio": float(round((df_primary["label"] == 1).mean(), 4)),
            "genuine_word_count_median": float(genuine_sub["word_count"].median()),
            "genuine_word_count_mean": float(round(genuine_sub["word_count"].mean(), 2)),
            "fake_word_count_median": float(fake_sub["word_count"].median()),
            "fake_word_count_mean": float(round(fake_sub["word_count"].mean(), 2)),
            "genuine_char_count_mean": float(round(genuine_sub["char_count"].mean(), 2)),
            "fake_char_count_mean": float(round(fake_sub["char_count"].mean(), 2))
        },
        "secondary_dataset": {
            "name": "Salminen_Kaggle_Amazon",
            "total_reviews": len(df_salminen),
            "fake_reviews": int((df_salminen["label"] == 1).sum()),
            "genuine_reviews": int((df_salminen["label"] == 0).sum()),
            "categories_count": int(df_salminen["product_category"].nunique()),
            "five_star_pct": float(round((df_salminen["rating"] == 5.0).mean() * 100, 2)),
            "one_star_pct": float(round((df_salminen["rating"] == 1.0).mean() * 100, 2))
        }
    }

    metrics_file = os.path.join(METRICS_DIR, "eda_summary.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved EDA metrics to: {metrics_file}")

    return summary


if __name__ == "__main__":
    res = run_full_eda()
    print("\nEDA Execution Complete! Summary:")
    print(json.dumps(res, indent=2))
