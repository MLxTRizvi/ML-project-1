"""
Explainable AI (XAI) and Model Attribution Module.
==================================================
Extracts global feature weights (log-odds coefficients) and local per-instance
word-level attributions for classical linear and tree-based classifiers.
"""

import os
import sys
import joblib
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend("Agg")
sns.set_theme(style="whitegrid")

FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def get_feature_names_from_pipeline(pipeline: Any) -> List[str]:
    """
    Extracts all feature names from a Pipeline containing FeatureUnion or TfidfVectorizer.
    """
    features_step = pipeline.named_steps.get("features")
    if features_step is None:
        features_step = pipeline.named_steps.get("tfidf")

    if features_step is None:
        raise ValueError("No recognizable feature extractor step found in pipeline.")

    if hasattr(features_step, "get_feature_names_out"):
        return list(features_step.get_feature_names_out())
    elif hasattr(features_step, "transformer_list"):
        # Handle FeatureUnion
        all_names = []
        for name, transformer in features_step.transformer_list:
            if hasattr(transformer, "get_feature_names_out"):
                all_names.extend([f"{name}__{f}" for f in transformer.get_feature_names_out()])
            elif hasattr(transformer, "named_steps"):
                extractor = transformer.named_steps.get("extractor")
                if extractor and hasattr(extractor, "get_feature_names_out"):
                    all_names.extend([f"{name}__{f}" for f in extractor.get_feature_names_out()])
                else:
                    all_names.extend([f"{name}__feat_{i}" for i in range(10)])
            else:
                all_names.extend([f"{name}__feat_{i}" for i in range(10)])
        return all_names
    else:
        return [f"feat_{i}" for i in range(50000)]


def extract_global_feature_importance(
    pipeline: Any,
    top_n: int = 25,
    save_csv: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts top positive (indicative of Fake) and top negative (indicative of Honest)
    feature coefficients from the fitted linear classifier.
    
    Returns:
        (top_fake_df, top_honest_df)
    """
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None:
        classifier = pipeline.named_steps.get("clf")

    if not hasattr(classifier, "coef_"):
        raise ValueError("Model does not have linear coefficients (coef_).")

    feature_names = get_feature_names_from_pipeline(pipeline)
    coefs = classifier.coef_[0]

    if len(feature_names) != len(coefs):
        # Fallback alignment if names count differs slightly
        min_len = min(len(feature_names), len(coefs))
        feature_names = feature_names[:min_len]
        coefs = coefs[:min_len]

    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs)
    })

    top_fake = df_coef.sort_values(by="coefficient", ascending=False).head(top_n).copy()
    top_honest = df_coef.sort_values(by="coefficient", ascending=True).head(top_n).copy()

    # Save complete sorted coefficients to CSV if requested
    if save_csv:
        df_sorted = df_coef.sort_values(by="coefficient", ascending=False)
        csv_path = os.path.join(METRICS_DIR, "top_feature_coefficients.csv")
        df_sorted.to_csv(csv_path, index=False)
        print(f"Saved complete feature coefficients to: {csv_path}")

    return top_fake, top_honest


def plot_top_feature_coefficients(top_fake: pd.DataFrame, top_honest: pd.DataFrame):
    """Figure 12: Generates a horizontal bar chart of top positive vs negative coefficients."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Top Honest Features (Negative coefficients in binary classification where 1=Fake)
    honest_clean = top_honest.copy()
    honest_clean["display_name"] = honest_clean["feature"].apply(
        lambda x: x.split("__")[-1] if "__" in x else x
    )
    axes[0].barh(honest_clean["display_name"][::-1], honest_clean["coefficient"][::-1].abs(), color="#2b5c8f", edgecolor="black")
    axes[0].set_title("Top 25 Indicators of GENUINE Reviews\n(Negative Log-Odds)", fontweight="bold")
    axes[0].set_xlabel("Coefficient Magnitude (|w|)")

    # Top Fake Features (Positive coefficients where 1=Fake)
    fake_clean = top_fake.copy()
    fake_clean["display_name"] = fake_clean["feature"].apply(
        lambda x: x.split("__")[-1] if "__" in x else x
    )
    axes[1].barh(fake_clean["display_name"][::-1], fake_clean["coefficient"][::-1], color="#d95f02", edgecolor="black")
    axes[1].set_title("Top 25 Indicators of FAKE / SUSPICIOUS Reviews\n(Positive Log-Odds)", fontweight="bold")
    axes[1].set_xlabel("Coefficient Magnitude (w)")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "12_top_feature_coefficients.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Top Feature Coefficients Figure: {out_path}")


def explain_single_prediction(pipeline: Any, text: str, top_k: int = 8) -> Dict[str, Any]:
    """
    Computes local explanation for a single review by evaluating active tokens and stylometrics.
    
    Returns:
        Dictionary with predicted class, probability, and influential features.
    """
    classifier = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("clf")
    features_step = pipeline.named_steps.get("features") or pipeline.named_steps.get("tfidf")

    # Get overall prediction and probability
    pred_class = int(pipeline.predict([text])[0])
    prob_fake = float("nan")
    if hasattr(pipeline, "predict_proba"):
        prob_fake = float(pipeline.predict_proba([text])[0, 1])

    # Transform text to sparse feature representation
    transformed = features_step.transform([text])
    feature_names = get_feature_names_from_pipeline(pipeline)
    coefs = classifier.coef_[0]

    # Non-zero feature activations for this specific review
    if hasattr(transformed, "tocoo"):
        coo = transformed.tocoo()
        active_indices = coo.col
        active_values = coo.data
    else:
        active_indices = np.where(transformed[0] != 0)[0]
        active_values = transformed[0, active_indices]

    contributions = []
    ling_labels = {
        "ling_char_count": "Character Count",
        "ling_word_count": "Word Count",
        "ling_avg_word_length": "Average Word Length",
        "ling_exclamation_count": "Exclamation Marks (!)",
        "ling_question_count": "Question Marks (?)",
        "ling_uppercase_ratio": "Capitalization / Caps Ratio",
        "ling_punctuation_ratio": "Punctuation Density",
        "ling_first_person_pronouns": "First-Person Pronouns (I, me)",
        "ling_lexical_diversity": "Lexical Diversity (Unique/Total)"
    }
    sentiment_labels = {
        "sentiment_compound": "VADER Compound Polarity",
        "sentiment_pos": "Positive Sentiment Tone",
        "sentiment_neu": "Neutral Sentiment Tone",
        "sentiment_neg": "Negative Sentiment Tone"
    }

    for idx, val in zip(active_indices, active_values):
        if idx < len(feature_names) and idx < len(coefs):
            fname = feature_names[idx]
            weight = coefs[idx]
            score = val * weight

            if "__" in fname:
                parts = fname.split("__", 1)
                branch = parts[0]
                name = parts[1]
            else:
                branch = "text"
                name = fname

            if branch == "linguistic" or name.startswith("ling_"):
                category = "Stylometry / Linguistics"
                human_label = ling_labels.get(name, name)
            elif branch == "sentiment" or name.startswith("sentiment_"):
                category = "Sentiment Polarity"
                human_label = sentiment_labels.get(name, name)
            elif branch == "char_tfidf":
                category = "Sub-word Pattern"
                human_label = f"Sub-word: '{name}'"
            else:
                category = "Vocabulary Token"
                human_label = f"Word: '{name}'"

            contributions.append({
                "raw_feature": fname,
                "feature": name,
                "branch": branch,
                "category": category,
                "human_label": human_label,
                "value": float(val),
                "weight": float(weight),
                "contribution": float(score)
            })

    df_contrib = pd.DataFrame(contributions)
    if not df_contrib.empty:
        suspicious_cues = (
            df_contrib[df_contrib["contribution"] > 0]
            .sort_values(by="contribution", ascending=False)
            .head(top_k)
            .to_dict(orient="records")
        )
        genuine_cues = (
            df_contrib[df_contrib["contribution"] < 0]
            .sort_values(by="contribution", ascending=True)
            .head(top_k)
            .to_dict(orient="records")
        )
    else:
        suspicious_cues, genuine_cues = [], []

    return {
        "text": text,
        "predicted_class": pred_class,
        "predicted_label": "Likely Suspicious" if pred_class == 1 else "Likely Genuine",
        "probability_fake": round(prob_fake, 4) if not np.isnan(prob_fake) else None,
        "suspicious_cues": suspicious_cues,
        "genuine_cues": genuine_cues
    }


if __name__ == "__main__":
    model_path = "models/best_model_enhanced.joblib"
    if not os.path.exists(model_path):
        model_path = "models/best_model.joblib"

    print(f"Loading model for explainability analysis: {model_path}")
    pipeline = joblib.load(model_path)

    top_fake, top_honest = extract_global_feature_importance(pipeline, top_n=20)
    plot_top_feature_coefficients(top_fake, top_honest)

    sample_text = "I LOVE this product!! It is absolutely the best purchase ever. Very fast shipping."
    print(f"\nExplaining prediction for: '{sample_text}'")
    explanation = explain_single_prediction(pipeline, sample_text)
    print("Prediction:", explanation["predicted_label"], f"(Prob Fake: {explanation['probability_fake']})")
    print("Top Suspicious Cues:", [c["feature"] for c in explanation["suspicious_cues"][:4]])
    print("Top Genuine Cues:", [c["feature"] for c in explanation["genuine_cues"][:4]])
