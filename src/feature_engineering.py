"""
Feature Engineering and Multi-Modal Feature Extraction.
========================================================
Combines textual, stylometric, and sentiment features into scikit-learn
pipelines without data leakage.

Pipelines:
- Pipeline A (Baseline): Pure TF-IDF n-gram text representations.
- Pipeline B (Enhanced): Multi-modal FeatureUnion combining Word TF-IDF,
                         Character TF-IDF, Scaled Linguistic Stylometrics,
                         and Scaled VADER Sentiment Scores.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    get_stopwords,
    extract_linguistic_features,
    clean_text_light
)
from src.sentiment_analysis import VADERFeatureExtractor


class LinguisticFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms review text into a dense 2D numpy array of 9 stylometric markers:
    1. char_count
    2. word_count
    3. avg_word_length
    4. exclamation_count
    5. question_count
    6. uppercase_ratio
    7. punctuation_ratio
    8. first_person_pronouns
    9. lexical_diversity
    """

    def __init__(self):
        self.feature_names = [
            "char_count",
            "word_count",
            "avg_word_length",
            "exclamation_count",
            "question_count",
            "uppercase_ratio",
            "punctuation_ratio",
            "first_person_pronouns",
            "lexical_diversity"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.Series):
            texts = X.fillna("").astype(str).tolist()
        elif isinstance(X, np.ndarray):
            texts = [str(x) for x in X]
        elif isinstance(X, list):
            texts = [str(x) for x in X]
        else:
            texts = [str(X)]

        matrix = np.zeros((len(texts), len(self.feature_names)), dtype=np.float32)
        for i, text in enumerate(texts):
            feats = extract_linguistic_features(text)
            for j, fname in enumerate(self.feature_names):
                matrix[i, j] = feats[fname]

        return matrix

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return [f"ling_{f}" for f in self.feature_names]


def build_baseline_feature_pipeline(
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    preserve_negations: bool = True
) -> TfidfVectorizer:
    """
    Constructs the Baseline (Pipeline A) feature extractor:
    Pure Word-level TF-IDF Vectorization with negation-preserving stopwords.
    """
    stops = list(get_stopwords(preserve_negations=preserve_negations))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=stops,
        sublinear_tf=True
    )
    return vectorizer


def build_enhanced_feature_pipeline(
    word_max_features: int = 40000,
    char_max_features: int = 15000,
    word_ngram_range: tuple = (1, 2),
    char_ngram_range: tuple = (2, 5),
    min_df: int = 2,
    preserve_negations: bool = True
) -> FeatureUnion:
    """
    Constructs the Enhanced (Pipeline B) multi-modal feature union:
    1. Word-level TF-IDF (unigrams & bigrams)
    2. Character-level TF-IDF (captures sub-word morphology and character patterns)
    3. Scaled Stylometric Linguistic Features (length, punctuation, capitalization)
    4. Scaled VADER Sentiment Scores (pos, neg, neu, compound)
    """
    stops = list(get_stopwords(preserve_negations=preserve_negations))

    # Branch 1: Word-level TF-IDF
    word_tfidf = TfidfVectorizer(
        max_features=word_max_features,
        ngram_range=word_ngram_range,
        min_df=min_df,
        stop_words=stops,
        sublinear_tf=True
    )

    # Branch 2: Char-level TF-IDF
    char_tfidf = TfidfVectorizer(
        max_features=char_max_features,
        ngram_range=char_ngram_range,
        min_df=min_df,
        analyzer="char",
        sublinear_tf=True
    )

    # Branch 3: Scaled Linguistic Stylometrics
    linguistic_pipe = Pipeline([
        ("extractor", LinguisticFeatureTransformer()),
        ("scaler", StandardScaler())
    ])

    # Branch 4: Scaled VADER Sentiment
    sentiment_pipe = Pipeline([
        ("extractor", VADERFeatureExtractor()),
        ("scaler", StandardScaler())
    ])

    enhanced_union = FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
        ("linguistic", linguistic_pipe),
        ("sentiment", sentiment_pipe)
    ])

    return enhanced_union


if __name__ == "__main__":
    sample_texts = [
        "This product is ABSOLUTELY amazing!! Didn't have any flaws.",
        "Worst purchase ever. Completely useless, broke immediately.",
        "It is okay. Average quality for the price paid."
    ]

    print("Testing Baseline Feature Extractor (Pipeline A)...")
    pipe_a = build_baseline_feature_pipeline(max_features=100)
    mat_a = pipe_a.fit_transform(sample_texts)
    print("Baseline matrix shape:", mat_a.shape)

    print("\nTesting Enhanced Multi-Modal Feature Extractor (Pipeline B)...")
    pipe_b = build_enhanced_feature_pipeline(word_max_features=100, char_max_features=100)
    mat_b = pipe_b.fit_transform(sample_texts)
    print("Enhanced matrix shape:", mat_b.shape)
    print("Feature dimensions:", mat_b.shape[1], "columns")
