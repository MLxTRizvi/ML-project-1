"""
Sentiment Analysis Module (Classical Rule-Based NLP).
=====================================================
Utilizes NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to
extract granular lexical sentiment metrics (pos, neg, neu, compound) and
provides a scikit-learn compatible transformer for machine learning pipelines.
"""

from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import nltk

class VADERSentimentAnalyzer:
    """
    Classical rule-based sentiment analysis engine using VADER.
    
    VADER is specifically attuned to sentiments expressed in social media and
    product reviews, incorporating:
    - Punctuation-based intensifiers ("great!!" vs "great")
    - Capitalization shifts ("GREAT" vs "great")
    - Degree modifiers ("extremely good" vs "marginally good")
    - Polarity shifts caused by contrastive conjunctions ("good, but...")
    """

    def __init__(
        self,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05
    ):
        try:
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def get_scores(self, text: str) -> Dict[str, float]:
        """
        Computes granular sentiment scores for a given text.
        
        Returns:
            Dictionary containing 'pos', 'neg', 'neu', and 'compound' scores.
        """
        if not isinstance(text, str) or not text.strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return self.sia.polarity_scores(text)

    def classify_sentiment(self, compound_score: float) -> str:
        """
        Maps a continuous compound polarity score [-1.0, 1.0] into discrete sentiment classes.
        
        - Compound >= pos_threshold: Positive
        - Compound <= neg_threshold: Negative
        - Otherwise: Neutral
        """
        if compound_score >= self.pos_threshold:
            return "Positive"
        elif compound_score <= self.neg_threshold:
            return "Negative"
        else:
            return "Neutral"

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Performs complete sentiment evaluation returning both scores and qualitative label.
        """
        scores = self.get_scores(text)
        sentiment_label = self.classify_sentiment(scores["compound"])
        return {
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
            "compound": scores["compound"],
            "sentiment_label": sentiment_label
        }

    def analyze_dataframe(self, df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
        """
        Efficiently evaluates an entire pandas DataFrame column and appends sentiment columns.
        """
        records = df[text_col].fillna("").apply(self.analyze).tolist()
        sentiment_df = pd.DataFrame(records)
        return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)


class VADERFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that transforms raw review text
    into a 2D numpy array of 4 VADER sentiment features:
    [pos, neg, neu, compound].
    
    Enables seamless inclusion into scikit-learn FeatureUnion or ColumnTransformer.
    """

    def __init__(self):
        self.analyzer = VADERSentimentAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        """
        Extracts 4 sentiment features per text input.
        Returns shape (n_samples, 4).
        """
        if isinstance(X, pd.Series):
            texts = X.fillna("").astype(str).tolist()
        elif isinstance(X, np.ndarray):
            texts = [str(x) for x in X]
        elif isinstance(X, list):
            texts = [str(x) for x in X]
        else:
            texts = [str(X)]

        feature_matrix = np.zeros((len(texts), 4), dtype=np.float32)
        for i, text in enumerate(texts):
            scores = self.analyzer.get_scores(text)
            feature_matrix[i, 0] = scores["pos"]
            feature_matrix[i, 1] = scores["neg"]
            feature_matrix[i, 2] = scores["neu"]
            feature_matrix[i, 3] = scores["compound"]

        return feature_matrix

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"]


if __name__ == "__main__":
    analyzer = VADERSentimentAnalyzer()
    samples = [
        "This product is outstanding! Absolutely loved the quality.",
        "Terrible experience. Broke on the first day, complete waste of money.",
        "The package arrived on Tuesday. It is a standard plastic cable."
    ]
    for s in samples:
        print(f"\nText: {s}")
        print("Analysis:", analyzer.analyze(s))

    extractor = VADERFeatureExtractor()
    feat_mat = extractor.transform(samples)
    print("\nExtracted Feature Matrix shape:", feat_mat.shape)
    print("Features:\n", feat_mat)
