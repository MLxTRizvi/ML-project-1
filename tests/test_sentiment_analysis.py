"""
Unit tests for VADER Sentiment Analysis Module.
"""

import unittest
import numpy as np
import pandas as pd
from src.sentiment_analysis import VADERSentimentAnalyzer, VADERFeatureExtractor


class TestSentimentAnalysis(unittest.TestCase):

    def setUp(self):
        self.analyzer = VADERSentimentAnalyzer()
        self.extractor = VADERFeatureExtractor()

    def test_positive_sentiment(self):
        text = "This product is absolutely amazing and wonderfully built!"
        res = self.analyzer.analyze(text)
        self.assertEqual(res["sentiment_label"], "Positive")
        self.assertGreater(res["compound"], 0.2)
        self.assertGreater(res["pos"], 0.1)

    def test_negative_sentiment(self):
        text = "Worst purchase ever! Terribly defective, arrived broken and useless."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["sentiment_label"], "Negative")
        self.assertLess(res["compound"], -0.2)
        self.assertGreater(res["neg"], 0.1)

    def test_neutral_sentiment(self):
        # Note: Words like 'screws' in hardware can trigger slang penalties in VADER
        text = "The shipment arrived on Tuesday in a blue cardboard box."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["sentiment_label"], "Neutral")
        self.assertGreater(res["neu"], 0.7)

    def test_dataframe_analysis(self):
        df = pd.DataFrame({
            "review_text": [
                "Great!",
                "Terrible!",
                "Neutral package."
            ]
        })
        enriched = self.analyzer.analyze_dataframe(df)
        self.assertIn("compound", enriched.columns)
        self.assertIn("sentiment_label", enriched.columns)
        self.assertEqual(len(enriched), 3)

    def test_feature_extractor_transformer(self):
        texts = ["Excellent item!", "Awful item!"]
        mat = self.extractor.fit_transform(texts)
        self.assertIsInstance(mat, np.ndarray)
        self.assertEqual(mat.shape, (2, 4))
        self.assertEqual(len(self.extractor.get_feature_names_out()), 4)


if __name__ == "__main__":
    unittest.main()
