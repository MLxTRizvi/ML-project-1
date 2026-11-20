"""
Unit tests for Explainability Module.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from src.explainability import (
    get_feature_names_from_pipeline,
    extract_global_feature_importance,
    explain_single_prediction
)


class TestExplainability(unittest.TestCase):

    def setUp(self):
        self.texts = [
            "This genuine vacuum cleaner works wonderfully and reliably.",
            "Great quality product, authentic seller.",
            "Total fake scam, defective item, do not buy.",
            "Horrible fake garbage, broken immediately."
        ]
        self.labels = [0, 0, 1, 1]

        self.pipeline = Pipeline([
            ("features", TfidfVectorizer()),
            ("classifier", LogisticRegression())
        ])
        self.pipeline.fit(self.texts, self.labels)

    def test_feature_names_extraction(self):
        fnames = get_feature_names_from_pipeline(self.pipeline)
        self.assertGreater(len(fnames), 5)
        self.assertIn("vacuum", fnames)
        self.assertIn("fake", fnames)

    def test_global_feature_importance(self):
        top_fake, top_honest = extract_global_feature_importance(self.pipeline, top_n=3, save_csv=False)
        self.assertEqual(len(top_fake), 3)
        self.assertEqual(len(top_honest), 3)
        self.assertIn("coefficient", top_fake.columns)
        self.assertIn("odds_ratio", top_fake.columns)

    def test_explain_single_prediction(self):
        review = "This fake product is a total scam!"
        explanation = explain_single_prediction(self.pipeline, review, top_k=3)
        self.assertEqual(explanation["predicted_class"], 1)
        self.assertEqual(explanation["predicted_label"], "Likely Suspicious")
        self.assertIsInstance(explanation["probability_fake"], float)
        self.assertGreater(len(explanation["suspicious_cues"]), 0)


if __name__ == "__main__":
    unittest.main()
