"""
Unit tests for Model Pipeline and Evaluation Module.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from src.evaluate import evaluate_classifier, compute_threshold_metrics


class TestModelPipeline(unittest.TestCase):

    def setUp(self):
        self.texts_train = [
            "This is a genuine product, works nicely.",
            "Loved it! High quality and sturdy.",
            "Worst fake garbage, broke immediately.",
            "Scam product, do not buy."
        ]
        self.y_train = np.array([0, 0, 1, 1])

        self.texts_test = [
            "Good product, fine quality.",
            "Fake and defective."
        ]
        self.y_test = np.array([0, 1])

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression())
        ])
        self.pipeline.fit(self.texts_train, self.y_train)

    def test_evaluate_classifier(self):
        metrics = evaluate_classifier(
            self.pipeline,
            self.texts_test,
            self.y_test,
            model_name="Test_LogisticRegression",
            train_time_sec=0.15
        )
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("precision_fake", metrics)
        self.assertIn("recall_fake", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("model_size_mb", metrics)
        self.assertEqual(metrics["model_name"], "Test_LogisticRegression")

    def test_threshold_metrics(self):
        y_scores = np.array([0.2, 0.8])
        th_df = compute_threshold_metrics(self.y_test, y_scores, thresholds=np.array([0.3, 0.5, 0.7]))
        self.assertEqual(len(th_df), 3)
        self.assertIn("precision_fake", th_df.columns)
        self.assertIn("recall_fake", th_df.columns)


if __name__ == "__main__":
    unittest.main()
