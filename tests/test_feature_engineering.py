"""
Unit tests for Feature Engineering Module and Pipeline Builders.
"""

import unittest
import numpy as np
import pandas as pd
from src.feature_engineering import (
    LinguisticFeatureTransformer,
    build_baseline_feature_pipeline,
    build_enhanced_feature_pipeline
)


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.sample_texts = [
            "I LOVE this vacuum cleaner!! It did not lose suction once.",
            "Complete disaster. Stopped working in 10 minutes, do not buy.",
            "Standard paper notebook with 100 pages."
        ]

    def test_linguistic_feature_transformer(self):
        trans = LinguisticFeatureTransformer()
        mat = trans.fit_transform(self.sample_texts)
        self.assertIsInstance(mat, np.ndarray)
        self.assertEqual(mat.shape, (3, 9))
        self.assertEqual(len(trans.get_feature_names_out()), 9)

    def test_baseline_feature_pipeline(self):
        pipe_a = build_baseline_feature_pipeline(max_features=50)
        mat_a = pipe_a.fit_transform(self.sample_texts)
        self.assertEqual(mat_a.shape[0], 3)
        self.assertGreater(mat_a.shape[1], 0)

    def test_enhanced_feature_pipeline(self):
        pipe_b = build_enhanced_feature_pipeline(
            word_max_features=50,
            char_max_features=50
        )
        mat_b = pipe_b.fit_transform(self.sample_texts)
        self.assertEqual(mat_b.shape[0], 3)
        # Should include word + char + 9 linguistic + 4 sentiment features
        self.assertGreater(mat_b.shape[1], 9 + 4)


if __name__ == "__main__":
    unittest.main()
