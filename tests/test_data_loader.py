"""
Unit tests for data loader and schema validation.
"""

import unittest
import pandas as pd
from src.data_loader import (
    load_cleaned_primary_dataset,
    load_salminen_dataset,
    validate_dataset_schema,
    UNIFIED_COLUMNS
)


class TestDataLoader(unittest.TestCase):
    
    def test_load_cleaned_primary_dataset_schema(self):
        df = load_cleaned_primary_dataset(sample_size=100)
        self.assertEqual(len(df), 100)
        for col in UNIFIED_COLUMNS:
            self.assertIn(col, df.columns)

    def test_load_salminen_dataset_schema(self):
        df = load_salminen_dataset(sample_size=100)
        self.assertEqual(len(df), 100)
        for col in UNIFIED_COLUMNS:
            self.assertIn(col, df.columns)
        self.assertEqual(df["source_dataset"].iloc[0], "Salminen_Kaggle_Product_Reviews")
        unique_labels = set(df["label"].unique())
        self.assertTrue(unique_labels.issubset({0, 1}))
            
    def test_schema_validation(self):
        df = load_cleaned_primary_dataset(sample_size=50)
        report = validate_dataset_schema(df)
        self.assertTrue(report["valid_schema"])
        self.assertEqual(len(report["missing_columns"]), 0)
        self.assertEqual(report["empty_texts"], 0)
        self.assertEqual(report["duplicate_review_ids"], 0)
        self.assertEqual(report["total_rows"], 50)
        
    def test_label_values(self):
        df = load_cleaned_primary_dataset(sample_size=50)
        unique_labels = set(df["label"].unique())
        self.assertTrue(unique_labels.issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
