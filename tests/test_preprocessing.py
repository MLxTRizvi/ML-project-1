"""
Unit tests for text preprocessing, negation preservation, and linguistic feature extraction.
"""

import unittest
import pandas as pd
from src.preprocessing import (
    clean_html,
    clean_urls,
    expand_contractions,
    normalize_repeated_chars,
    clean_text_light,
    clean_text_standard,
    clean_text_lemmatized,
    extract_linguistic_features,
    TextPreprocessorTransformer
)


class TestPreprocessing(unittest.TestCase):

    def test_clean_html(self):
        html_text = "This is a <b>great</b> product!<br/><p>Highly recommended.</p>"
        cleaned = clean_html(html_text)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("<br/>", cleaned)
        self.assertNotIn("</p>", cleaned)
        self.assertIn("great", cleaned)

    def test_clean_urls(self):
        url_text = "Check out https://amazon.com/dp/12345 or www.review.org for details."
        cleaned = clean_urls(url_text)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("www.", cleaned)
        self.assertIn("details", cleaned)

    def test_expand_contractions(self):
        text = "I don't think it won't work, but it wasn't broken."
        expanded = expand_contractions(text)
        self.assertIn("do not", expanded)
        self.assertIn("will not", expanded)
        self.assertIn("was not", expanded)

    def test_normalize_repeated_chars(self):
        text = "This is soooo coooool and veeeery goooood"
        normalized = normalize_repeated_chars(text)
        self.assertNotIn("soooo", normalized)
        self.assertIn("soo", normalized)
        self.assertIn("good", normalized)

    def test_negation_preservation(self):
        text = "This was not a good purchase. I would never buy again."
        
        # When preserving negations:
        cleaned_with_neg = clean_text_standard(text, preserve_negations=True)
        self.assertIn("not", cleaned_with_neg.split())
        self.assertNotIn("was", cleaned_with_neg.split())  # standard stopword removed
        self.assertNotIn("this", cleaned_with_neg.split())

        # When NOT preserving negations:
        cleaned_no_neg = clean_text_standard(text, preserve_negations=False)
        self.assertNotIn("not", cleaned_no_neg.split())

    def test_lemmatization(self):
        text = "He loved running and buying products"
        lemmatized = clean_text_lemmatized(text)
        # Check that stemmed/lemmatized forms appear (love, run, buy, product)
        self.assertIn("love", lemmatized)
        self.assertIn("run", lemmatized)
        self.assertIn("buy", lemmatized)
        self.assertIn("product", lemmatized)

    def test_extract_linguistic_features(self):
        text = "WOW!! Is this REAL? I bought it for my family."
        feats = extract_linguistic_features(text)
        self.assertEqual(feats["exclamation_count"], 2.0)
        self.assertEqual(feats["question_count"], 1.0)
        self.assertGreater(feats["uppercase_ratio"], 0.1)
        self.assertGreater(feats["first_person_pronouns"], 1.0)  # "I", "my"
        self.assertGreater(feats["lexical_diversity"], 0.5)

    def test_sklearn_transformer(self):
        transformer = TextPreprocessorTransformer(strategy="standard", preserve_negations=True)
        series = pd.Series(["I didn't like this product!", "Great quality!"])
        transformed = transformer.fit_transform(series)
        self.assertEqual(len(transformed), 2)
        self.assertIn("not", transformed.iloc[0])


if __name__ == "__main__":
    unittest.main()
