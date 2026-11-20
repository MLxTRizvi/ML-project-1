"""
Text Preprocessing and NLP Cleaning Module.
============================================
Provides production-grade text normalization, negation-preserving stopwords,
lemmatization, linguistic feature extraction, and scikit-learn compatible transformers.
"""

import re
import string
from typing import List, Set, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# NLTK imports with graceful fallback
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _nltk_stopwords = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
except Exception:
    _nltk_stopwords = set()
    _lemmatizer = None

# Comprehensive Contraction Expansion Dictionary
CONTRACTIONS_MAP: Dict[str, str] = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "ain't": "is not",
    "aren't": "are not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "wouldn't": "would not",
    "'s": " is",
    "'re": " are",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is"
}

# Negation terms that must NEVER be removed in sentiment or deceptive analysis
NEGATION_WORDS: Set[str] = {
    "not", "no", "never", "nor", "neither", "none", "nobody",
    "nowhere", "hardly", "scarcely", "barely", "nothing", "without",
    "cannot", "n't"
}

# Compile Regex Patterns for High Performance
HTML_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")  # "sooooo" -> "soo"
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)

# Base Stopwords Fallback if NLTK is offline
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


def get_stopwords(preserve_negations: bool = True) -> Set[str]:
    """
    Returns the stopword set, optionally preserving negation words.
    """
    base_stops = _nltk_stopwords if _nltk_stopwords else DEFAULT_STOPWORDS
    stops = set(base_stops)
    if preserve_negations:
        stops = stops - NEGATION_WORDS
    return stops


def clean_html(text: str) -> str:
    """Removes HTML tags and entities."""
    return HTML_PATTERN.sub(" ", text)


def clean_urls(text: str) -> str:
    """Removes HTTP/HTTPS/WWW URLs."""
    return URL_PATTERN.sub(" ", text)


def expand_contractions(text: str) -> str:
    """Expands common English contractions (e.g. don't -> do not)."""
    text_lower = text.lower()
    for contraction, expansion in CONTRACTIONS_MAP.items():
        if contraction in text_lower:
            text = re.sub(
                r"\b" + re.escape(contraction) + r"\b",
                expansion,
                text,
                flags=re.IGNORECASE
            )
    return text


def normalize_repeated_chars(text: str) -> str:
    """Compresses 3+ repeated characters down to 2 (e.g. 'loooove' -> 'loove')."""
    return REPEATED_CHARS_PATTERN.sub(r"\1\1", text)


def clean_whitespace(text: str) -> str:
    """Collapses multiple spaces, tabs, and newlines into a single space."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text_raw(text: str) -> str:
    """Baseline: minimal whitespace trimming."""
    if not isinstance(text, str):
        return ""
    return text.strip()


def clean_text_light(text: str) -> str:
    """
    Light preprocessing: removes HTML and URLs, expands contractions,
    normalizes repeated characters, and standardizes whitespace.
    Preserves case and punctuation.
    """
    if not isinstance(text, str):
        return ""
    text = clean_html(text)
    text = clean_urls(text)
    text = expand_contractions(text)
    text = normalize_repeated_chars(text)
    text = clean_whitespace(text)
    return text


def clean_text_standard(
    text: str,
    remove_stopwords: bool = True,
    preserve_negations: bool = True
) -> str:
    """
    Standard NLP preprocessing:
    Light cleaning + lowercasing + punctuation removal + optional negation-aware stopwords.
    """
    if not isinstance(text, str):
        return ""
    text = clean_text_light(text)
    text = text.lower()
    text = text.translate(PUNCT_TRANSLATOR)
    
    tokens = text.split()
    if remove_stopwords:
        stops = get_stopwords(preserve_negations=preserve_negations)
        tokens = [t for t in tokens if t not in stops]
        
    return " ".join(tokens)


def clean_text_lemmatized(
    text: str,
    preserve_negations: bool = True
) -> str:
    """
    Full preprocessing: Standard cleaning followed by WordNet lemmatization.
    """
    standard = clean_text_standard(text, remove_stopwords=True, preserve_negations=preserve_negations)
    if not standard:
        return ""
    if _lemmatizer is None:
        return standard
    tokens = [
        _lemmatizer.lemmatize(_lemmatizer.lemmatize(t, pos="v"), pos="n")
        for t in standard.split()
    ]
    return " ".join(tokens)


def extract_linguistic_features(text: str) -> Dict[str, float]:
    """
    Extracts stylometric and linguistic markers indicative of deceptive text:
    - Character count
    - Word count
    - Average word length
    - Exclamation mark count
    - Question mark count
    - Uppercase ratio (shouting)
    - Punctuation ratio
    - First-person pronoun count ('I', 'me', 'my', 'we', 'our')
    - Lexical diversity (Type-Token Ratio)
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "char_count": 0.0,
            "word_count": 0.0,
            "avg_word_length": 0.0,
            "exclamation_count": 0.0,
            "question_count": 0.0,
            "uppercase_ratio": 0.0,
            "punctuation_ratio": 0.0,
            "first_person_pronouns": 0.0,
            "lexical_diversity": 0.0
        }

    raw_len = len(text)
    words = text.split()
    word_count = len(words)
    
    # Uppercase and punctuation ratios
    upper_count = sum(1 for c in text if c.isupper())
    punct_count = sum(1 for c in text if c in string.punctuation)
    exclamations = text.count("!")
    questions = text.count("?")
    
    # Average word length
    clean_words = [w.translate(PUNCT_TRANSLATOR).lower() for w in words]
    clean_words = [w for w in clean_words if w]
    clean_word_count = len(clean_words)
    total_word_len = sum(len(w) for w in clean_words)
    avg_word_len = total_word_len / clean_word_count if clean_word_count > 0 else 0.0
    
    # First-person pronouns
    first_person_set = {"i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves"}
    fp_count = sum(1 for w in clean_words if w in first_person_set)
    
    # Lexical diversity (Type-Token Ratio)
    unique_words = set(clean_words)
    lexical_diversity = len(unique_words) / clean_word_count if clean_word_count > 0 else 0.0

    return {
        "char_count": float(raw_len),
        "word_count": float(word_count),
        "avg_word_length": float(round(avg_word_len, 3)),
        "exclamation_count": float(exclamations),
        "question_count": float(questions),
        "uppercase_ratio": float(round(upper_count / raw_len, 4)) if raw_len > 0 else 0.0,
        "punctuation_ratio": float(round(punct_count / raw_len, 4)) if raw_len > 0 else 0.0,
        "first_person_pronouns": float(fp_count),
        "lexical_diversity": float(round(lexical_diversity, 4))
    }


class TextPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that encapsulates text cleaning.
    Prevents data leakage when used within an sklearn Pipeline.
    """
    
    def __init__(
        self,
        strategy: str = "standard",
        preserve_negations: bool = True
    ):
        self.strategy = strategy
        self.preserve_negations = preserve_negations
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        cleaner = clean_text_standard
        if self.strategy == "raw":
            cleaner = clean_text_raw
        elif self.strategy == "light":
            cleaner = clean_text_light
        elif self.strategy == "lemmatized":
            cleaner = lambda t: clean_text_lemmatized(t, preserve_negations=self.preserve_negations)
        else:
            cleaner = lambda t: clean_text_standard(t, preserve_negations=self.preserve_negations)

        if isinstance(X, pd.Series):
            return X.fillna("").apply(cleaner)
        elif isinstance(X, (list, np.ndarray)):
            return [cleaner(str(t)) for t in X]
        else:
            return cleaner(str(X))


if __name__ == "__main__":
    sample = "I LOVED this product!! It didn't break at all. Check it out at https://amazon.com <br/> Very goooood!"
    print("Sample:", sample)
    print("\nRaw:", clean_text_raw(sample))
    print("\nLight:", clean_text_light(sample))
    print("\nStandard (Preserving negations):", clean_text_standard(sample, preserve_negations=True))
    print("\nStandard (Stripping negations):", clean_text_standard(sample, preserve_negations=False))
    print("\nLemmatized:", clean_text_lemmatized(sample))
    print("\nLinguistic features:\n", extract_linguistic_features(sample))
