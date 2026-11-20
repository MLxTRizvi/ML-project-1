"""
Data Loader and Schema Standardization Module.
==============================================
Provides memory-efficient, leak-free data ingestion and schema harmonization
for fake product review detection across disparate research datasets.

Unified Schema:
- review_id (str): Unique review identifier.
- reviewer_id (str): Reviewer ID (enables group-based splitting).
- product_id (str): ASIN or product ID.
- product_category (str): Product category or domain (optional).
- review_title (str): Title or summary of review.
- review_text (str): Main body of the review.
- rating (float): Numerical rating (1.0 to 5.0).
- verified_purchase (float/bool): 1.0/True if verified, else 0.0/False, or NaN.
- timestamp (str): Date/time string.
- label (int): Binary label (1 = Fake/Deceptive, 0 = Honest/Genuine).
- original_label (str): Raw label from source dataset.
- source_dataset (str): Identifier of the original dataset.
"""

import os
from typing import Optional, Dict, Any, Generator
import pandas as pd
import numpy as np


UNIFIED_COLUMNS = [
    "review_id",
    "reviewer_id",
    "product_id",
    "product_category",
    "review_title",
    "review_text",
    "rating",
    "verified_purchase",
    "timestamp",
    "label",
    "original_label",
    "source_dataset"
]


def _resolve_path(rel_path: str) -> str:
    """
    Intelligently resolves relative filepaths whether code is executed from
    the project root, a Jupyter notebook inside notebooks/, or tests/.
    """
    if os.path.exists(rel_path):
        return rel_path

    # Try relative to project root (one directory up from src/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    candidate = os.path.join(project_root, rel_path)
    if os.path.exists(candidate):
        return candidate

    # Try relative to parent of current working directory
    cwd_candidate = os.path.join("..", rel_path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    return rel_path


def load_cleaned_primary_dataset(
    filepath: str = "dataset/cleaned_reviews.csv",
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads the cleaned 32,940-row primary dataset into the unified schema.
    
    Args:
        filepath: Path to cleaned_reviews.csv.
        sample_size: Optional integer to sample rows for quick development.
        random_state: Random seed for sampling.
        
    Returns:
        pd.DataFrame adhering to the unified schema.
    """
    filepath = _resolve_path(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    
    # Harmonize with unified schema
    harmonized = pd.DataFrame()
    harmonized["review_id"] = df["review_id"].astype(str)
    harmonized["reviewer_id"] = df["reviewer_id"].astype(str)
    harmonized["product_id"] = np.nan
    harmonized["product_category"] = "Amazon Products (Unspecified)"
    harmonized["review_title"] = df["review_title"].fillna("").astype(str)
    harmonized["review_text"] = df["review_text"].fillna("").astype(str)
    harmonized["rating"] = np.nan
    harmonized["verified_purchase"] = np.nan
    harmonized["timestamp"] = np.nan
    harmonized["label"] = df["label"].astype(int)
    harmonized["original_label"] = df["label"].astype(str)
    harmonized["source_dataset"] = "He_et_al_Amazon_Cleaned"

    if sample_size and sample_size < len(harmonized):
        harmonized = harmonized.sample(
            n=sample_size,
            random_state=random_state
        ).reset_index(drop=True)

    return harmonized[UNIFIED_COLUMNS]


def load_enriched_metadata_dataset(
    filepath: str = "dataset/public_reviews_dataset_cleaned.csv",
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads the full 23-column raw dataset, filters unambiguous labels,
    and preserves rich metadata (ratings, dates, helpfulness, photos).
    
    Returns:
        pd.DataFrame with unified schema and metadata columns.
    """
    filepath = _resolve_path(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source dataset not found at: {filepath}")

    use_cols = [
        "asin", "review_id", "reviewer_id", "review_title",
        "review_text", "review_rating", "review_date",
        "number_of_helpful", "number_of_photos",
        "reviewer_labeled_fake", "reviewer_labeled_honest",
        "review_is_removed_by_amazon"
    ]
    
    df = pd.read_csv(filepath, usecols=use_cols, low_memory=False)

    # Filter unambiguous labels only
    mask = (
        ((df["reviewer_labeled_fake"] == 1) & (df["reviewer_labeled_honest"] == 0)) |
        ((df["reviewer_labeled_fake"] == 0) & (df["reviewer_labeled_honest"] == 1))
    )
    df = df[mask].copy()

    # Drop duplicate review IDs and empty review text
    df = df.drop_duplicates(subset=["review_id"])
    df = df.dropna(subset=["review_text"])

    # Harmonize into unified schema
    harmonized = pd.DataFrame()
    harmonized["review_id"] = df["review_id"].astype(str)
    harmonized["reviewer_id"] = df["reviewer_id"].astype(str)
    harmonized["product_id"] = df["asin"].astype(str)
    harmonized["product_category"] = "Amazon Products"
    harmonized["review_title"] = df["review_title"].fillna("").astype(str)
    harmonized["review_text"] = df["review_text"].fillna("").astype(str)
    harmonized["rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
    harmonized["verified_purchase"] = np.nan
    harmonized["timestamp"] = df["review_date"].astype(str)
    harmonized["label"] = df["reviewer_labeled_fake"].astype(int)
    harmonized["original_label"] = np.where(
        df["reviewer_labeled_fake"].astype(int) == 1, "fake", "honest"
    )
    harmonized["source_dataset"] = "He_et_al_Amazon_Full"

    # Additional preserved metadata as separate columns
    harmonized["number_of_helpful"] = pd.to_numeric(df["number_of_helpful"], errors="coerce").fillna(0)
    harmonized["number_of_photos"] = pd.to_numeric(df["number_of_photos"], errors="coerce").fillna(0)
    harmonized["removed_by_amazon"] = pd.to_numeric(df["review_is_removed_by_amazon"], errors="coerce").fillna(0)

    if sample_size and sample_size < len(harmonized):
        harmonized = harmonized.sample(
            n=sample_size,
            random_state=random_state
        ).reset_index(drop=True)

    return harmonized


def load_salminen_dataset(
    filepath: str = "dataset/fake reviews dataset.csv",
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads the Salminen et al. (Kaggle) fake reviews dataset (40,432 rows)
    into the unified schema.
    
    Labels:
        CG (Computer Generated) -> 1 (Fake)
        OR (Original Review)    -> 0 (Honest)
    """
    filepath = _resolve_path(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Salminen dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    harmonized = pd.DataFrame()
    harmonized["review_id"] = [f"SALMINEN_{i:06d}" for i in range(len(df))]
    harmonized["reviewer_id"] = np.nan
    harmonized["product_id"] = np.nan
    harmonized["product_category"] = df["category"].astype(str)
    harmonized["review_title"] = ""
    harmonized["review_text"] = df["text_"].fillna("").astype(str)
    harmonized["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    harmonized["verified_purchase"] = np.nan
    harmonized["timestamp"] = np.nan
    harmonized["label"] = np.where(df["label"].str.strip() == "CG", 1, 0)
    harmonized["original_label"] = df["label"].astype(str)
    harmonized["source_dataset"] = "Salminen_Kaggle_Product_Reviews"

    if sample_size and sample_size < len(harmonized):
        harmonized = harmonized.sample(
            n=sample_size,
            random_state=random_state
        ).reset_index(drop=True)

    return harmonized[UNIFIED_COLUMNS]


def validate_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates a DataFrame against the unified schema requirements.
    
    Returns:
        Dictionary with validation results and diagnostic metrics.
    """
    missing_cols = [c for c in UNIFIED_COLUMNS if c not in df.columns]
    is_valid_schema = len(missing_cols) == 0

    total_rows = len(df)
    missing_texts = df["review_text"].isna().sum() if "review_text" in df.columns else total_rows
    empty_texts = (df["review_text"].str.strip() == "").sum() if "review_text" in df.columns else 0
    duplicate_ids = df["review_id"].duplicated().sum() if "review_id" in df.columns else 0
    
    label_distribution = {}
    if "label" in df.columns:
        label_distribution = df["label"].value_counts().to_dict()

    return {
        "valid_schema": is_valid_schema,
        "missing_columns": missing_cols,
        "total_rows": total_rows,
        "missing_texts": int(missing_texts),
        "empty_texts": int(empty_texts),
        "duplicate_review_ids": int(duplicate_ids),
        "label_distribution": label_distribution
    }


def chunked_csv_reader(
    filepath: str,
    chunksize: int = 10000,
    usecols: Optional[list] = None
) -> Generator[pd.DataFrame, None, None]:
    """
    Memory-efficient generator for reading massive CSV datasets in chunks.
    
    Args:
        filepath: Path to CSV.
        chunksize: Number of rows per chunk.
        usecols: Specific columns to load.
        
    Yields:
        pd.DataFrame chunks.
    """
    for chunk in pd.read_csv(filepath, chunksize=chunksize, usecols=usecols, low_memory=False):
        yield chunk


if __name__ == "__main__":
    print("Testing data_loader on primary cleaned dataset...")
    df_clean = load_cleaned_primary_dataset()
    val_clean = validate_dataset_schema(df_clean)
    print("Cleaned Dataset Validation:", val_clean)
    print("Shape:", df_clean.shape)
    print("Sample:\n", df_clean[["review_id", "label", "source_dataset"]].head(2))

    meta_path = _resolve_path("dataset/public_reviews_dataset_cleaned.csv")
    if os.path.exists(meta_path):
        print("\nTesting data_loader on enriched metadata dataset...")
        df_meta = load_enriched_metadata_dataset(meta_path, sample_size=1000)
        val_meta = validate_dataset_schema(df_meta)
        print("Enriched Metadata Validation:", val_meta)
        print("Rating distribution in sample:\n", df_meta["rating"].value_counts())
    else:
        print("\nNote: dataset/public_reviews_dataset_cleaned.csv not present (optional raw file).")
