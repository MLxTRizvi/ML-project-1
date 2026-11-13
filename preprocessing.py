
import pandas as pd

# Load dataset
df = pd.read_csv(
    "dataset/public_reviews_dataset_cleaned.csv",
    low_memory=False
)

# Select only explicitly labelled reviews
df = df[
    (
        (df["reviewer_labeled_fake"] == 1) &
        (df["reviewer_labeled_honest"] == 0)
    ) |
    (
        (df["reviewer_labeled_fake"] == 0) &
        (df["reviewer_labeled_honest"] == 1)
    )
].copy()

# Create target variable
df["label"] = df["reviewer_labeled_fake"].astype(int)

# Select required columns
df = df[["review_id", "reviewer_id", "review_title",
         "review_text", "label"]]

# Remove duplicate records and empty text
df = df.drop_duplicates(subset="review_id")
df = df.dropna(subset=["review_text"])

# Save cleaned dataset
df.to_csv("dataset/cleaned_reviews.csv", index=False)

# Display results
print("Preprocessing completed!")
print("Dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True) * 100)
