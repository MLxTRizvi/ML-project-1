
import pandas as pd

df = pd.read_csv("dataset/public_reviews_dataset_cleaned.csv")

label_columns = [
    "reviewer_classified_fake",
    "reviewer_classified_honest",
    "reviewer_labeled_fake",
    "reviewer_labeled_honest",
    "review_is_removed_by_amazon"
]

for col in label_columns:
    print("\n", col)
    print(df[col].value_counts(dropna=False))

print("\nFake and honest label combinations:")
print(
    df.groupby(
        ["reviewer_labeled_fake", "reviewer_labeled_honest"],
        dropna=False
    ).size()
)

print("\nMissing review text:")
print(df["review_text"].isna().sum())

print("\nDuplicate review IDs:")
print(df["review_id"].duplicated().sum())
