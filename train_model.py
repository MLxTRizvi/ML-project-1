
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_reviews.csv")

# Combine review title and review text
df["text"] = (
    df["review_title"].fillna("").astype(str) + " " +
    df["review_text"].fillna("").astype(str)
).str.strip()

X = df["text"]
y = df["label"]
groups = df["reviewer_id"]

# Split by reviewer to prevent data leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]
y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# Verify that reviewers do not overlap
train_reviewers = set(groups.iloc[train_index])
test_reviewers = set(groups.iloc[test_index])

assert train_reviewers.isdisjoint(test_reviewers)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Reviewer overlap:", len(train_reviewers & test_reviewers))

# Define machine learning models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}

results = []
best_model = None
best_score = -1
best_name = ""

# Train and evaluate each model
for name, classifier in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english"
        )),
        ("classifier", classifier)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    macro_f1 = f1_score(
        y_test, predictions, average="macro",
        zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision (Fake):", round(precision, 4))
    print("Recall (Fake):", round(recall, 4))
    print("F1-score (Fake):", round(f1, 4))
    print("Macro F1-score:", round(macro_f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Honest", "Fake"],
        zero_division=0
    ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Macro_F1": macro_f1
    })

    # Select best model using macro F1
    if macro_f1 > best_score:
        best_score = macro_f1
        best_model = pipeline
        best_name = name
        best_predictions = predictions

# Save model comparison
comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="Macro_F1",
    ascending=False
)

comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# Save best trained pipeline
joblib.dump(best_model, "models/best_model.joblib")

# Generate confusion matrix for best model
cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Honest", "Fake"],
    yticklabels=["Honest", "Fake"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - " + best_name)
plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

# Display final results
print("\n==============================")
print("MODEL TRAINING COMPLETED")
print("==============================")

print("Best Model:", best_name)
print("Best Macro F1-score:", round(best_score, 4))

print("\nModel Comparison:")
print(comparison.to_string(index=False))

print("\nSaved files:")
print("models/best_model.joblib")
print("outputs/model_comparison.csv")
print("outputs/confusion_matrix.png")
