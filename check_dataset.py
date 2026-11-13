
import pandas as pd
import os

folder = "dataset"

for file in os.listdir(folder):
    if file.endswith(".csv"):
        path = os.path.join(folder, file)

        df = pd.read_csv(path)

        print("Dataset:", file)
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nDataset Shape:")
        print(df.shape)

        print("\nFirst 5 Rows:")
        print(df.head())

        print("\nLabel Distribution:")
        if "label" in df.columns:
            print(df["label"].value_counts())

        break
