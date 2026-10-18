from pathlib import Path
import pandas as pd
from config import DATA_DIR, LABEL_MAP


def main():
    print("=== Persian Sentiment Dataset Analysis ===")
    for split, filename in [("Train", "train.csv"), ("Validation", "val.csv"), ("Test", "test.csv")]:
        path = DATA_DIR / filename
        df = pd.read_csv(path)
        print(f"\n{split}: {len(df)} rows")
        print("Columns:", df.columns.tolist())
        print("Raw labels:")
        print(df["label"].value_counts().sort_index())
        if set(df["label"].unique()).issubset(LABEL_MAP):
            mapped = df["label"].map(LABEL_MAP)
            print("Three-class labels:")
            print(mapped.value_counts())


if __name__ == "__main__":
    main()
