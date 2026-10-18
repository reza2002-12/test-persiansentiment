import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import DATA_DIR, MODEL_DIR
from utils import clean_batch
from feature_extraction import build_tfidf_vectorizer
from models import get_models


LABEL_MAP = {
    0: "negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "positive",
}


def load_data():
    train = pd.read_csv(DATA_DIR / "train_oversampled.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")

    train = train.dropna(subset=["text", "label"])
    val = val.dropna(subset=["text", "label"])

    train["text"] = clean_batch(train["text"].tolist())
    val["text"] = clean_batch(val["text"].tolist())

    train = train[train["text"].str.strip() != ""]
    val = val[val["text"].str.strip() != ""]

    train["sentiment"] = train["label"].map(LABEL_MAP)
    val["sentiment"] = val["label"].map(LABEL_MAP)

    return train, val


def main():
    print("Loading data...")

    train, val = load_data()

    y_train = train["sentiment"]
    y_val = val["sentiment"]

    print(f"Train samples: {len(train)}")
    print(f"Validation samples: {len(val)}")

    print("Building TF-IDF...")

    vectorizer = build_tfidf_vectorizer()

    X_train = vectorizer.fit_transform(train["text"])
    X_val = vectorizer.transform(val["text"])

    print("Training models...")

    models = get_models()

    results = []

    for name, model in models.items():
        print(f"\nTraining: {name}")

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(
            y_val, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_val, predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        })

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Macro F1: {f1 * 100:.2f}%")


    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["f1_macro"].idxmax()
    ]

    best_name = best_row["model"]
    best_model = models[best_name]

    print("\n==============================")
    print("OVERSAMPLED RESULTS")
    print("==============================")
    print(results_df.to_string(index=False))

    print("\nBest model:", best_name)
    print(f"Best Macro F1: {best_row['f1_macro'] * 100:.2f}%")

    output_dir = MODEL_DIR / "oversampled"
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, output_dir / "best_model.joblib")
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")
    joblib.dump(best_name, output_dir / "best_model_name.joblib")

    results_df.to_csv(
        output_dir / "validation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nOversampled model saved successfully.")


if __name__ == "__main__":
    main()