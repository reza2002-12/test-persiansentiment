import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from config import DATA_DIR, MODEL_DIR
from utils import clean_batch


LABEL_MAP = {
    0: "negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "positive",
}


def main():
    print("Loading test data...")

    test = pd.read_csv(DATA_DIR / "test.csv")

    test = test.dropna(subset=["text", "label"])

    test["text"] = clean_batch(test["text"].tolist())

    test = test[test["text"].str.strip() != ""]

    test["sentiment"] = test["label"].map(LABEL_MAP)

    print(f"Test samples: {len(test)}")

    print("Loading oversampled model...")

    model_dir = MODEL_DIR / "oversampled"

    model = joblib.load(model_dir / "best_model.joblib")
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    model_name = joblib.load(model_dir / "best_model_name.joblib")

    print(f"Model: {model_name}")

    X_test = vectorizer.transform(test["text"])
    y_test = test["sentiment"]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    print("\n==============================")
    print("OVERSAMPLED TEST RESULTS")
    print("==============================")

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro Precision: {precision * 100:.2f}%")
    print(f"Macro Recall: {recall * 100:.2f}%")
    print(f"Macro F1: {f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["negative", "neutral", "positive"],
            zero_division=0,
        )
    )

    result = pd.DataFrame({
        "text": test["text"],
        "true_label": y_test,
        "predicted_label": predictions,
    })

    result.to_csv(
        "results/oversampled_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Test predictions saved successfully.")


if __name__ == "__main__":
    main()