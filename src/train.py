from pathlib import Path
import sys
import time
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, LABEL_MAP, RANDOM_STATE
from utils import clean_batch
from feature_extraction import build_tfidf_vectorizer
from models import get_models
from evaluate import evaluate_predictions, save_confusion_matrix, save_comparison


def load_split(name):
    filename = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}[name]
    df = pd.read_csv(DATA_DIR / filename, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"]).copy()
    unknown = sorted(set(df["label"].unique()) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"Unknown labels in {filename}: {unknown}")
    df["sentiment"] = df["label"].map(LABEL_MAP)
    df["clean_text"] = clean_batch(df["text"].tolist())
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    return df


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading train/validation/test splits...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
    print("\nTrain three-class distribution:")
    print(train_df["sentiment"].value_counts())

    # TF-IDF is fitted only on training text to avoid information leakage.
    vectorizer = build_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])
    y_train, y_val, y_test = train_df["sentiment"], val_df["sentiment"], test_df["sentiment"]

    print(f"\nTF-IDF matrix: {X_train.shape}")
    models = get_models(RANDOM_STATE)
    label_order = ["negative", "neutral", "positive"]
    validation_results = []
    fitted = {}

    for name, model in models.items():
        start = time.time()
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        result = evaluate_predictions(y_val, val_pred, name, label_order)
        result["training_seconds"] = round(time.time() - start, 2)
        validation_results.append(result)
        fitted[name] = model
        print(pd.Series(result).to_string())

        # Optional CV is performed only on the training split; validation remains untouched.
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
        result["cv_f1_macro_mean"] = cv.mean()
        result["cv_f1_macro_std"] = cv.std()
        print(f"5-fold CV F1-macro: {cv.mean():.4f} +/- {cv.std():.4f}")
        save_confusion_matrix(y_val, val_pred, label_order, f"{name}_validation", RESULTS_DIR)

    val_df_results, _, _ = save_comparison(validation_results, RESULTS_DIR)
    best_name = val_df_results.iloc[0]["model"]
    print(f"\nSelected model from validation: {best_name}")

    # Refit the selected model on train + validation only after model selection.
    combined_text = pd.concat([train_df["clean_text"], val_df["clean_text"]], ignore_index=True)
    combined_y = pd.concat([y_train, y_val], ignore_index=True)
    final_vectorizer = build_tfidf_vectorizer()
    X_combined = final_vectorizer.fit_transform(combined_text)
    X_final_test = final_vectorizer.transform(test_df["clean_text"])

    final_model = get_models(RANDOM_STATE)[best_name]
    final_model.fit(X_combined, combined_y)
    test_pred = final_model.predict(X_final_test)

    test_result = evaluate_predictions(y_test, test_pred, best_name, label_order)
    print("\n=== FINAL TEST RESULT ===")
    print(pd.Series(test_result).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, test_pred, labels=label_order, zero_division=0))
    save_confusion_matrix(y_test, test_pred, label_order, f"{best_name}_test", RESULTS_DIR)

    # Save final artifacts for the application.
    joblib.dump(final_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(final_vectorizer, MODEL_DIR / "vectorizer.joblib")
    joblib.dump(best_name, MODEL_DIR / "best_model_name.joblib")
    joblib.dump(label_order, MODEL_DIR / "labels.joblib")

    pd.DataFrame([test_result]).to_csv(
        RESULTS_DIR / "final_test_result.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\nSaved final model and vectorizer to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
