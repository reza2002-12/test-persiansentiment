from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)


def evaluate_predictions(y_true, y_pred, model_name, label_order):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_order, average="macro", zero_division=0
    )
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def save_confusion_matrix(y_true, y_pred, labels, model_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    safe = "_".join(model_name.lower().split())
    path = out_dir / f"confusion_matrix_{safe}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_comparison(results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    csv_path = out_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    ax = df.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot(
        kind="bar", figsize=(10, 5), ylim=(0, 1.05)
    )
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.tick_params(axis="x", rotation=15)
    fig = ax.get_figure()
    fig.tight_layout()
    chart_path = out_dir / "model_comparison.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return df, csv_path, chart_path
