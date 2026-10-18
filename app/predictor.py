from pathlib import Path
import sys
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import MODEL_DIR, DISPLAY_LABELS
from utils import clean_text


class SentimentPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "best_model.joblib")
        self.vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
        self.model_name = joblib.load(MODEL_DIR / "best_model_name.joblib")

    def predict(self, text: str):
        cleaned = clean_text(text)
        if not cleaned:
            return None, 0.0, {}, cleaned
        X = self.vectorizer.transform([cleaned])
        pred = self.model.predict(X)[0]
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            probs_dict = {str(c): float(p) for c, p in zip(self.model.classes_, probs)}
            confidence = max(probs_dict.values())
        else:
            probs_dict = {str(pred): 1.0}
            confidence = 1.0
        return str(pred), confidence, probs_dict, cleaned


if __name__ == "__main__":
    predictor = SentimentPredictor()
    print(f"Model: {predictor.model_name}")
    print("Type exit to stop.")
    while True:
        text = input("\nمتن فارسی: ").strip()
        if text.lower() == "exit":
            break
        label, confidence, probs, cleaned = predictor.predict(text)
        print("نتیجه:", DISPLAY_LABELS.get(label, label))
        print(f"میزان اطمینان مدل: {confidence * 100:.2f}%")
        print("متن پس از پیش‌پردازش:", cleaned)
        print("احتمال کلاس‌ها:", probs)
