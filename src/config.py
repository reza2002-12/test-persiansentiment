from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# SentiPers label IDs in the dataset used in this project:
# 0 = very negative, 1 = negative, 2 = neutral, 3 = positive, 4 = very positive.
# The thesis requires three classes, so the two negative classes are merged,
# and the two positive classes are merged.
LABEL_MAP = {
    0: "negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "positive",
}

DISPLAY_LABELS = {
    "negative": "منفی",
    "neutral": "خنثی",
    "positive": "مثبت",
}

RANDOM_STATE = 42