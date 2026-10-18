from pathlib import Path
import sys
import joblib
import streamlit as st

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

# Allow importing project modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import clean_batch


# ---------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------
MODEL_PATH = MODEL_DIR / "best_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"
MODEL_NAME_PATH = MODEL_DIR / "best_model_name.joblib"
LABELS_PATH = MODEL_DIR / "labels.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
model_name = joblib.load(MODEL_NAME_PATH)
label_order = joblib.load(LABELS_PATH)


# ---------------------------------------------------------
# Display labels
# ---------------------------------------------------------
DISPLAY_LABELS = {
    "negative": "منفی",
    "neutral": "خنثی",
    "positive": "مثبت",
}


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="سامانه تحلیل احساسات فارسی",
    page_icon="💬",
    layout="centered",
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #dddddd;
    }

    .result-title {
        font-size: 20px;
        font-weight: bold;
    }

    .result-label {
        font-size: 30px;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">سامانه تشخیص احساسات متن فارسی</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "این سامانه با استفاده از روش‌های یادگیری ماشین، "
    "احساس متن فارسی را در سه کلاس مثبت، منفی و خنثی تشخیص می‌دهد."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------
with st.expander("اطلاعات مدل"):
    st.write(f"مدل انتخاب‌شده: {model_name}")
    st.write("روش استخراج ویژگی: TF-IDF")
    st.write("تعداد کلاس‌ها: 3")
    st.write("کلاس‌ها: مثبت، منفی، خنثی")


# ---------------------------------------------------------
# Text input
# ---------------------------------------------------------
text = st.text_area(
    "متن فارسی خود را وارد کنید:",
    height=180,
    placeholder="مثلاً: کیفیت این محصول خیلی خوب بود و از خرید آن راضی هستم.",
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("🔍 تحلیل احساس متن", use_container_width=True):

    if not text.strip():
        st.warning("لطفاً ابتدا یک متن فارسی وارد کنید.")
        st.stop()

    # Preprocess text using the same preprocessing used during training
    cleaned_text = clean_batch([text])[0]

    if not cleaned_text.strip():
        st.warning("پس از پیش‌پردازش، متن قابل تحلیل نبود.")
        st.stop()

    # TF-IDF transformation
    X = vectorizer.transform([cleaned_text])

    # Prediction
    prediction = model.predict(X)[0]

    # Convert model label to Persian
    predicted_label = DISPLAY_LABELS.get(
        str(prediction),
        str(prediction)
    )
    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        confidence = float(max(probabilities)) * 100

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)

        # Convert decision scores approximately to probabilities
        import numpy as np

        scores = np.asarray(scores)

        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T

        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        confidence = float(np.max(probabilities)) * 100

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------
    st.markdown("---")

    st.subheader("نتیجه تحلیل")

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-title">احساس تشخیص داده‌شده</div>
            <div class="result-label">{predicted_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence is not None:
        st.write("")

        st.metric(
            "میزان اطمینان مدل",
            f"{confidence:.2f}%"
        )

        st.progress(
            min(max(confidence / 100, 0.0), 1.0)
        )

    # -----------------------------------------------------
    # Technical information
    # -----------------------------------------------------
    with st.expander("نمایش اطلاعات پردازش"):

        st.write("متن اولیه:")
        st.write(text)

        st.write("متن پس از پیش‌پردازش:")
        st.write(cleaned_text)

        st.write("مدل:")
        st.write(model_name)