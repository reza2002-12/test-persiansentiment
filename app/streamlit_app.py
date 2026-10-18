from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DISPLAY_LABELS
from predictor import SentimentPredictor

st.set_page_config(page_title="تحلیل احساسات فارسی", page_icon="🎭", layout="centered")
st.markdown("""
<style>
.stApp, textarea, .stMarkdown, .stTextArea, .stTextInput {direction: rtl; text-align: right;}
</style>
""", unsafe_allow_html=True)

st.title("🎭 سامانه تشخیص و تحلیل احساسات در متون فارسی")
st.caption("پروژه کارشناسی مهندسی کامپیوتر — روش‌های یادگیری ماشین")

@st.cache_resource
def load_predictor():
    return SentimentPredictor()

try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error("مدل نهایی پیدا نشد. ابتدا train.py را اجرا کنید.")
    st.stop()

st.info(f"مدل فعال: **{predictor.model_name}**")
text = st.text_area("متن فارسی را وارد کنید:", height=160, placeholder="مثلاً: این محصول کیفیت بسیار خوبی دارد و از خرید آن راضی هستم.")

if st.button("تحلیل احساس", type="primary"):
    if not text.strip():
        st.warning("لطفاً یک متن وارد کنید.")
    else:
        label, confidence, probs, cleaned = predictor.predict(text)
        if label is None:
            st.warning("متن پس از پیش‌پردازش خالی شد و قابل تحلیل نیست.")
        else:
            st.subheader(f"احساس تشخیص داده‌شده: {DISPLAY_LABELS.get(label, label)}")
            st.metric("میزان اطمینان مدل", f"{confidence * 100:.2f}%")

            st.write("توزیع احتمال بین کلاس‌ها:")
            for cls in ["positive", "neutral", "negative"]:
                if cls in probs:
                    st.write(DISPLAY_LABELS[cls])
                    st.progress(min(max(probs[cls], 0.0), 1.0))

            with st.expander("نمایش متن پس از پیش‌پردازش"):
                st.write(cleaned)

st.divider()
st.caption("این سامانه برای پروژه کارشناسی طراحی و پیاده‌سازی شده است.")
