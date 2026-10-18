import re

ARABIC_TO_PERSIAN_MAP = {
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا", "ئ": "ی",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

DIACRITICS = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0670\u0640"

# Do not remove sentiment-critical negation words such as نه، نیست، نبود، هرگز.
PERSIAN_STOPWORDS = set("""
از به را که این آن با برای در تا هم یا و ای اما اگر یک دو سه چند
است شد شده باشد بود بودم بودی بودیم بودید بودند
او آنها ایشان ما شما من تو خود خودش خودم خودت
هر همه هیچ دیگر دیگری چون چرا چگونه کجا کی چه چیزی
روی زیر بالا پایین کنار بین میان جلوی عقب پشت
اینکه چنانچه بنابراین بلکه ولی همچنین همچنان بسیار خیلی کاملا
یک اینها آنها اینجا آنجا سپس آنگاه
شد شدند شدم شدی گشت گشتند
باشیم باشید باشند بودن هستم هستی هستیم هستید هستند
""".split())

# Words/phrases that must survive stopword filtering because they carry polarity.
NEGATION_WORDS = {"نه", "نیست", "نیستم", "نیستی", "نیستیم", "نیستید", "نیستند",
                  "نبود", "نبودم", "نبودی", "نبودیم", "نبودید", "نبودند",
                  "نشد", "نشدم", "نشدند", "هرگز", "نمی", "نمیشود", "نمی‌شود"}

PERSIAN_STOPWORDS -= NEGATION_WORDS

PERSIAN_CHARS = re.compile(r"[^ا-یآ-یءٔ‌ ]", re.UNICODE)
MULTISPACE = re.compile(r"\s+")
URL = re.compile(r"(?:https?://\S+|www\.\S+)")
MENTION = re.compile(r"[@#]\S+")
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]+",
    flags=re.UNICODE,
)


def normalize_persian_chars(text: str) -> str:
    for src, dst in ARABIC_TO_PERSIAN_MAP.items():
        text = text.replace(src, dst)
    return text.translate(str.maketrans("", "", DIACRITICS))


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = normalize_persian_chars(text)
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = EMOJI.sub(" ", text)
    text = text.replace("\u200c", " ")
    text = PERSIAN_CHARS.sub(" ", text)
    text = MULTISPACE.sub(" ", text).strip()
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in PERSIAN_STOPWORDS]
    tokens = [t for t in tokens if len(t) >= 2]
    return " ".join(tokens)


def clean_batch(texts):
    return [clean_text(t) for t in texts]
