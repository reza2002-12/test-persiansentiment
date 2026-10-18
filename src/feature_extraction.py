from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=15000,
        sublinear_tf=True,
    )
