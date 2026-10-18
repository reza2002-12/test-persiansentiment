from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_models(random_state=42):
    return {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=2.0, class_weight="balanced", random_state=random_state
        ),
        "Linear SVM": SVC(
            kernel="linear", C=1.5, probability=True, class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1
        ),
    }
