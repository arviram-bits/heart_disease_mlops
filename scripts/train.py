from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "heart_disease_clean.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]
FEATURES = NUMERIC + CATEGORICAL


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if not set(FEATURES + ["target"]).issubset(df.columns):
        raise ValueError("Dataset is missing required columns")

    df = df.dropna(subset=["target"]).copy()
    X = df[FEATURES]
    y = df["target"].astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError("Both target classes need at least 5 records")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])

    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {"classifier__C": [0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=42, class_weight="balanced"
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5],
            },
        ),
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    results = []
    best_score = -1
    best_model = None
    best_name = None

    for name, (classifier, params) in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        search = GridSearchCV(
            pipeline,
            params,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        model = search.best_estimator_
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "cv_roc_auc": search.best_score_,
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_roc_auc": roc_auc_score(y_test, probabilities),
            "best_parameters": str(search.best_params_),
        })

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = model
            best_name = name

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        best_model,
        ARTIFACT_DIR / "heart_disease_pipeline.joblib",
    )
    pd.DataFrame(results).to_csv(
        REPORT_DIR / "ci_training_results.csv", index=False
    )

    print("Selected model:", best_name)
    print("Best CV ROC-AUC:", best_score)
    print("Training results saved.")


if __name__ == "__main__":
    main()
