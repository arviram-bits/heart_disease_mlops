
from pathlib import Path

import joblib
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "heart_disease_pipeline.joblib"
)

# Input features expected by the trained model
FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]


def load_model():
    """Load the saved preprocessing and classification pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_patient(patient_data, model=None):
    """
    Predict the binary class for one patient.

    patient_data must contain all 13 input features.
    """
    missing = set(FEATURES) - set(patient_data.keys())
    extra = set(patient_data.keys()) - set(FEATURES)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    if extra:
        raise ValueError(f"Unexpected features: {sorted(extra)}")

    # Ensure the same feature names and order as training
    patient_df = pd.DataFrame(
        [[patient_data[col] for col in FEATURES]],
        columns=FEATURES
    )

    if patient_df.isna().any().any():
        raise ValueError("Patient input must not contain missing values.")

    if model is None:
        model = load_model()

    prediction = int(model.predict(patient_df)[0])
    probabilities = model.predict_proba(patient_df)[0]

    return {
        "prediction": prediction,
        "label": (
            "Disease present" if prediction == 1
            else "No disease detected"
        ),
        "confidence": float(probabilities[prediction]),
        "probability_disease_present": float(probabilities[1])
    }


if __name__ == "__main__":
    # Example values for software testing only.
    # These are not a real patient record or clinical assessment.
    sample_patient = {
        "age": 55,
        "trestbps": 130,
        "chol": 240,
        "thalach": 150,
        "oldpeak": 1.0,
        "sex": 1,
        "cp": 2,
        "fbs": 0,
        "restecg": 1,
        "exang": 0,
        "slope": 2,
        "ca": 0,
        "thal": 3
    }

    result = predict_patient(sample_patient)
    print(result)
