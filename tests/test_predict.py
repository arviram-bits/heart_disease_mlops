
import numpy as np
import pytest

from src.predict import FEATURES, predict_patient


class DummyModel:
    """A fake classifier for testing inference logic."""

    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.20, 0.80]])


@pytest.fixture
def valid_patient():
    return {
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
        "thal": 3,
    }


def test_valid_prediction(valid_patient):
    result = predict_patient(valid_patient, model=DummyModel())

    assert result["prediction"] == 1
    assert result["label"] == "Disease present"
    assert result["confidence"] == pytest.approx(0.80)
    assert result["probability_disease_present"] == pytest.approx(0.80)


def test_missing_feature_is_rejected(valid_patient):
    valid_patient.pop("age")

    with pytest.raises(ValueError, match="Missing features"):
        predict_patient(valid_patient, model=DummyModel())


def test_unexpected_feature_is_rejected(valid_patient):
    valid_patient["unknown_feature"] = 123

    with pytest.raises(ValueError, match="Unexpected features"):
        predict_patient(valid_patient, model=DummyModel())


def test_missing_value_is_rejected(valid_patient):
    valid_patient["chol"] = np.nan

    with pytest.raises(ValueError, match="must not contain missing"):
        predict_patient(valid_patient, model=DummyModel())


def test_all_expected_features_are_defined():
    assert len(FEATURES) == 13
    assert len(set(FEATURES)) == 13
