
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_model, predict_patient


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict a model class from heart disease dataset features.",
    version="1.0.0",
)


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=1, le=120)
    trestbps: float = Field(gt=0)
    chol: float = Field(gt=0)
    thalach: float = Field(gt=0)
    oldpeak: float
    sex: Literal[0, 1]
    cp: int = Field(ge=0, le=3)
    fbs: Literal[0, 1]
    restecg: int = Field(ge=0, le=2)
    exang: Literal[0, 1]
    slope: int = Field(ge=0, le=3)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=7)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PatientInput):
    try:
        result = predict_patient(data.model_dump())
        return result
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Saved model artifact is missing.",
        ) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
