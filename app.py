
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"

app = FastAPI(title="Iris Flower Prediction API")

model = None
scaler = None
encoder = None


class FlowerInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.on_event("startup")
def load_artifacts():
    global model, scaler, encoder

    required_files = [MODEL_PATH, SCALER_PATH, ENCODER_PATH]
    missing = [str(path.name) for path in required_files if not path.exists()]

    if missing:
        raise RuntimeError(
            f"Missing model artifacts: {missing}. Train the model first."
        )

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)


@app.get("/")
def home():
    return {"message": "Iris Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(flower: FlowerInput):
    values = np.array([[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width,
    ]], dtype=float)

    if not np.isfinite(values).all():
        raise HTTPException(status_code=422, detail="Measurements must be finite numbers.")

    scaled_values = scaler.transform(values)
    probabilities = model.predict(scaled_values, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    species = str(encoder.inverse_transform([class_index])[0])

    return {
        "predicted_species": species,
        "confidence": round(float(probabilities[class_index]), 4),
    }