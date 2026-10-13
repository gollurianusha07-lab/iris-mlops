
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Iris Flower Prediction")

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

    model = load_model(BASE_DIR / "model.keras")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    encoder = joblib.load(BASE_DIR / "encoder.pkl")


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iris Flower Predictor</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #e0f2fe, #f3e8ff);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 24px;
    color: #1e293b;
}
.card {
    background: white;
    width: 100%;
    max-width: 560px;
    padding: 32px;
    border-radius: 20px;
    box-shadow: 0 15px 45px rgba(30, 41, 59, 0.12);
}
h1 { text-align: center; margin: 0 0 10px; }
.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 28px;
}
.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
}
label {
    display: block;
    font-size: 14px;
    font-weight: bold;
    margin-bottom: 8px;
}
input {
    width: 100%;
    padding: 12px;
    border: 1px solid #cbd5e1;
    border-radius: 9px;
    font-size: 16px;
}
input:focus {
    outline: 2px solid #818cf8;
    border-color: #6366f1;
}
button {
    width: 100%;
    padding: 14px;
    margin-top: 24px;
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
}
button:hover { background: #4338ca; }
button:disabled { opacity: 0.6; cursor: wait; }
#result {
    display: none;
    margin-top: 22px;
    padding: 18px;
    border-radius: 12px;
    background: #ecfdf5;
    text-align: center;
}
#species { font-size: 23px; font-weight: bold; color: #047857; }
#error { color: #b91c1c; margin-top: 16px; text-align: center; }
footer {
    text-align: center;
    margin-top: 24px;
    color: #64748b;
    font-size: 12px;
}
@media (max-width: 450px) {
    .grid { grid-template-columns: 1fr; }
    .card { padding: 22px; }
}
</style>
</head>
<body>
<div class="card">
    <h1>🌸 Iris Flower Predictor</h1>
    <p class="subtitle">Enter flower measurements to predict its species.</p>

    <form id="predictionForm">
        <div class="grid">
            <div>
                <label for="sl">Sepal Length (cm)</label>
                <input id="sl" type="number" min="0.1" step="any"
                       value="5.1" required>
            </div>
            <div>
                <label for="sw">Sepal Width (cm)</label>
                <input id="sw" type="number" min="0.1" step="any"
                       value="3.5" required>
            </div>
            <div>
                <label for="pl">Petal Length (cm)</label>
                <input id="pl" type="number" min="0.1" step="any"
                       value="1.4" required>
            </div>
            <div>
                <label for="pw">Petal Width (cm)</label>
                <input id="pw" type="number" min="0.1" step="any"
                       value="0.2" required>
            </div>
        </div>

        <button id="predictButton" type="submit">Predict Flower</button>
    </form>

    <div id="result" aria-live="polite">
        <p>Predicted Species</p>
        <div id="species"></div>
        <p>Confidence: <strong id="confidence"></strong></p>
    </div>
    <div id="error" role="alert"></div>

    <footer>Powered by a neural network · FastAPI · Docker</footer>
</div>

<script>
const form = document.getElementById("predictionForm");
const button = document.getElementById("predictButton");
const result = document.getElementById("result");
const errorBox = document.getElementById("error");

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    result.style.display = "none";
    errorBox.textContent = "";
    button.disabled = true;
    button.textContent = "Predicting...";

    const values = [
        Number(document.getElementById("sl").value),
        Number(document.getElementById("sw").value),
        Number(document.getElementById("pl").value),
        Number(document.getElementById("pw").value)
    ];

    try {
        if (!values.every(v => Number.isFinite(v) && v > 0)) {
            throw new Error("Please enter valid positive measurements.");
        }

        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                sepal_length: values[0],
                sepal_width: values[1],
                petal_length: values[2],
                petal_width: values[3]
            })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        document.getElementById("species").textContent =
            data.predicted_species;
        document.getElementById("confidence").textContent =
            (data.confidence * 100).toFixed(2) + "%";
        result.style.display = "block";
    } catch (error) {
        errorBox.textContent = error.message || "Could not connect to the API.";
    } finally {
        button.disabled = false;
        button.textContent = "Predict Flower";
    }
});
</script>
</body>
</html>
""")


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

    if not np.isfinite(values).all() or (values <= 0).any():
        raise HTTPException(
            status_code=422,
            detail="Measurements must be finite positive numbers."
        )

    scaled_values = scaler.transform(values)
    probabilities = model.predict(scaled_values, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    species = str(encoder.inverse_transform([class_index])[0])

    return {
        "predicted_species": species,
        "confidence": round(float(probabilities[class_index]), 4),
    }
