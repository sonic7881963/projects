
import json
import joblib
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

model = joblib.load(ARTIFACT_DIR / "text_classifier.joblib")
with open(ARTIFACT_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

label_map = {int(k): v for k, v in metadata["labels"].items()}

app = FastAPI(
    title="Module 3 NLP Classifier API",
    description="A microservice that classifies text into one of four categories from the earlier assignment.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify")

class PredictionResponse(BaseModel):
    predicted_label_id: int
    predicted_label_name: str
    class_probabilities: Dict[str, float]

@app.get("/")
def root():
    return {
        "message": "NLP classifier API is running.",
        "available_endpoints": ["/health", "/predict", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": metadata["model_name"],
        "labels": metadata["labels"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    text = request.text.strip()
    pred_id = int(model.predict([text])[0])
    probas = model.predict_proba([text])[0]

    prob_dict = {
        label_map[i]: round(float(probas[i]), 6)
        for i in range(len(probas))
    }

    return PredictionResponse(
        predicted_label_id=pred_id,
        predicted_label_name=label_map[pred_id],
        class_probabilities=prob_dict
    )
