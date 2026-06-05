from fastapi import FastAPI
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    image_base64: str
    language: str = "English"


app = FastAPI(title="Automated Sign Language Recognition API")


@app.get("/health")
def health():
    return {"status": "ready", "stack": ["FastAPI", "TensorFlow/Keras", "OpenCV", "MediaPipe"]}


@app.post("/predict")
def predict(payload: PredictionRequest):
    # Replace this deterministic placeholder with loaded Keras weights for deployment.
    return {
        "sign": "Open Palm",
        "translation": "Hello" if payload.language == "English" else "Localized greeting",
        "confidence": 94.2,
    }