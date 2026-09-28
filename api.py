import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "flight_price_model.joblib"
META_PATH = BASE / "models" / "metadata.json"

app = FastAPI(
    title="FlightFare AI REST API",
    description="Machine-Learning Flight Ticket Price Prediction API Service",
    version="1.0.0",
)

class FlightFeatures(BaseModel):
    country: str = Field("Pakistan", example="Pakistan")
    airline: str = Field(..., example="PIA")
    source_city: str = Field(..., example="Karachi")
    departure_time: str = Field(..., example="Morning")
    stops: str = Field(..., example="zero")
    arrival_time: str = Field(..., example="Afternoon")
    destination_city: str = Field(..., example="Islamabad")
    travel_class: str = Field(..., alias="class", example="Economy")
    duration: float = Field(..., example=2.0, gt=0)
    days_left: int = Field(..., example=15, ge=1)

    class Config:
        populate_by_name = True

_model = None
_metadata = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail="Model file not found. Please run train.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def get_metadata():
    global _metadata
    if _metadata is None:
        if not META_PATH.exists():
            raise HTTPException(status_code=500, detail="Metadata file not found. Please run train.py first.")
        _metadata = json.loads(META_PATH.read_text())
    return _metadata

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH.exists(),
        "metadata_loaded": META_PATH.exists(),
    }

@app.get("/metadata")
def metadata_info():
    return get_metadata()

@app.post("/predict")
def predict_fare(payload: FlightFeatures):
    model = get_model()
    data = payload.dict(by_alias=True)
    row = pd.DataFrame([data])

    try:
        prediction = float(model.predict(row)[0])
        currency = "PKR" if payload.country.lower() == "pakistan" else "INR"
        return {
            "country": payload.country,
            "predicted_price": round(prediction, 2),
            "currency": currency,
            "status": "success",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
