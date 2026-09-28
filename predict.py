from pathlib import Path
from functools import lru_cache
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "flight_price_model.joblib"

@lru_cache(maxsize=1)
def _get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train the model first with: python train.py")
    return joblib.load(MODEL_PATH)

def predict_price(features: dict) -> float:
    """
    Predict flight ticket price given input features.

    Parameters:
    - features (dict): Dictionary with keys:
      ['airline', 'source_city', 'departure_time', 'stops',
       'arrival_time', 'destination_city', 'class', 'duration', 'days_left']

    Returns:
    - float: Predicted fare in INR (₹)
    """
    model = _get_model()
    X = pd.DataFrame([features])
    return float(model.predict(X)[0])

if __name__ == "__main__":
    sample = {
        "country": "Pakistan",
        "airline": "PIA",
        "source_city": "Karachi",
        "departure_time": "Morning",
        "stops": "zero",
        "arrival_time": "Afternoon",
        "destination_city": "Islamabad",
        "class": "Economy",
        "duration": 2.0,
        "days_left": 15,
    }
    fare = predict_price(sample)
    print(f"Sample Predicted Fare: PKR {fare:,.2f}")



