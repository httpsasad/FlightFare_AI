import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "Clean_Dataset.csv"

def generate_flight_dataset(num_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 🇵🇰 Pakistan Specs
    pk_airlines = ["PIA", "Airblue", "SereneAir", "Fly Jinnah", "AirSial", "Fly Pakistan", "Southair"]
    pk_cities = ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta", "Multan", "Sialkot", "Faisalabad", "Bahawalpur"]
    
    # 🇮🇳 India Specs
    in_airlines = ["Air India", "AirAsia", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
    in_cities = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]

    time_slots = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    stops_opts = ["zero", "one", "two_or_more"]
    classes = ["Economy", "Business"]
    
    records = []
    
    for i in range(num_samples):
        country = np.random.choice(["Pakistan", "India"], p=[0.5, 0.5])
        
        if country == "Pakistan":
            airline = np.random.choice(pk_airlines, p=[0.25, 0.20, 0.18, 0.12, 0.15, 0.05, 0.05])
            src = np.random.choice(pk_cities)
            dest = np.random.choice([c for c in pk_cities if c != src])
            base_price = 16000.0  # PKR Base
            duration = np.round(np.random.uniform(1.0, 14.0), 2)
            currency = "PKR"
        else:
            airline = np.random.choice(in_airlines, p=[0.25, 0.15, 0.10, 0.20, 0.10, 0.20])
            src = np.random.choice(in_cities)
            dest = np.random.choice([c for c in in_cities if c != src])
            base_price = 4500.0   # INR Base
            duration = np.round(np.random.uniform(1.0, 25.0), 2)
            currency = "INR"

        dep_time = np.random.choice(time_slots)
        arr_time = np.random.choice(time_slots)
        stops = np.random.choice(stops_opts, p=[0.5, 0.4, 0.1])
        travel_class = np.random.choice(classes, p=[0.7, 0.3])
        days_left = np.random.randint(1, 50)
        flight_code = f"{airline[:3].upper()}-{np.random.randint(100, 999)}"

        class_mult = 4.2 if travel_class == "Business" else 1.0
        stops_mult = 1.0 if stops == "zero" else (1.35 if stops == "one" else 1.6)
        days_mult = 1.5 - 0.5 * np.exp(-0.03 * (days_left - 1))
        
        duration_cost = duration * (800.0 if country == "Pakistan" else 250.0)
        noise = np.random.normal(0, base_price * 0.05)
        
        price = (base_price * class_mult * stops_mult * days_mult) + duration_cost + noise
        min_floor = 10000.0 if country == "Pakistan" else 2000.0
        price = np.maximum(price, min_floor)

        records.append({
            "country": country,
            "airline": airline,
            "flight": flight_code,
            "source_city": src,
            "departure_time": dep_time,
            "stops": stops,
            "arrival_time": arr_time,
            "destination_city": dest,
            "class": travel_class,
            "duration": duration,
            "days_left": days_left,
            "currency": currency,
            "price": np.round(price, 0),
        })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    print(f"Generating synthetic flight dataset with Pakistan & India routes ({CSV_PATH})...")
    df = generate_flight_dataset(num_samples=10000)
    df.to_csv(CSV_PATH, index=False)
    print(f"[SUCCESS] Created dataset with {len(df):,} records (Pakistan & India) at {CSV_PATH}")


