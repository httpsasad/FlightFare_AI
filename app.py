import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.live_pricing import get_live_flight_price

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "flight_price_model.joblib"
META_PATH = BASE / "models" / "metadata.json"
RESULTS_PATH = BASE / "models" / "model_results.csv"

st.set_page_config(
    page_title="FlightFare AI — Ticket Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CACHED MODEL & METADATA LOADERS ---
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    if not META_PATH.exists():
        return None
    return json.loads(META_PATH.read_text())

@st.cache_data
def load_benchmark_results():
    if not RESULTS_PATH.exists():
        return None
    return pd.read_csv(RESULTS_PATH)

model = load_model()
metadata = load_metadata()
results_df = load_benchmark_results()

# --- HEADER SECTION ---
st.title("✈️ FlightFare AI")
st.caption("Machine-learning flight ticket price estimation engine for Pakistan 🇵🇰 & India 🇮🇳")

if model is None or metadata is None:
    st.error("⚠️ Production model not found! Please run `python train.py` first to train and save the pipeline.")
    st.stop()

# --- SIDEBAR & MODEL STATS ---
st.sidebar.header("🤖 Model Metadata")
st.sidebar.metric(label="Active Model", value=metadata.get("best_model", "Unknown"))
if "best_model_metrics" in metadata:
    r2_val = metadata["best_model_metrics"].get("R2", 0.0)
    st.sidebar.metric(label="Test R² Score", value=f"{r2_val * 100:.2f}%")
    st.sidebar.metric(label="Mean Absolute Error (MAE)", value=f"{metadata['best_model_metrics'].get('MAE', 0):,.0f}")
st.sidebar.caption(f"Trained on {metadata.get('train_rows', 0):,} flight records.")

# --- OPTIONS MAP ---
PK_AIRLINES = ["PIA", "Airblue", "SereneAir", "Fly Jinnah", "AirSial", "Fly Pakistan", "Southair"]
PK_CITIES = ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta", "Multan", "Sialkot", "Faisalabad", "Bahawalpur"]

IN_AIRLINES = ["Air India", "AirAsia", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
IN_CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]

TIME_SLOTS = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
STOPS_OPTS = ["zero", "one", "two_or_more"]
CLASSES = ["Economy", "Business"]

# --- MAIN TABS ---
tab_predictor, tab_analytics = st.tabs(["✈️ Fare Predictor", "📊 Model Benchmarks & Metrics"])

with tab_predictor:
    st.subheader("Flight Details & Route Selection")
    
    country = st.radio("Select Region / Country", ["🇵🇰 Pakistan", "🇮🇳 India"], horizontal=True)
    is_pk = "Pakistan" in country
    
    airline_options = PK_AIRLINES if is_pk else IN_AIRLINES
    city_options = PK_CITIES if is_pk else IN_CITIES
    currency_label = "PKR (Rs.)" if is_pk else "INR (₹)"
    currency_symbol = "Rs." if is_pk else "₹"
    country_val = "Pakistan" if is_pk else "India"

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            airline = st.selectbox("Airline", airline_options)
            source_city = st.selectbox("Source City", city_options, index=0)
            departure_time = st.selectbox("Departure Time", TIME_SLOTS, index=1)

        with c2:
            stops = st.selectbox("Stops", STOPS_OPTS)
            arrival_time = st.selectbox("Arrival Time", TIME_SLOTS, index=2)
            destination_city = st.selectbox("Destination City", city_options, index=2 if len(city_options) > 2 else 1)

        with c3:
            travel_class = st.selectbox("Class", CLASSES)
            duration = st.number_input("Duration (hours)", min_value=0.5, max_value=50.0, value=2.0 if is_pk else 2.5, step=0.5)
            days_left = st.number_input("Days Before Departure", min_value=1, max_value=50, value=15, step=1)

        submitted = st.form_submit_button("Predict Ticket Price", use_container_width=True)

    if submitted:
        if source_city == destination_city:
            st.warning("⚠️ **Warning:** Source City and Destination City are identical. Please choose distinct cities for a valid flight route.")
        else:
            row = pd.DataFrame([{
                "country": country_val,
                "airline": airline,
                "source_city": source_city,
                "departure_time": departure_time,
                "stops": stops,
                "arrival_time": arrival_time,
                "destination_city": destination_city,
                "class": travel_class,
                "duration": duration,
                "days_left": days_left,
            }])

            try:
                prediction = float(model.predict(row)[0])
                
                with st.spinner("Fetching live prices from Google / Web Search..."):
                    live_data = get_live_flight_price(source_city, destination_city, airline, country_val)
                    
                bg_gradient = "linear-gradient(135deg, #004d40 0%, #00796b 100%)" if is_pk else "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)"
                
                if live_data["status"] == "success":
                    st.success("✅ **Real-Time Live Price Found via Web Search!**")
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
                                    padding: 25px; border-radius: 12px; text-align: center; color: white; margin-bottom: 20px;">
                            <h3 style="margin:0; font-weight: 400;">Current Live Web Price</h3>
                            <h1 style="font-size: 2.8rem; margin: 10px 0; font-weight: 700;">{currency_symbol} {live_data['price']:,.0f}</h1>
                            <p style="margin:0; opacity: 0.9;">Snippet: <i>"{live_data['snippet']}"</i></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("⚠️ Could not fetch real-time live price. Falling back to AI model prediction.")

                st.markdown(
                    f"""
                    <div style="background: {bg_gradient};
                                padding: 25px; border-radius: 12px; text-align: center; color: white; margin-top: 20px;">
                        <h3 style="margin:0; font-weight: 400;">AI Estimated Price</h3>
                        <h1 style="font-size: 2.8rem; margin: 10px 0; font-weight: 700;">{currency_symbol} {prediction:,.0f}</h1>
                        <p style="margin:0; opacity: 0.9;">Route: {source_city} ✈️ {destination_city} | Airline: {airline} ({travel_class})</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab_analytics:
    st.subheader("Model Performance Benchmark")
    st.write(
        "During model training (`python train.py`), multiple regression algorithms were "
        "trained and evaluated on the 30% held-out test split of the combined dataset. "
        "The table below details their performance sorted by R² score."
    )
    if results_df is not None:
        st.dataframe(
            results_df.style.highlight_max(subset=["R2"], color="#d4edda")
                      .highlight_min(subset=["MAE", "RMSE", "MAPE"], color="#d4edda"),
            use_container_width=True
        )
    else:
        st.info("No benchmark results found. Run training pipeline to generate comparison.")


