# ✈️ FlightFare AI — Flight Ticket Price Prediction

An end-to-end machine learning project that predicts airline ticket prices from flight attributes such as airline, route, stops, departure/arrival time, class, duration, and days remaining before departure.

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Data cleaning and feature filtering (automatic removal of index and non-predictive IDs)
- Preprocessing pipeline (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`)
- Benchmarking of 12 regression algorithms (including XGBoost, LightGBM, CatBoost, Random Forest)
- Evaluation metrics: MAE, MSE, RMSE, R², and MAPE
- Automatic best-model selection & metadata export
- Cached Streamlit prediction dashboard & performance metrics tab
- FastAPI REST Service (`api.py`) with Swagger documentation (`/docs`)
- CLI Python interface (`predict.py`)

## 📊 Dataset

This project is designed for the Kaggle **Flight Price Prediction** dataset:

https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

Expected file path:

`data/Clean_Dataset.csv`

## 🧠 Models Benchmarked

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree
5. Random Forest
6. Extra Trees
7. Gradient Boosting
8. XGBoost
9. LightGBM
10. CatBoost
11. KNN
12. Bagging Regressor

## 🏆 Model Selection

The training pipeline evaluates every model on the same 30% held-out test set and selects the model with the highest R² score.

## 🛠️ Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Put `Clean_Dataset.csv` inside `data/`.

## ▶️ Train Model

```bash
python train.py
```

The trained model and metadata are saved to `models/`.

## 🌐 Run Streamlit Dashboard

```bash
streamlit run app.py
```

## ⚡ Run REST API Service

```bash
python api.py
# Or using uvicorn directly:
uvicorn api:app --reload --port 8000
```

Access Swagger UI interactive docs at: `http://localhost:8000/docs`

## 📁 Project Structure

```text
FlightFare_AI_Project/
├── app.py          # Streamlit UI Dashboard
├── train.py        # ML Training & Benchmarking Pipeline
├── predict.py      # Python Module API / CLI Interface
├── api.py          # FastAPI REST API Service
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Clean_Dataset.csv
├── models/         # Auto-generated artifacts (flight_price_model.joblib, metadata.json, model_results.csv)
└── notebooks/
    └── flight_price_prediction_eda.ipynb
```

## ⚠️ Important

This model predicts prices based on historical patterns in the supplied dataset. It should be presented as a portfolio ML project, not as a live airline-pricing or guaranteed-fare system.

