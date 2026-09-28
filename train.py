"""
FlightFare AI - Model Training & Benchmarking Pipeline.

Expected dataset:
data/Clean_Dataset.csv

The pipeline automatically cleans the dataset, filters out index/id columns,
preprocesses numeric and categorical features, benchmarks 12 regressors (including XGBoost, LightGBM, CatBoost),
and saves the highest-performing pipeline and metadata.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "Clean_Dataset.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop index, high-cardinality flight identifier, and currency columns if present
    drop_patterns = ["unnamed", "flight", "id", "flight_id", "index", "currency"]
    cols_to_drop = [c for c in df.columns if any(p in c.lower() for p in drop_patterns)]
    df = df.drop(columns=cols_to_drop, errors="ignore")


    if "price" not in df.columns:
        raise ValueError("Target column 'price' was not found.")

    df = df.dropna(subset=["price"])
    df = df.drop_duplicates()

    # Convert obvious numeric columns if loaded as strings.
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.95:
            df[col] = converted

    return df


def build_preprocessor(X: pd.DataFrame):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = X.select_dtypes(include=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")

    return preprocessor


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download Clean_Dataset.csv and place it in data/."
        )

    df = clean_data(pd.read_csv(DATA_PATH))
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Bagging": BaggingRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.08,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=300,
            learning_rate=0.08,
            random_seed=42,
            verbose=0,
        )

    results = []
    fitted = {}

    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}\n")

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator),
        ])

        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        row = {"Model": name, **compute_metrics(y_test, pred)}
        results.append(row)
        fitted[name] = pipe

    result_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\n=== MODEL BENCHMARK RESULTS ===")
    print(result_df.to_string(index=False))

    best_name = result_df.iloc[0]["Model"]
    best_pipeline = fitted[best_name]

    joblib.dump(best_pipeline, MODEL_DIR / "flight_price_model.joblib")
    result_df.to_csv(MODEL_DIR / "model_results.csv", index=False)

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    categorical_options = {}
    for col in categorical_cols:
        categorical_options[col] = sorted(X[col].unique().tolist())

    metadata = {
        "target": "price",
        "best_model": best_name,
        "features": X.columns.tolist(),
        "categorical_features": categorical_cols,
        "categorical_options": categorical_options,
        "numeric_features": X.select_dtypes(include=np.number).columns.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model_metrics": result_df.iloc[0].to_dict(),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n[BEST MODEL]: {best_name} (R^2 = {result_df.iloc[0]['R2']:.4f})")
    print("Saved: models/flight_price_model.joblib")
    print("Saved: models/model_results.csv")
    print("Saved: models/metadata.json")



if __name__ == "__main__":
    main()

