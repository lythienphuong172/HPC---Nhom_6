import os
import time
import joblib
import shap
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# 1. PATH CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "model", "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_PATH = os.path.join(MODEL_DIR, "ffnn.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
NUMERIC_PATH = os.path.join(MODEL_DIR, "numeric_cols.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "ffnn_threshold.pkl")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# =========================
# 2. LOAD MODEL + OBJECTS
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURE_PATH)
numeric_cols = joblib.load(NUMERIC_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))

# =========================
# 3. LOAD BACKGROUND DATA CHO SHAP
# =========================

test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop("isFraud", axis=1)
X_test = X_test[feature_names].astype("float32")

X_background = X_test.sample(
    n=min(100, len(X_test)),
    random_state=42
)


def predict_fn(x):
    x = np.array(x).astype("float32")
    return model.predict(x, verbose=0).flatten()


try:
    explainer = shap.KernelExplainer(
        predict_fn,
        X_background.values
    )
    EXPLAINER_TYPE = "KernelExplainer"
except Exception as e:
    explainer = None
    EXPLAINER_TYPE = f"SHAP init failed: {str(e)}"

# =========================
# 4. STATS STORAGE
# =========================

API_STATS = {
    "total_predictions": 0,
    "fraud_detected": 0,
    "total_latency_ms": 0.0
}

# =========================
# 5. FASTAPI APP
# =========================

app = FastAPI(
    title="Fraud Detection API",
    description="API dự đoán gian lận giao dịch PaySim bằng FFNN và giải thích bằng SHAP",
    version="1.0"
)

# =========================
# 6. INPUT SCHEMA
# =========================

class TransactionInput(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    nameDest: str


# =========================
# 7. PREPROCESS INPUT
# =========================

def preprocess_input(data: TransactionInput):

    df = pd.DataFrame([data.model_dump()])

    # Feature engineering giống preprocessing.py
    df["isMerchant"] = df["nameDest"].str.startswith("M").astype(int)

    df["day"] = df["step"] // 24

    df["errorBalanceOrig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )

    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    df["isOrigBalanceZeroAfter"] = (
        df["newbalanceOrig"] == 0
    ).astype(int)

    # One-hot encoding cho type
    type_columns = [
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER"
    ]

    for col in type_columns:
        df[col] = 0

    type_value = df.loc[0, "type"]
    type_col = "type_" + type_value

    if type_col in type_columns:
        df[type_col] = 1

    # Drop cột không dùng
    df = df.drop(["type", "nameDest"], axis=1)

    # Đảm bảo đủ feature và đúng thứ tự
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    # Scale đúng như preprocessing.py
    # Chỉ scale numeric_cols, không scale one-hot/binary
    df_scaled = df.copy()

    numeric_input = df[numeric_cols].astype("float32")
    scaled_numeric = scaler.transform(numeric_input)

    df_scaled.loc[:, numeric_cols] = scaled_numeric

    return df_scaled.values.astype("float32"), df


# =========================
# 8. ROUTES
# =========================

@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "best_model": "FFNN",
        "threshold": float(threshold),
        "xai": EXPLAINER_TYPE,
        "available_routes": [
            "/",
            "/health",
            "/stats",
            "/predict",
            "/explain",
            "/debug"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "API is healthy",
        "version": "1.0",
        "model": "FFNN",
        "threshold": float(threshold),
        "xai": EXPLAINER_TYPE
    }


@app.get("/stats")
def stats():
    total = API_STATS["total_predictions"]
    fraud = API_STATS["fraud_detected"]

    avg_latency = (
        API_STATS["total_latency_ms"] / total
        if total > 0 else 0.0
    )

    fraud_rate = (
        fraud / total * 100
        if total > 0 else 0.0
    )

    return {
        "total_predictions": total,
        "fraud_detected": fraud,
        "fraud_rate_pct": fraud_rate,
        "avg_latency_ms": avg_latency
    }


@app.post("/predict")
def predict(data: TransactionInput):

    start_time = time.perf_counter()

    X_scaled, X_raw = preprocess_input(data)

    fraud_prob = model.predict(X_scaled, verbose=0)[0][0]

    prediction = int(fraud_prob >= threshold)

    result = "Fraud" if prediction == 1 else "Not Fraud"

    latency_ms = (time.perf_counter() - start_time) * 1000

    API_STATS["total_predictions"] += 1
    API_STATS["fraud_detected"] += prediction
    API_STATS["total_latency_ms"] += latency_ms

    return {
        "fraud_probability": float(fraud_prob),
        "threshold": float(threshold),
        "prediction": prediction,
        "result": result,
        "processing_time_ms": latency_ms
    }


@app.post("/explain")
def explain(data: TransactionInput):

    if explainer is None:
        return {
            "error": "SHAP explainer is not available",
            "xai_status": EXPLAINER_TYPE
        }

    X_scaled, X_raw = preprocess_input(data)

    fraud_prob = model.predict(X_scaled, verbose=0)[0][0]

    shap_values = explainer.shap_values(
        X_scaled,
        nsamples=100
    )

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    if shap_values.ndim == 1:
        local_shap = shap_values
    else:
        local_shap = shap_values[0]

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "value": X_raw.values.flatten(),
        "shap_value": local_shap
    })

    explanation_df["abs_shap"] = explanation_df["shap_value"].abs()

    explanation_df = explanation_df.sort_values(
        by="abs_shap",
        ascending=False
    )

    top_features = explanation_df.head(10)[
        ["feature", "value", "shap_value"]
    ].to_dict(orient="records")

    prediction = int(fraud_prob >= threshold)

    return {
        "fraud_probability": float(fraud_prob),
        "threshold": float(threshold),
        "prediction": prediction,
        "result": "Fraud" if prediction == 1 else "Not Fraud",
        "xai_method": EXPLAINER_TYPE,
        "top_explanation": top_features
    }


@app.get("/debug")
def debug():
    return {
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "n_features": len(feature_names),
        "n_numeric_cols": len(numeric_cols),
        "threshold": float(threshold)
    }