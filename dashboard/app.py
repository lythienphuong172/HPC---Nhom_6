from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# =========================
# 1. CONFIG
# =========================

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]

NORMAL_SAMPLE = {
    "step": 10,
    "type": "PAYMENT",
    "amount": 250.0,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4750.0,
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 1250.0,
    "nameDest": "M1979787155",
}

FRAUD_SAMPLE = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "nameDest": "C553264065",
}

HIGH_AMOUNT_SAMPLE = {
    "step": 250,
    "type": "CASH_OUT",
    "amount": 1_000_000.0,
    "oldbalanceOrg": 1_000_000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 50_000.0,
    "newbalanceDest": 1_050_000.0,
    "nameDest": "C1286084959",
}

PRESETS = {
    "Giao dịch bình thường": NORMAL_SAMPLE,
    "Giao dịch nghi gian lận": FRAUD_SAMPLE,
    "Giao dịch giá trị lớn": HIGH_AMOUNT_SAMPLE,
}

# =========================
# 2. PAGE SETUP
# =========================

st.set_page_config(
    page_title="Fraud Detection DL Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .small-note {font-size: 0.88rem; opacity: .75;}
    .risk-safe {
        padding: 12px 16px;
        border-radius: 14px;
        background: rgba(22,163,74,.13);
        border: 1px solid rgba(22,163,74,.3);
    }
    .risk-fraud {
        padding: 12px 16px;
        border-radius: 14px;
        background: rgba(220,38,38,.13);
        border: 1px solid rgba(220,38,38,.3);
    }
    .risk-mid {
        padding: 12px 16px;
        border-radius: 14px;
        background: rgba(245,158,11,.13);
        border: 1px solid rgba(245,158,11,.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 3. API HELPERS
# =========================

def api_get(path: str) -> Optional[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        return res.json()
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        return None


def api_post(path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        res = requests.post(f"{API_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        return res.json()
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        return None


def normalize_predict_response(response: Dict[str, Any]) -> Dict[str, Any]:
    prob = float(response.get("fraud_probability", 0.0) or 0.0)
    threshold = float(response.get("threshold", 0.5) or 0.5)

    if "is_fraud" in response:
        is_fraud = bool(response["is_fraud"])
    else:
        is_fraud = bool(int(response.get("prediction", 0)))

    risk_level = response.get("risk_level")
    if not risk_level:
        if prob >= threshold:
            risk_level = "HIGH"
        elif prob >= threshold * 0.7:
            risk_level = "MEDIUM"
        elif prob >= threshold * 0.4:
            risk_level = "LOW"
        else:
            risk_level = "SAFE"

    return {
        "fraud_probability": prob,
        "is_fraud": is_fraud,
        "prediction": int(is_fraud),
        "result": response.get("result", "Fraud" if is_fraud else "Not Fraud"),
        "threshold": threshold,
        "risk_level": risk_level,
        "raw": response,
    }


def make_gauge(probability: float, threshold: float = 0.5) -> go.Figure:
    value = probability * 100
    threshold_value = threshold * 100

    if probability >= threshold:
        bar_color = "#dc2626"
    elif probability >= threshold * 0.7:
        bar_color = "#f59e0b"
    else:
        bar_color = "#16a34a"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Xác suất gian lận", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, threshold_value * 0.4], "color": "rgba(22,163,74,.20)"},
                    {"range": [threshold_value * 0.4, threshold_value], "color": "rgba(245,158,11,.22)"},
                    {"range": [threshold_value, 100], "color": "rgba(220,38,38,.20)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.8,
                    "value": threshold_value,
                },
            },
        )
    )

    fig.update_layout(height=270, margin=dict(l=15, r=15, t=45, b=10))
    return fig


def make_shap_chart(items: List[Dict[str, Any]]) -> Optional[go.Figure]:
    if not items:
        return None

    df = pd.DataFrame(items)

    if not {"feature", "shap_value"}.issubset(df.columns):
        return None

    df["impact_abs"] = df["shap_value"].abs()
    df = df.sort_values("impact_abs", ascending=True).tail(10)
    df["direction"] = df["shap_value"].apply(
        lambda x: "Tăng rủi ro" if x > 0 else "Giảm rủi ro"
    )

    fig = px.bar(
        df,
        x="shap_value",
        y="feature",
        orientation="h",
        color="direction",
        hover_data=["value", "impact_abs"],
        title="Top yếu tố ảnh hưởng theo SHAP",
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="Tác động",
    )

    return fig


def build_payload_from_form(defaults: Dict[str, Any]) -> Dict[str, Any]:
    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Step / giờ giao dịch",
            min_value=1,
            value=int(defaults["step"]),
            step=1,
        )

        tx_type = st.selectbox(
            "Loại giao dịch",
            TRANSACTION_TYPES,
            index=TRANSACTION_TYPES.index(defaults["type"]),
        )

        amount = st.number_input(
            "Số tiền giao dịch",
            min_value=0.0,
            value=float(defaults["amount"]),
            step=100.0,
        )

        name_dest = st.text_input(
            "Tên tài khoản nhận - nameDest",
            value=str(defaults["nameDest"]),
        )

    with col2:
        old_org = st.number_input(
            "Số dư người gửi trước giao dịch",
            min_value=0.0,
            value=float(defaults["oldbalanceOrg"]),
            step=100.0,
        )

        new_org = st.number_input(
            "Số dư người gửi sau giao dịch",
            min_value=0.0,
            value=float(defaults["newbalanceOrig"]),
            step=100.0,
        )

        old_dest = st.number_input(
            "Số dư người nhận trước giao dịch",
            min_value=0.0,
            value=float(defaults["oldbalanceDest"]),
            step=100.0,
        )

        new_dest = st.number_input(
            "Số dư người nhận sau giao dịch",
            min_value=0.0,
            value=float(defaults["newbalanceDest"]),
            step=100.0,
        )

    return {
        "step": int(step),
        "type": tx_type,
        "amount": float(amount),
        "oldbalanceOrg": float(old_org),
        "newbalanceOrig": float(new_org),
        "oldbalanceDest": float(old_dest),
        "newbalanceDest": float(new_dest),
        "nameDest": name_dest,
    }


# =========================
# 4. SESSION STATE
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None

if "last_payload" not in st.session_state:
    st.session_state.last_payload = NORMAL_SAMPLE.copy()

if "last_error" not in st.session_state:
    st.session_state.last_error = ""


# =========================
# 5. SIDEBAR
# =========================

with st.sidebar:
    st.title("⚙️ Cấu hình")
    st.caption("Dashboard gọi REST API FastAPI để dự đoán và giải thích giao dịch.")
    st.code(API_URL, language="text")

    st.divider()

    st.subheader("Luồng hệ thống")
    st.markdown(
        """
        1. Nhập giao dịch PaySim  
        2. Dashboard gửi `/predict`  
        3. API tiền xử lý + scale  
        4. FFNN dự đoán xác suất gian lận  
        5. Dashboard gọi `/explain` để lấy SHAP  
        """
    )

    st.divider()

    if st.button("🧹 Xóa lịch sử", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_prediction = None
        st.session_state.last_explanation = None
        st.rerun()


# =========================
# 6. HEADER + STATUS
# =========================

st.title("Fraud Detection Deep Learning Dashboard")
st.caption(
    "Dashboard tương tác cho đồ án phát hiện gian lận giao dịch tài chính với FastAPI, FFNN và SHAP XAI."
)

home = api_get("/")
health = api_get("/health")
stats = api_get("/stats")

if not home and not health:
    st.error(
        "Không kết nối được API. Hãy chạy FastAPI trước: "
        "`python -m uvicorn api.main:app --reload`"
    )

    if st.session_state.last_error:
        st.code(st.session_state.last_error)

    st.stop()

status_cols = st.columns(4)

status_cols[0].success("API Online")
status_cols[1].info(f"Model: {(home or {}).get('best_model', 'FFNN')}")
status_cols[2].info(f"XAI: {(home or {}).get('xai', 'N/A')}")
status_cols[3].info(f"Version: {(home or health or {}).get('version', '1.0')}")

if stats:
    st.subheader("Metrics hệ thống")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Tổng predict", f"{stats.get('total_predictions', 0):,}")
    m2.metric("Fraud phát hiện", f"{stats.get('fraud_detected', 0):,}")
    m3.metric("Fraud rate", f"{stats.get('fraud_rate_pct', 0):.2f}%")
    m4.metric("Avg latency", f"{stats.get('avg_latency_ms', 0):.2f} ms")
else:
    st.info("API hiện tại chưa có `/stats`, dashboard vẫn hoạt động với `/predict` và `/explain`.")

st.divider()


# =========================
# 7. MAIN TABS
# =========================

tab_predict, tab_batch, tab_history = st.tabs(
    ["Dự đoán giao dịch", "Test nhanh nhiều mẫu", "Lịch sử"]
)


with tab_predict:
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Nhập thông tin giao dịch")

        preset_name = st.radio(
            "Chọn mẫu nhập nhanh",
            list(PRESETS.keys()),
            horizontal=True,
        )

        defaults = PRESETS[preset_name]

        with st.form("predict_form"):
            payload = build_payload_from_form(defaults)

            c_submit, c_explain = st.columns(2)

            submit = c_submit.form_submit_button(
                "Predict",
                use_container_width=True,
            )

            submit_explain = c_explain.form_submit_button(
                "Predict + Explain",
                use_container_width=True,
            )

        if submit or submit_explain:
            st.session_state.last_payload = payload

            start = time.perf_counter()
            raw_pred = api_post("/predict", payload)
            client_latency = (time.perf_counter() - start) * 1000

            if raw_pred is None:
                st.error("API `/predict` lỗi hoặc không phản hồi. Kiểm tra log FastAPI.")
                if st.session_state.last_error:
                    st.code(st.session_state.last_error)
            else:
                pred = normalize_predict_response(raw_pred)
                pred["client_latency_ms"] = round(client_latency, 2)

                st.session_state.last_prediction = pred

                if submit_explain:
                    with st.spinner("Đang gọi SHAP explain. KernelExplainer có thể chạy hơi lâu..."):
                        explain_data = api_post("/explain", payload)
                    st.session_state.last_explanation = explain_data
                else:
                    st.session_state.last_explanation = None

                st.session_state.history.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": payload["type"],
                        "amount": payload["amount"],
                        "probability_pct": round(pred["fraud_probability"] * 100, 3),
                        "prediction": "Fraud" if pred["is_fraud"] else "Not Fraud",
                        "risk_level": pred["risk_level"],
                        "latency_ms": round(client_latency, 2),
                    }
                )

    with right:
        st.subheader("Kết quả dự đoán")

        pred = st.session_state.last_prediction

        if not pred:
            st.info("Nhập giao dịch ở bên trái rồi nhấn Predict.")
        else:
            st.plotly_chart(
                make_gauge(pred["fraud_probability"], pred["threshold"]),
                use_container_width=True,
            )

            if pred["is_fraud"]:
                st.markdown(
                    f"""
                    <div class='risk-fraud'>
                    <b>🚨 FRAUD</b><br>
                    Risk level: {pred['risk_level']} · 
                    Probability: {pred['fraud_probability'] * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif pred["fraud_probability"] >= pred["threshold"] * 0.4:
                st.markdown(
                    f"""
                    <div class='risk-mid'>
                    <b>⚠️ Cần theo dõi</b><br>
                    Risk level: {pred['risk_level']} · 
                    Probability: {pred['fraud_probability'] * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class='risk-safe'>
                    <b>✅ NOT FRAUD</b><br>
                    Risk level: {pred['risk_level']} · 
                    Probability: {pred['fraud_probability'] * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            k1, k2, k3 = st.columns(3)

            k1.metric("Threshold", f"{pred['threshold']:.2f}")
            k2.metric("Client latency", f"{pred.get('client_latency_ms', 0):.2f} ms")
            k3.metric("Prediction", pred["result"])

            with st.expander("Xem JSON response từ API"):
                st.json(pred["raw"])

        explanation = st.session_state.last_explanation

        if explanation:
            st.divider()
            st.subheader("Giải thích SHAP cho giao dịch")

            top_items = explanation.get("top_explanation", [])

            fig = make_shap_chart(top_items)

            if fig:
                st.plotly_chart(fig, use_container_width=True)

            if top_items:
                st.dataframe(pd.DataFrame(top_items), use_container_width=True)

            with st.expander("Xem JSON explain đầy đủ"):
                st.json(explanation)


with tab_batch:
    st.subheader("Test nhanh 3 mẫu giao dịch")
    st.caption(
        "Phần này giúp demo nhanh: một mẫu bình thường, một mẫu nghi gian lận, một mẫu giá trị lớn."
    )

    if st.button("Chạy test nhanh", use_container_width=True):
        rows = []
        progress = st.progress(0)

        for i, (name, payload) in enumerate(PRESETS.items(), start=1):
            start = time.perf_counter()
            raw = api_post("/predict", payload)
            latency = (time.perf_counter() - start) * 1000

            if raw:
                pred = normalize_predict_response(raw)

                rows.append(
                    {
                        "case": name,
                        "type": payload["type"],
                        "amount": payload["amount"],
                        "probability_pct": round(pred["fraud_probability"] * 100, 3),
                        "prediction": "Fraud" if pred["is_fraud"] else "Not Fraud",
                        "risk_level": pred["risk_level"],
                        "latency_ms": round(latency, 2),
                    }
                )

            progress.progress(i / len(PRESETS))

        if rows:
            batch_df = pd.DataFrame(rows)

            st.dataframe(batch_df, use_container_width=True)

            fig = px.bar(
                batch_df,
                x="case",
                y="probability_pct",
                color="prediction",
                title="So sánh xác suất gian lận giữa các mẫu",
                text="probability_pct",
            )

            fig.update_layout(
                height=420,
                yaxis_title="Fraud probability (%)",
                xaxis_title="",
            )

            st.plotly_chart(fig, use_container_width=True)


with tab_history:
    st.subheader("Lịch sử dự đoán trong phiên dashboard")

    if not st.session_state.history:
        st.info("Chưa có giao dịch nào được dự đoán trong phiên này.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        st.dataframe(hist_df, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            fig = px.line(
                hist_df.reset_index(),
                x="index",
                y="probability_pct",
                markers=True,
                title="Fraud probability theo thời gian test",
            )

            fig.update_layout(
                height=360,
                xaxis_title="Lần test",
                yaxis_title="Probability (%)",
            )

            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                hist_df,
                x="prediction",
                color="prediction",
                title="Số lượng kết quả Fraud / Not Fraud",
            )

            fig.update_layout(
                height=360,
                xaxis_title="Kết quả",
                yaxis_title="Số lượng",
            )

            st.plotly_chart(fig, use_container_width=True)


st.divider()
st.caption(
    "Fraud Detection DL Dashboard · FastAPI + Streamlit + TensorFlow + SHAP · PaySim Dataset"
)