import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# =========================
# 1. PATH CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "saved_models", "ffnn.h5")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "saved_models", "feature_names.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "model", "saved_models", "ffnn_threshold.pkl")

SAVE_XAI_DIR = os.path.join(BASE_DIR, "xai", "results")
os.makedirs(SAVE_XAI_DIR, exist_ok=True)

# =========================
# 2. LOAD DATA
# =========================

test_df = pd.read_csv(DATA_PATH)

X_test = test_df.drop("isFraud", axis=1)
y_test = test_df["isFraud"]

feature_names = joblib.load(FEATURE_PATH)
threshold = joblib.load(THRESHOLD_PATH)

X_test = X_test[feature_names].astype("float32")

print("X_test shape:", X_test.shape)
print("Threshold FFNN:", threshold)

# =========================
# 3. LOAD FFNN MODEL
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("Loaded FFNN model successfully!")
model.summary()

# =========================
# 4. CHỌN SAMPLE CHO SHAP
# =========================

background_size = 100
explain_size = 300

X_background = X_test.sample(
    n=min(background_size, len(X_test)),
    random_state=42
)

y_prob_test = model.predict(X_test.values, verbose=0).flatten()

X_test_with_prob = X_test.copy()
X_test_with_prob["fraud_probability"] = y_prob_test
X_test_with_prob["isFraud"] = y_test.values

X_explain = X_test_with_prob.sort_values(
    by="fraud_probability",
    ascending=False
).head(explain_size)

X_explain = X_explain.drop(
    ["fraud_probability", "isFraud"],
    axis=1
)

# Lưu lại tập dữ liệu được dùng để giải thích SHAP
X_explain.to_csv(
    os.path.join(SAVE_XAI_DIR, "x_explain_top_fraud_probability.csv"),
    index=False)

X_background_np = X_background.values.astype("float32")
X_explain_np = X_explain.values.astype("float32")

# =========================
# 5. SHAP KERNEL EXPLAINER
# =========================

print("\nUsing SHAP KernelExplainer for FFNN...")

def predict_fn(x):
    x = np.array(x).astype("float32")
    return model.predict(x, verbose=0).flatten()

explainer = shap.KernelExplainer(
    predict_fn,
    X_background_np
)

shap_values = explainer.shap_values(
    X_explain_np,
    nsamples=100
)

shap_values = np.array(shap_values)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

print("SHAP completed!")
print("SHAP values shape:", shap_values.shape)

# =========================
# 6. SAVE SHAP VALUES
# =========================

shap_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)

shap_df.to_csv(
    os.path.join(SAVE_XAI_DIR, "shap_values_ffnn.csv"),
    index=False
)

# =========================
# 7. GLOBAL FEATURE IMPORTANCE
# =========================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_ABS_SHAP": mean_abs_shap
}).sort_values(
    by="Mean_ABS_SHAP",
    ascending=False
)

print("\n===== TOP 20 IMPORTANT FEATURES =====")
print(importance_df.head(20))

importance_df.to_csv(
    os.path.join(SAVE_XAI_DIR, "shap_feature_importance_ffnn.csv"),
    index=False
)

# =========================
# 8. SHAP BAR PLOT
# =========================

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.title("SHAP Global Feature Importance - FFNN")
plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_XAI_DIR, "shap_bar_ffnn.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# 9. SHAP SUMMARY PLOT
# =========================

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_names,
    show=False
)

plt.title("SHAP Summary Plot - FFNN")
plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_XAI_DIR, "shap_summary_ffnn.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# 10. LOCAL EXPLANATION
# =========================

sample_index = 0

sample_data = X_explain.iloc[[sample_index]]
sample_prob = model.predict(sample_data.values.astype("float32"), verbose=0)[0][0]
sample_pred = int(sample_prob >= threshold)

print("\n===== LOCAL EXPLANATION =====")
print("Sample index:", sample_index)
print("Predicted fraud probability:", sample_prob)
print("Threshold:", threshold)
print("Prediction:", "Fraud" if sample_pred == 1 else "Not Fraud")

local_shap = shap_values[sample_index]

local_df = pd.DataFrame({
    "Feature": feature_names,
    "Feature_Value": sample_data.values.flatten(),
    "SHAP_Value": local_shap,
    "ABS_SHAP": np.abs(local_shap)
}).sort_values(
    by="ABS_SHAP",
    ascending=False
)

print("\n===== TOP 15 FEATURES FOR THIS TRANSACTION =====")
print(local_df.head(15))

local_df.to_csv(
    os.path.join(SAVE_XAI_DIR, f"local_explanation_sample_{sample_index}.csv"),
    index=False
)

# =========================
# 11. WATERFALL PLOT
# =========================

try:
    expected_value = explainer.expected_value

    if isinstance(expected_value, list):
        expected_value = expected_value[0]

    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value.flatten()[0]

    explanation = shap.Explanation(
        values=local_shap,
        base_values=expected_value,
        data=sample_data.values.flatten(),
        feature_names=feature_names
    )

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False
    )

    plt.title("SHAP Waterfall Plot - FFNN Transaction")
    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_XAI_DIR, f"shap_waterfall_sample_{sample_index}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

except Exception as e:
    print("\nWaterfall plot error:")
    print(e)

print("\nXAI SHAP analysis for FFNN completed successfully!")
print("Results saved to:", SAVE_XAI_DIR)