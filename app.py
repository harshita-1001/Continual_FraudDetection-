import streamlit as st
import torch
from model.resnet_model import FraudResNet

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("🛡 Continual Learning Fraud Detection")

# -------------------------------
# Model Metadata (Static Metrics)
# -------------------------------
MODEL_INFO = {
    "v1": {
        "path": "saved_models/fraud_model_v1.pth",
        "accuracy": 0.9989,
        "recall": 0.61,
        "f1": 0.69
    },
    "v2": {
        "path": "saved_models/fraud_model_v2.pth",
        "accuracy": 0.9993,
        "recall": 0.82,
        "f1": 0.85
    }
}

# -------------------------------
# Sidebar: Model Selection & Metrics
# -------------------------------
st.sidebar.title("⚙️ Model Settings")

model_version = st.sidebar.radio(
    "Select Model Version",
    ["v1", "v2"]
)

st.sidebar.markdown("### 📊 Model Metrics")
st.sidebar.metric("Accuracy", MODEL_INFO[model_version]["accuracy"])
st.sidebar.metric("Recall", MODEL_INFO[model_version]["recall"])
st.sidebar.metric("F1-Score", MODEL_INFO[model_version]["f1"])

# -------------------------------
# Load Selected Model
# -------------------------------
model = FraudResNet(input_dim=30)
model.load_state_dict(
    torch.load(MODEL_INFO[model_version]["path"], map_location="cpu")
)
model.eval()

# -------------------------------
# Input Section
# -------------------------------
st.subheader("Enter Transaction Features")

inputs = []
for i in range(30):
    inputs.append(
        st.number_input(
            f"Feature V{i+1}",
            value=0.0,
            format="%.6f"
        )
    )

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Fraud"):
    x = torch.tensor(inputs).float().unsqueeze(0)

    with torch.no_grad():
        prob = model(x).item()

    st.metric("Fraud Probability", f"{prob:.4f}")

    if prob > 0.0:
        st.error("⚠️ FRAUD DETECTED")
    else:
        st.success("✅ Legitimate Transaction")
