"""
Streamlit web app: upload an audio file -> Genuine / Deepfake + confidence.

Run locally:   streamlit run app.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import tempfile
import numpy as np
import torch
import streamlit as st

from audio_utils import extract_features, LABELS, SAMPLE_RATE
from model import build_model

MODEL_PATH = "models/deepfake_cnn.pt"

st.set_page_config(page_title="Deepfake Speech Detector", page_icon="🎙️", layout="centered")


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    threshold = float(ckpt.get("decision_threshold", 0.5))
    return model, device, threshold


@torch.no_grad()
def classify(path, model, device, threshold):
    feat = extract_features(path)
    x = torch.from_numpy(feat[None, None]).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred = int(probs[1] >= threshold)
    return pred, probs


st.title("🎙️ Deepfake Speech Detector")
st.caption("Upload a speech recording to check whether it is **Genuine (Human)** "
           "or a **Deepfake (AI-Generated)** voice.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Train the model first "
             f"(`python train.py`) or add the weights to the repo.")
    st.stop()

model, device, threshold = load_model()

uploaded = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="The audio is resampled to 16 kHz mono and trimmed/padded to 3 seconds.",
)

if uploaded is not None:
    st.audio(uploaded)
    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    try:
        with st.spinner("Analyzing audio..."):
            pred, probs = classify(tmp_path, model, device, threshold)
    finally:
        os.unlink(tmp_path)

    label = LABELS[pred]
    confidence = float(probs[pred]) * 100

    st.divider()
    if pred == 0:
        st.success(f"### ✅ {label}")
    else:
        st.error(f"### 🚨 {label}")

    st.metric("Confidence", f"{confidence:.2f}%")
    st.progress(min(int(confidence), 100))

    col1, col2 = st.columns(2)
    col1.metric("Genuine (Human)", f"{probs[0]*100:.2f}%")
    col2.metric("Deepfake (AI)", f"{probs[1]*100:.2f}%")

    with st.expander("How it works"):
        st.write(
            "The audio is converted to a **64-band log-mel spectrogram** "
            f"(16 kHz, 3 s) and passed through a convolutional neural network "
            "trained on the **Fake-or-Real** dataset. The network outputs a "
            "probability for each class; the higher one is shown as the prediction."
        )
else:
    st.info("👆 Upload an audio file to get a prediction.")
