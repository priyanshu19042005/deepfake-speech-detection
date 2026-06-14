"""
Standalone inference: classify a single audio file as Genuine or Deepfake.

Uses the exact same preprocessing/feature pipeline as training (audio_utils.py),
so behavior matches the trained model.

Usage:
    python predict.py path/to/audio.wav
    python predict.py path/to/audio.wav --model models/deepfake_cnn.pt --json
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import argparse
import numpy as np
import torch

from audio_utils import extract_features, LABELS
from model import build_model


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # Calibrated operating point (P(deepfake) >= threshold -> deepfake).
    threshold = float(ckpt.get("decision_threshold", 0.5))
    return model, threshold


@torch.no_grad()
def predict_file(path, model, device, threshold=0.5):
    feat = extract_features(path)                       # (H, W)
    x = torch.from_numpy(feat[None, None]).to(device)   # (1,1,H,W)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred = int(probs[1] >= threshold)                   # apply calibrated threshold
    return {
        "file": path,
        "prediction": LABELS[pred],
        "label_index": pred,
        "confidence": float(probs[pred]),
        "prob_genuine": float(probs[0]),
        "prob_deepfake": float(probs[1]),
    }


def main():
    ap = argparse.ArgumentParser(description="Classify audio as Genuine or Deepfake.")
    ap.add_argument("audio", help="path to an audio file (.wav, .mp3, .flac, ...)")
    ap.add_argument("--model", default="models/deepfake_cnn.pt")
    ap.add_argument("--json", action="store_true", help="output raw JSON")
    args = ap.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, threshold = load_model(args.model, device)
    result = predict_file(args.audio, model, device, threshold)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n  File       : {os.path.basename(result['file'])}")
        print(f"  Prediction : {result['prediction']}")
        print(f"  Confidence : {result['confidence']*100:.2f}%")
        print(f"  (genuine={result['prob_genuine']*100:.2f}%  "
              f"deepfake={result['prob_deepfake']*100:.2f}%)\n")


if __name__ == "__main__":
    main()
