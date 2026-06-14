"""
Evaluate the trained model on the held-out TEST split.

Produces the full metrics report (accuracy, F1, EER, per-class accuracy),
saves a confusion-matrix PNG and an ROC-curve PNG to models/, and writes a
JSON metrics file used by the README.

Usage:
    python evaluate.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from model import build_model
from metrics import full_report, print_report, equal_error_rate


def load_npz(path):
    d = np.load(path)
    return d["X"], d["y"]


@torch.no_grad()
def predict_all(model, X, device, batch=256):
    model.eval()
    scores = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch][:, None]).to(device)
        prob = torch.softmax(model(xb), dim=1)[:, 1]
        scores.append(prob.cpu().numpy())
    return np.concatenate(scores)


def plot_confusion(cm, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    classes = ["Genuine", "Deepfake"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_roc(y_true, scores, eer, path):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.scatter([eer], [1 - eer], color="red", zorder=5, label=f"EER={eer*100:.2f}%")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/deepfake_cnn.pt")
    ap.add_argument("--test", default="features/test.npz")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the calibrated decision threshold")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    thr = args.threshold if args.threshold is not None else ckpt.get("decision_threshold", 0.5)
    print(f"loaded {args.model} (trained {ckpt.get('epoch','?')} ep, "
          f"decision threshold={thr:.3f})")

    Xte, yte = load_npz(args.test)
    print(f"test set: {Xte.shape}")
    scores = predict_all(model, Xte, device)

    preds = (scores >= thr).astype(int)
    rep = full_report(yte, preds, scores)
    print_report(rep, title=f"TEST (threshold={thr:.3f})")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    plot_confusion(rep["confusion_matrix"], "models/confusion_matrix.png")
    plot_roc(yte, scores, rep["eer"], "models/roc_curve.png")
    with open("models/test_metrics.json", "w") as f:
        json.dump(rep, f, indent=2)
    print("\nsaved: models/confusion_matrix.png, models/roc_curve.png, models/test_metrics.json")

    # Threshold check summary
    ok = (rep["accuracy"] >= 0.80 and rep["f1"] >= 0.80 and rep["eer"] <= 0.12
          and rep["per_class_acc"]["genuine"] >= 0.75
          and rep["per_class_acc"]["deepfake"] >= 0.75)
    print("\nALL THRESHOLDS MET" if ok else "\nSOME THRESHOLDS NOT MET")


if __name__ == "__main__":
    main()
