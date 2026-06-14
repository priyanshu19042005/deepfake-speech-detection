"""Evaluation metrics, incl. Equal Error Rate (EER)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_curve,
)


def equal_error_rate(y_true, scores):
    """
    EER = point where False Acceptance Rate == False Rejection Rate.
    `scores` = predicted probability of the POSITIVE class (1 = deepfake).
    Returns (eer, threshold).
    """
    fpr, tpr, thr = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thr[idx])


def per_class_accuracy(cm):
    """Diagonal / row-sum for each class -> recall per class."""
    cm = np.asarray(cm, dtype=float)
    return (cm.diagonal() / cm.sum(axis=1).clip(min=1e-9)).tolist()


def full_report(y_true, y_pred, scores):
    """Bundle all required metrics into a dict."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    eer, eer_thr = equal_error_rate(y_true, scores)
    pca = per_class_accuracy(cm)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "eer": eer,
        "eer_threshold": eer_thr,
        "per_class_acc": {"genuine": pca[0], "deepfake": pca[1]},
        "confusion_matrix": cm.tolist(),
    }


def print_report(rep, title="Report"):
    print(f"\n===== {title} =====")
    print(f"  Accuracy        : {rep['accuracy']*100:.2f}%   (threshold >= 80%)")
    print(f"  F1 score        : {rep['f1']*100:.2f}%   (threshold >= 80%)")
    print(f"  EER             : {rep['eer']*100:.2f}%   (threshold <= 12%)")
    print(f"  Per-class acc   : genuine={rep['per_class_acc']['genuine']*100:.2f}%  "
          f"deepfake={rep['per_class_acc']['deepfake']*100:.2f}%   (each >= 75%)")
    cm = rep["confusion_matrix"]
    print( "  Confusion matrix       pred_genuine  pred_deepfake")
    print(f"        actual_genuine  {cm[0][0]:>12}  {cm[0][1]:>13}")
    print(f"        actual_deepfake {cm[1][0]:>12}  {cm[1][1]:>13}")
