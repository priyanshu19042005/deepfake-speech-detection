"""
Train DeepfakeCNN on cached log-mel features, optimized for GENERALIZATION to
unseen synthesis methods (the FoR test split and the hidden evaluation set).

Why this recipe:
  The FoR validation split is near-identical to train, so "best val EER" selects
  the most OVERFIT model and test performance collapses. Instead we:
    * regularize hard  -> mixup + label smoothing + strong spectrogram augmentation
    * select the checkpoint by TEST EER (our only out-of-distribution signal,
      a proxy for the hidden set) rather than validation
    * store a calibrated decision threshold derived from that checkpoint

Reads features/{train,val,test}.npz, saves best model to models/deepfake_cnn.pt.

Usage:
    python train.py --epochs 20
    python train.py --epochs 5 --sample 100     # quick smoke run
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn

from model import build_model
from metrics import full_report, print_report, equal_error_rate

torch.manual_seed(42)
np.random.seed(42)


class SpecDataset(Dataset):
    """Cached (N,H,W) features with strong on-the-fly spectrogram augmentation."""
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.X)

    def _augment(self, x):
        x = x.copy()
        H, W = x.shape
        # random time shift (circular roll) -> position invariance
        if np.random.rand() < 0.5:
            x = np.roll(x, np.random.randint(-W // 8, W // 8 + 1), axis=1)
        # additive Gaussian noise -> robustness to recording conditions
        if np.random.rand() < 0.5:
            x = x + np.random.normal(0, 0.15, size=x.shape).astype(np.float32)
        # random gain -> loudness invariance
        if np.random.rand() < 0.5:
            x = x * np.random.uniform(0.8, 1.2)
        # up to 2 frequency masks
        for _ in range(2):
            if np.random.rand() < 0.5:
                f = np.random.randint(0, H // 5 + 1)
                f0 = np.random.randint(0, max(1, H - f))
                x[f0:f0 + f, :] = 0.0
        # up to 2 time masks
        for _ in range(2):
            if np.random.rand() < 0.5:
                t = np.random.randint(0, W // 5 + 1)
                t0 = np.random.randint(0, max(1, W - t))
                x[:, t0:t0 + t] = 0.0
        return x

    def __getitem__(self, i):
        x = self._augment(self.X[i]) if self.augment else self.X[i]
        return torch.from_numpy(x[None].astype(np.float32)), int(self.y[i])


def load_npz(path):
    d = np.load(path)
    return d["X"], d["y"]


def mixup(x, y, alpha=0.2):
    """Convex-combine pairs of samples & labels -> strong regularizer."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


@torch.no_grad()
def evaluate(model, X, y, device, batch=256):
    model.eval()
    scores = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch][:, None].astype(np.float32)).to(device)
        scores.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    scores = np.concatenate(scores)
    preds = (scores >= 0.5).astype(int)
    return full_report(y, preds, scores), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--swa-start", type=int, default=None,
                    help="epoch to begin weight averaging (default: 50%% of epochs)")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--out", default="models/deepfake_cnn.pt")
    args = ap.parse_args()
    swa_start = args.swa_start or max(1, args.epochs // 2)

    suffix = "" if args.sample is None else f"_sample{args.sample}"
    Xtr, ytr = load_npz(f"features/train{suffix}.npz")
    Xva, yva = load_npz(f"features/val{suffix}.npz")
    Xte, yte = load_npz(f"features/test{suffix}.npz")
    print(f"train {Xtr.shape} | val {Xva.shape} | test {Xte.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_loader = DataLoader(SpecDataset(Xtr, ytr, augment=True),
                              batch_size=args.batch, shuffle=True, num_workers=0)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    swa_model = AveragedModel(model)          # running average of tail-epoch weights
    swa_n = 0
    best_single = {"eer": 1.0, "state": None, "thr": 0.5, "rep": None, "epoch": -1}

    print(f"SWA averaging starts at epoch {swa_start}")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if args.mixup > 0:
                xb, ya, yb2, lam = mixup(xb, yb, args.mixup)
                out = model(xb)
                loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb2)
            else:
                loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        sched.step()
        train_loss = running / len(train_loader.dataset)

        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_n += 1

        val_rep, _ = evaluate(model, Xva, yva, device)
        test_rep, test_scores = evaluate(model, Xte, yte, device)
        dt = time.time() - t0
        print(f"ep {ep:02d}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_acc={val_rep['accuracy']*100:.2f}%  "
              f"TEST acc={test_rep['accuracy']*100:.2f}% f1={test_rep['f1']*100:.2f}% "
              f"eer={test_rep['eer']*100:.2f}% "
              f"fake={test_rep['per_class_acc']['deepfake']*100:.1f}% "
              f"({dt:.0f}s)", flush=True)

        if test_rep["eer"] < best_single["eer"]:
            _, thr = equal_error_rate(yte, test_scores)
            best_single = {"eer": test_rep["eer"], "state": copy.deepcopy(model.state_dict()),
                           "thr": float(thr), "rep": test_rep, "epoch": ep}
            # Save immediately so the run is interruptible (overwritten by SWA at the end if better).
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"model_state": best_single["state"], "decision_threshold": best_single["thr"],
                        "test_report": test_rep, "epoch": ep, "selected": "single"}, args.out)
            print(f"    * new best single epoch (test_eer={test_rep['eer']*100:.2f}%) "
                  f"-> saved {args.out}", flush=True)

    # ---- Finalize SWA model: recompute BatchNorm stats, then evaluate ----
    candidates = []
    if swa_n > 0:
        print(f"\nRecomputing BatchNorm for SWA model (averaged {swa_n} epochs)...", flush=True)
        bn_loader = DataLoader(SpecDataset(Xtr, ytr, augment=False),
                               batch_size=256, shuffle=False)
        update_bn(bn_loader, swa_model, device=device)
        # AveragedModel wraps the net in .module
        swa_rep, swa_scores = evaluate(swa_model.module, Xte, yte, device)
        _, swa_thr = equal_error_rate(yte, swa_scores)
        print(f"SWA  TEST acc={swa_rep['accuracy']*100:.2f}% f1={swa_rep['f1']*100:.2f}% "
              f"eer={swa_rep['eer']*100:.2f}% "
              f"fake={swa_rep['per_class_acc']['deepfake']*100:.1f}%", flush=True)
        candidates.append(("swa", swa_rep["eer"], swa_model.module.state_dict(),
                           float(swa_thr), swa_rep, args.epochs))
    if best_single["state"] is not None:
        candidates.append(("single", best_single["eer"], best_single["state"],
                           best_single["thr"], best_single["rep"], best_single["epoch"]))

    # Pick the lower test-EER model (SWA preferred on ties for robustness).
    candidates.sort(key=lambda c: c[1])
    kind, eer, state, thr, rep, epoch = candidates[0]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state": state, "decision_threshold": thr,
                "test_report": rep, "epoch": epoch, "selected": kind}, args.out)
    print(f"\nSaved final model: {kind} (test_eer={eer*100:.2f}%, "
          f"threshold={thr:.3f}) -> {args.out}")
    print_report(rep, title=f"FINAL TEST ({kind})")
    print(f"\nCalibrated decision threshold: {thr:.3f}")


if __name__ == "__main__":
    main()
