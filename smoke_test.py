"""Quick end-to-end sanity check: load a few files, extract features, run torch."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import glob
import numpy as np
import torch
import torch.nn as nn

from audio_utils import extract_features, N_MELS, N_FRAMES

DATA = "data/for-norm/for-norm/training"

print("=== Feature extraction smoke test ===")
samples = []
for cls, label in [("real", 0), ("fake", 1)]:
    files = sorted(glob.glob(os.path.join(DATA, cls, "*.wav")))[:5]
    for f in files:
        feat = extract_features(f)
        samples.append((feat, label))
        print(f"  {cls:4s} {os.path.basename(f)[:30]:30s} -> {feat.shape}  "
              f"mean={feat.mean():+.3f} std={feat.std():.3f}")

X = np.stack([s[0] for s in samples])[:, None, :, :]  # (N,1,H,W)
y = np.array([s[1] for s in samples])
print(f"\nBatch tensor: X={X.shape}, y={y.tolist()}")

print("\n=== Torch sanity (tiny CNN forward pass) ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(8, 2),
).to(device)
out = model(torch.tensor(X, device=device))
print("forward output shape:", tuple(out.shape), "(expected (10, 2))")
print("\nSMOKE TEST PASSED ✅" if out.shape == (10, 2) else "FAILED")
