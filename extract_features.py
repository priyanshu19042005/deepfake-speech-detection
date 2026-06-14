"""
Extract log-mel features for every file in the FoR 'norm' splits and cache them
to features/{train,val,test}.npz. Runs once; training then reads the cache.

Usage:
    python extract_features.py              # full extraction, all splits
    python extract_features.py --limit 100  # quick timing test (100/class/split)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import glob
import time
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from audio_utils import extract_features, N_MELS, N_FRAMES

DATA_ROOT = "data/for-norm/for-norm"
SPLITS = {"train": "training", "val": "validation", "test": "testing"}
CLASSES = {"real": 0, "fake": 1}  # 0 = genuine, 1 = deepfake


def list_files(split_dir, limit=None):
    """Return (filepath, label) pairs for a split."""
    items = []
    for cls, label in CLASSES.items():
        files = sorted(glob.glob(os.path.join(DATA_ROOT, split_dir, cls, "*.wav")))
        if limit:
            files = files[:limit]
        items.extend((f, label) for f in files)
    return items


def _worker(path):
    try:
        return extract_features(path)
    except Exception as e:
        print(f"  WARN failed {os.path.basename(path)}: {e}", flush=True)
        return None


def process_split(name, split_dir, limit=None, workers=None):
    items = list_files(split_dir, limit)
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    print(f"[{name}] {len(paths)} files -> extracting with {workers or 'auto'} workers...", flush=True)

    t0 = time.time()
    feats, keep_labels = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (feat, label) in enumerate(zip(ex.map(_worker, paths, chunksize=16), labels)):
            if feat is not None:
                feats.append(feat)
                keep_labels.append(label)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"    {i+1}/{len(paths)}  ({rate:.0f} files/s)", flush=True)

    X = np.stack(feats).astype(np.float32)        # (N, H, W)
    y = np.array(keep_labels, dtype=np.int64)
    dt = time.time() - t0
    print(f"[{name}] done: X={X.shape} y={y.shape} in {dt:.1f}s "
          f"({len(paths)/dt:.0f} files/s)", flush=True)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="files per class per split")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    args = ap.parse_args()

    os.makedirs("features", exist_ok=True)
    grand0 = time.time()
    for name in args.splits:
        X, y = process_split(name, SPLITS[name], limit=args.limit, workers=args.workers)
        suffix = "" if args.limit is None else f"_sample{args.limit}"
        out = f"features/{name}{suffix}.npz"
        np.savez_compressed(out, X=X, y=y)
        size_mb = os.path.getsize(out) / 1e6
        print(f"    saved {out} ({size_mb:.1f} MB)\n", flush=True)
    print(f"ALL DONE in {time.time()-grand0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
