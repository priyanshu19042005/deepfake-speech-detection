"""
Audio preprocessing and mel-spectrogram features.

Training, predict.py and the app all import from here so they run the exact same
pipeline. If these ever differ, the model scores great in training and badly on
real audio, so I keep it in one place.
"""
import os

# Must be set before numpy/torch import OpenMP-linked libs (Anaconda quirk).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import librosa

# ---- Fixed audio / feature parameters ----
SAMPLE_RATE = 16000      # FoR "norm" is already 16 kHz
DURATION = 3.0           # seconds; pad/crop every clip to this length
N_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 64              # mel bands -> feature "height"
N_FFT = 1024
HOP_LENGTH = 256         # -> ~188 frames for 3 s; feature "width"
N_FRAMES = 1 + N_SAMPLES // HOP_LENGTH


def load_audio(path):
    """Load an audio file as mono 16 kHz float32 waveform."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def fix_length(y):
    """Pad with zeros or center-crop to exactly N_SAMPLES."""
    if len(y) > N_SAMPLES:
        start = (len(y) - N_SAMPLES) // 2
        y = y[start:start + N_SAMPLES]
    elif len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)), mode="constant")
    return y


def logmel_from_waveform(y):
    """Waveform -> log-mel spectrogram, shape (N_MELS, N_FRAMES), per-sample normalized."""
    y = fix_length(y)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    # Per-sample standardization -> robustness to loudness/recording differences.
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    # Guard against off-by-one frame counts.
    if logmel.shape[1] < N_FRAMES:
        logmel = np.pad(logmel, ((0, 0), (0, N_FRAMES - logmel.shape[1])), mode="constant")
    else:
        logmel = logmel[:, :N_FRAMES]
    return logmel.astype(np.float32)


def extract_features(path):
    """End-to-end: file path -> (N_MELS, N_FRAMES) log-mel feature."""
    return logmel_from_waveform(load_audio(path))


# Label convention used everywhere: 0 = real (genuine), 1 = fake (deepfake).
LABELS = {0: "Genuine (Human)", 1: "Deepfake (AI-Generated)"}
