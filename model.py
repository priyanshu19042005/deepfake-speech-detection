"""
Lightweight CNN for log-mel spectrogram classification (genuine vs deepfake).

Kept small on purpose: training is CPU-only, and a compact model both trains
fast and generalizes better to the hidden set than an over-parameterized one.
Input:  (B, 1, N_MELS=64, N_FRAMES=188)
Output: (B, 2) logits  ->  0 = genuine, 1 = deepfake
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, p_drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p_drop),
        )

    def forward(self, x):
        return self.block(x)


class DeepfakeCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16, 0.05),
            ConvBlock(16, 32, 0.10),
            ConvBlock(32, 64, 0.15),
            ConvBlock(64, 128, 0.20),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model():
    return DeepfakeCNN(n_classes=2)


if __name__ == "__main__":
    m = build_model()
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.randn(4, 1, 64, 188)
    print(f"params: {n_params:,}")
    print(f"forward: {tuple(x.shape)} -> {tuple(m(x).shape)}")
