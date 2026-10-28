"""LEEDNet (Alqarni et al., 2026), raw-EEG branch only.

The full LEEDNet runs parallel branches on the raw window, its power spectrum and its wavelet transform. This
model keeps only the raw-EEG branch, the version used for the paper's comparisons. Module names follow the
multi-branch original (``branches.0``, ``classifier``), so its single-branch checkpoints load directly.
"""

import torch.nn as nn


class ConvBranch1D(nn.Module):
    def __init__(self, in_channels, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(32, output_size)  # auxiliary per-branch head of the original; not used for training

    def forward(self, x):
        x = self.conv(x)                      # [B, 32, T/2]
        x = self.global_pool(x).squeeze(-1)   # [B, 32]
        return x, self.linear(x)


class LEEDNetRaw(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.branches = nn.ModuleList([ConvBranch1D(1, n_classes)])
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        features, _ = self.branches[0](x)
        return self.classifier(features)
