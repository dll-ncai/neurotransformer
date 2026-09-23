"""Building blocks shared by Neurotransformer and its ablation variants."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveLayer(nn.Module):
    """Causal dilated convolution with a WaveNet-style gated activation and a residual connection.

    Input and output shape: ``(B, C, L)``.
    """

    def __init__(self, in_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.filter = nn.Conv1d(in_channels, in_channels, 1)
        self.gate = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)

        nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        nn.init.xavier_uniform_(self.filter.weight, gain=1.0)
        nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
        nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)

    def forward(self, x):
        output = self.conv(x)
        z = self.tanh(self.filter(output)) * self.sig(self.gate(output))
        # Drop the trailing padded samples so the convolution stays causal.
        z = z[:, :, :-self.padding]
        z = self.conv2(z)
        return x + z


class WaveBlock(nn.Module):
    """1x1 channel projection followed by a stack of WaveLayers with dilations 1, 2, 4, ..., 2**(n-1)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super().__init__()
        self.layers = nn.ModuleList()
        dilations = [2 ** i for i in range(dilation_rates)]
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
        for dilation in dilations:
            self.layers.append(WaveLayer(out_channels, kernel_size, dilation))
        nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)

    def forward(self, x):
        x = self.conv1d(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MFFMBlock(nn.Module):
    """Multi-scale feature fusion module: two densely connected Conv-BN-ReLU stages.

    Maps ``(B, C, L)`` to ``(B, C + 24, L)``.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, input):
        x1 = F.relu(self.bn1(self.conv1(input)))
        x1 = torch.cat((x1, input), dim=1)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        return torch.cat((x2, x1), dim=1)
