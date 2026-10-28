"""WaveNet+LSTM (Albaqami et al., 2023) for single-channel EEG windows of shape ``(batch, 1, 400)``.

Two branches run on the same window: a WaveNet stack (gated dilated convolutions, then an LSTM) and an
attention-LSTM branch. Their features are concatenated and classified by one linear layer.
"""

import torch
from torch import nn
from torch.nn import functional as F


class AlternateLayer(nn.Module):
    """Attention-LSTM branch: maps ``(B, C, T)`` to a 60-dimensional feature vector."""

    def __init__(self, input_shape):
        super(AlternateLayer, self).__init__()
        channels, seq_len = input_shape
        self.channels = channels
        self.seq_len = seq_len

        # First LSTM over the temporal dimension (feature dim = channels)
        self.timdistLSTM = nn.LSTM(input_size=channels, hidden_size=64, num_layers=1, batch_first=True)

        # Attention over the hidden dimension
        self.attFCN = nn.Linear(64, 64)

        # Second LSTM over the attended sequence
        self.seqLSTM = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)

        # Collapse temporal information into a 60-dim feature vector
        self.findense = nn.Linear(64, 60)
        self.fintanh = nn.Tanh()

        torch.nn.init.xavier_uniform_(self.attFCN.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.findense.weight, gain=1.0)

    def forward(self, x):
        x = x.flip(-1)                 # the original runs this branch on the time-reversed signal
        x = x.transpose(1, 2)          # (B, T, C), so the LSTM runs over time
        x, _ = self.timdistLSTM(x)     # (B, T, 64)

        att = F.softmax(self.attFCN(x), dim=-1)
        x = x * att                    # attention across the feature dimension

        x, _ = self.seqLSTM(x)         # (B, T, 64)
        x = F.dropout(x, 0.2, training=self.training)
        x = x.mean(dim=1)              # global average pooling over time: (B, 64)
        return self.fintanh(self.findense(x))  # (B, 60)


class WaveLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bn=False):
        super(WaveLayer, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.filter = nn.Conv1d(in_channels, in_channels, 1)
        self.gate = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)
        self.bn = bn

        if self.bn:
            self.bn1 = nn.BatchNorm1d(in_channels)
            self.bnf = nn.BatchNorm1d(in_channels)
            self.bng = nn.BatchNorm1d(in_channels)

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.filter.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.gate.weight, gain=1.0)

    def forward(self, x):
        output = F.relu(self.conv(x))
        if self.bn:
            output = self.bn1(output)

        filt = self.filter(output)
        if self.bn:
            filt = self.bnf(filt)

        gate = self.gate(output)
        if self.bn:
            gate = self.bng(gate)

        z = self.tanh(filt) * self.sig(gate)

        # crop the extra (causal) padding to restore the input length
        z = z[:, :, :-self.padding]
        z = self.conv2(z)
        return x + z


class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates, bn=False):
        super(WaveBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
        for dilation in [2 ** i for i in range(dilation_rates)]:
            self.layers.append(WaveLayer(out_channels, kernel_size, dilation, bn))

        torch.nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)

    def forward(self, x):
        x = F.relu(self.conv1d(x))
        for layer in self.layers:
            x = layer(x)
        return x


class WaveNet(nn.Module):
    """WaveNet branch: four WaveBlocks, each followed by average pooling by 2 (400 -> 25 steps), then an LSTM."""

    def __init__(self, input_shape):
        super(WaveNet, self).__init__()
        channels, seq_len = input_shape
        assert seq_len % 16 == 0, "Sequence length must be divisible by 16 for the current pooling setup."

        self.block1 = WaveBlock(channels, 16, 3, 8)
        self.block2 = WaveBlock(16, 32, 3, 5)
        self.block3 = WaveBlock(32, 64, 3, 3)
        self.block4 = WaveBlock(64, 64, 2, 2)

        self.lstmblock = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)

        # Unused in forward; kept so trained checkpoints load (it classifies when the branch runs on its own).
        self.dense_layer = nn.Linear(64, 2)
        torch.nn.init.xavier_uniform_(self.dense_layer.weight, gain=1.0)

    def forward(self, x):
        x = F.avg_pool1d(self.block1(x), 2)  # T: 400 -> 200
        x = F.avg_pool1d(self.block2(x), 2)  # T: 200 -> 100
        x = F.avg_pool1d(self.block3(x), 2)  # T: 100 -> 50
        x = F.avg_pool1d(self.block4(x), 2)  # T: 50 -> 25

        x = x.transpose(1, 2)                # (B, 25, 64)
        _, (h_n, _) = self.lstmblock(x)
        x = h_n[-1]                          # last hidden state: (B, 64)
        return F.dropout(x, 0.5, training=self.training)


class WaveNetEnd(nn.Module):
    def __init__(self, input_size, n_classes):
        super(WaveNetEnd, self).__init__()
        self.dense_layer = nn.Linear(input_size, n_classes)
        torch.nn.init.xavier_uniform_(self.dense_layer.weight, gain=1.0)

    def forward(self, x):
        return self.dense_layer(x)


class WaveNetLSTM(nn.Module):
    """Full model: WaveNet branch (64 features) and attention-LSTM branch (60 features), then a linear layer."""

    def __init__(self, n_classes=3, input_shape=(1, 400)):
        super(WaveNetLSTM, self).__init__()
        self.wavenet = WaveNet(input_shape)
        self.alternate = AlternateLayer(input_shape)
        self.wavenetend = WaveNetEnd(64 + 60, n_classes)

    def forward(self, x):
        y = self.wavenet(x)                  # (B, 64)
        z = self.alternate(x)                # (B, 60)
        return self.wavenetend(torch.cat((y, z), dim=-1))
