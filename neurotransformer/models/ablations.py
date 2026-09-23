"""Ablation variants of Neurotransformer.

Each class removes one component of the full model. Layer names and shapes match the original
experiment code so that checkpoints trained with it load with ``strict=True``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MFFMBlock, WaveBlock
from .neurotransformer import D_MODEL, Neurotransformer


class NeurotransformerWORes(Neurotransformer):
    """Without the residual multi-scale convolutions (MFFM blocks); only the dilated branch remains."""

    def __init__(self, n_classes=3):
        super().__init__(n_classes, use_mffm=False)


class NeurotransformerWODil(Neurotransformer):
    """Without the dilated gated WaveBlock branch."""

    def __init__(self, n_classes=3):
        super().__init__(n_classes, use_dilated=False)


class NeurotransformerWOEncoder(Neurotransformer):
    """Without the Transformer encoder; the [CLS] decoder attends directly to the convolutional features."""

    def __init__(self, n_classes=3):
        super().__init__(n_classes, use_encoder=False)


class NeurotransformerWODecoder(Neurotransformer):
    """Without the [CLS]-token Transformer decoder; the encoder output is mean-pooled over time."""

    def __init__(self, n_classes=3):
        super().__init__(n_classes, use_decoder=False)


class NeurotransformerWOConv(nn.Module):
    """Without any convolutional front end: a 1x1 projection feeds the raw signal to the transformer."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.input_proj = nn.Conv1d(1, D_MODEL, kernel_size=1)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=8, dropout=0, batch_first=True),
            num_layers=3,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=D_MODEL, nhead=8, dropout=0, batch_first=True),
            num_layers=3,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL))
        self.fc = nn.Linear(D_MODEL, n_classes)

    def forward(self, x):
        x = self.input_proj(x).permute(0, 2, 1)  # (B, L, C)
        x = self.encoder(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = self.transformer_decoder(tgt=cls_tokens, memory=x).squeeze(1)
        return self.fc(x)


class NeurotransformerWOTrans(nn.Module):
    """Without the transformer: a purely convolutional network with global average pooling.

    This variant uses a two-channel input stem (average- and max-pooled copies of the signal) and
    applies the second MFFM/WaveBlock stage.
    """

    def __init__(self, n_classes=3):
        super().__init__()
        self.mffm_block1 = MFFMBlock(2)
        self.wave_block1 = WaveBlock(2, 26, 3, 5)
        self.mffm_block2 = MFFMBlock(32)
        self.wave_block2 = WaveBlock(32, 56, 3, 5)
        self.mffm_block3 = MFFMBlock(32)
        self.conv1 = nn.Conv1d(in_channels=26, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(2)
        self.conv2 = nn.Conv1d(in_channels=56, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=56, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, n_classes)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x1 = F.avg_pool1d(x, kernel_size=2, stride=2)
        x2 = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.bn1(torch.cat((x1, x2), dim=1))
        x = self.mffm_block1(x) + self.wave_block1(x)

        x = F.dropout1d(x, 0.5, training=self.training)

        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.mffm_block2(x) + self.wave_block2(x)
        x = self.bn3(self.conv2(x))
        x = self.mffm_block3(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.bn4(self.conv3(x))

        x = torch.mean(x, dim=2)
        return self.fc(x)
