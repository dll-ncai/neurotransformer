"""Neurotransformer: a hybrid convolution-transformer model for single-channel EEG window classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MFFMBlock, WaveBlock

D_MODEL = 32


class Neurotransformer(nn.Module):
    """Neurotransformer.

    Pipeline::

        x (B, 1, L) -> [MFFM block || dilated WaveBlock] (summed) -> spatial dropout -> max-pool
                    -> Conv-BN-ReLU -> Conv-BN -> MFFM block -> max-pool -> Conv-BN   (B, 32, L/4)
                    -> Transformer encoder (3 layers)
                    -> Transformer decoder (3 layers) queried by a learnable [CLS] token
                    -> linear classifier  (B, n_classes)

    The model was developed on 2 s windows of one EEG channel sampled at 200 Hz, i.e. ``(B, 1, 400)``,
    but any length divisible by 4 works.

    The ``use_*`` flags switch off individual components and are used by the ablation variants in
    :mod:`neurotransformer.models.ablations`. With every flag at its default this is the full model.

    Args:
        n_classes: number of output classes.
        use_mffm: keep the MFFM (residual multi-scale convolution) blocks.
        use_dilated: keep the dilated gated WaveBlock branch.
        use_encoder: keep the Transformer encoder.
        use_decoder: keep the [CLS]-token Transformer decoder; if False, the encoder output is mean-pooled.
    """

    def __init__(self, n_classes=3, use_mffm=True, use_dilated=True, use_encoder=True, use_decoder=True):
        super().__init__()
        if not (use_mffm or use_dilated):
            raise ValueError("At least one of use_mffm / use_dilated must be True.")
        self.use_mffm = use_mffm
        self.use_dilated = use_dilated
        self.use_encoder = use_encoder
        self.use_decoder = use_decoder

        # Modules are created in the same order as the original experiment code so that a given
        # seed yields identical initial weights. mffm_block2 / wave_block2 are not used in
        # forward(); they are kept so released checkpoints load with strict=True and the parameter
        # count matches the reported 0.975 M.
        if use_mffm:
            self.mffm_block1 = MFFMBlock(1)
        if use_dilated:
            self.wave_block1 = WaveBlock(1, 25, 3, 5)
        if use_mffm:
            self.mffm_block2 = MFFMBlock(32)
        if use_dilated:
            self.wave_block2 = WaveBlock(32, 56, 3, 5)
        if use_mffm:
            self.mffm_block3 = MFFMBlock(32)
        self.conv1 = nn.Conv1d(in_channels=25, out_channels=D_MODEL, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=D_MODEL, out_channels=D_MODEL, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(D_MODEL)
        self.conv3 = nn.Conv1d(in_channels=56 if use_mffm else D_MODEL, out_channels=D_MODEL, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(D_MODEL)
        self.bn3 = nn.BatchNorm1d(D_MODEL)
        if use_encoder:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=8, dropout=0, batch_first=True),
                num_layers=3,
            )
        if use_decoder:
            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=D_MODEL, nhead=8, dropout=0, batch_first=True),
                num_layers=3,
            )
            # Learnable classification token (one query per sample).
            self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL))
        self.fc = nn.Linear(D_MODEL, n_classes)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        if self.use_mffm and self.use_dilated:
            x = self.mffm_block1(x) + self.wave_block1(x)
        elif self.use_mffm:
            x = self.mffm_block1(x)
        else:
            x = self.wave_block1(x)

        # Spatial (channel-wise) dropout.
        x = F.dropout1d(x, 0.5, training=self.training)

        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.use_mffm:
            x = self.mffm_block3(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.bn3(self.conv3(x))

        # (B, C, L) -> (B, L, C) for the transformer.
        x = x.permute(0, 2, 1)
        if self.use_encoder:
            x = self.encoder(x)

        if self.use_decoder:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, C)
            x = self.transformer_decoder(tgt=cls_tokens, memory=x)
            x = x.squeeze(1)
        else:
            x = x.mean(dim=1)

        return self.fc(x)
