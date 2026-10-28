"""SCNet (Wu et al., 2023) for single-channel EEG windows of shape ``(batch, 1, L)``."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFFMBlock(nn.Module):
    """Two Conv1d (k = 5) - BN - ReLU layers (8 and 16 filters); each output is concatenated with its input."""

    def __init__(self, in_channels):
        super(MFFMBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1_cat = torch.cat((x1, x), dim=1)
        x2 = F.relu(self.bn2(self.conv2(x1_cat)))
        return torch.cat((x2, x1_cat), dim=1)


class SILM(nn.Module):
    """Appends the channel-wise mean, standard deviation and maximum to the input."""

    def __init__(self):
        super(SILM, self).__init__()

    def forward(self, x):
        gap = torch.mean(x, dim=1, keepdim=True)
        gsp = torch.std(x, dim=1, keepdim=True, unbiased=False)
        gmp, _ = torch.max(x, dim=1, keepdim=True)
        gap = F.dropout(gap, 0.05, training=self.training)
        gsp = F.dropout(gsp, 0.05, training=self.training)
        gmp = F.dropout(gmp, 0.05, training=self.training)
        return torch.cat((x, gap, gsp, gmp), dim=1)


class SCNet(nn.Module):
    def __init__(self, n_classes=3):
        super(SCNet, self).__init__()
        self.silm = SILM()
        self.bn1 = nn.BatchNorm1d(8)
        self.mffm_block1 = MFFMBlock(8)
        self.mffm_block2 = MFFMBlock(8)

        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.mffm_block3 = MFFMBlock(32)
        self.mffm_block4 = MFFMBlock(32)
        # conv2 takes the 56 channels of the summed MFFMBlock3/4 outputs
        self.conv2 = nn.Conv1d(in_channels=56, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.mffm_block5 = MFFMBlock(32)
        self.conv3 = nn.Conv1d(in_channels=56, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, n_classes)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.silm(x)                                 # [B,4,L]
        x1 = F.avg_pool1d(x, 2, 2)                       # [B,4,L/2]
        x2 = F.max_pool1d(x, 2, 2)                       # [B,4,L/2]
        x = torch.cat((x1, x2), dim=1)                   # [B,8,L/2]
        x = self.bn1(x)

        y1 = self.mffm_block1(x)                         # [B,32,L/2]
        y2 = self.mffm_block2(x)                         # [B,32,L/2]
        x = y1 + y2                                      # [B,32,L/2]
        x = F.dropout1d(x, 0.5, training=self.training)  # channel-wise, as dropout2d did on 3-D input

        x = F.max_pool1d(x, 2, 2)                        # [B,32,L/4]
        x = F.relu(self.bn2(self.conv1(x)))              # [B,32,L/4]

        y1 = self.mffm_block3(x)                         # [B,56,L/4]
        y2 = self.mffm_block4(x)                         # [B,56,L/4]
        x = y1 + y2                                      # [B,56,L/4]
        x = F.relu(self.bn3(self.conv2(x)))              # [B,32,L/4]

        x = self.mffm_block5(x)                          # [B,56,L/4]
        x = F.max_pool1d(x, 2, 2)                        # [B,56,L/8]

        x = self.conv3(x)                                # [B,32,L/8]
        x = x.mean(dim=2)                                # [B,32]
        return self.fc(x)
