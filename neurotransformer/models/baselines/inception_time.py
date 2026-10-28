"""InceptionTime (Ismail Fawaz et al., 2020): one network of the ensemble, for inputs ``(batch, C, L)``."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionTime(nn.Module):
    """Six Inception modules with a residual shortcut after every third, global average pooling and a linear layer.

    ``forward`` returns log-probabilities; ``nn.CrossEntropyLoss`` and softmax give the same result on them as
    on logits.
    """

    def __init__(
        self,
        n_classes=3,
        in_channels=1,
        nb_filters=32,
        use_residual=True,
        use_bottleneck=True,
        depth=6,
        kernel_size=41,
        bottleneck_size=32,
    ):
        super(InceptionTime, self).__init__()
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        # as in the Keras original: kernel sizes 40, 20, 10, here rounded up to odd (41, 21, 11)
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = bottleneck_size

        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        # track channels for blocks and residuals
        current_channels = in_channels
        group_in_channels = in_channels

        for d in range(depth):
            block = _InceptionModule(
                in_channels=current_channels,
                nb_filters=nb_filters,
                bottleneck_size=bottleneck_size,
                kernel_size=self.kernel_size,
                use_bottleneck=use_bottleneck,
            )
            self.blocks.append(block)

            # block output channels (branches = len(ks) + 1)
            out_ch = len(block.ks) * nb_filters + nb_filters

            # add a shortcut at the end of each residual group of three blocks
            if use_residual and d % 3 == 2:
                self.shortcuts.append(_ShortcutLayer(in_channels=group_in_channels, out_channels=out_ch))
                group_in_channels = out_ch
            else:
                self.shortcuts.append(None)

            current_channels = out_ch

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(current_channels, n_classes)

    def forward(self, x):
        residual = x
        for idx, block in enumerate(self.blocks):
            out = block(x)
            if self.use_residual and idx % 3 == 2:
                shortcut = self.shortcuts[idx]
                if shortcut is not None:
                    out = shortcut(residual, out)
                residual = out
            x = out
        x = self.global_pool(x).squeeze(-1)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=1)


class _InceptionModule(nn.Module):
    def __init__(self, in_channels, nb_filters, bottleneck_size, kernel_size, use_bottleneck):
        super(_InceptionModule, self).__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
            conv_in = bottleneck_size
        else:
            conv_in = in_channels

        # odd kernel sizes, so "same" padding is symmetric
        ks = []
        for i in range(3):
            k = kernel_size // (2 ** i)
            if k % 2 == 0:
                k += 1
            ks.append(k)
        self.ks = ks

        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, nb_filters, k, padding=(k - 1) // 2, bias=False)
            for k in ks
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        self.batchnorm = nn.BatchNorm1d(nb_filters * (len(ks) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x_in = self.bottleneck(x) if self.use_bottleneck else x
        conv_outs = [conv(x_in) for conv in self.convs]
        conv_outs.append(self.maxpool_conv(self.maxpool(x)))
        out = torch.cat(conv_outs, dim=1)
        out = self.batchnorm(out)
        return self.relu(out)


class _ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ShortcutLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, out_tensor):
        sc = self.bn(self.conv(x))
        return self.relu(sc + out_tensor)
