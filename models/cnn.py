"""Small CNN for CIFAR10/CIFAR100/ImageNet32-scale inputs."""

import torch.nn as nn


class SmallCNN(nn.Module):
    """VGG-style conv stack; channel counts are the compression target for
    ot_compression/compress_cnn.py (channel pruning via OT on filters)."""

    def __init__(self, in_channels: int = 3, channels=(32, 64, 128), num_classes: int = 10):
        super().__init__()
        self.channels = list(channels)
        blocks = []
        c_in = in_channels
        for c_out in self.channels:
            blocks.append(nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.channels[-1], num_classes)

    def forward(self, x, return_activations: bool = False):
        post_acts = []
        for block in self.blocks:
            x = block(x)
            post_acts.append(x)
        pooled = self.pool(x).flatten(1)
        logits = self.classifier(pooled)
        if return_activations:
            return logits, post_acts
        return logits
