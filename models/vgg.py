"""VGG11/13/16/19 for 32x32, 3-channel inputs (CIFAR10/CIFAR100/ImageNet32).

Ported from alexandertheus/Intra-Fusion (model_architectures/vgg.py) —
"Towards Meta-Pruning via Optimal Transport" (ICLR 2024) — which in turn took
the base architecture from https://github.com/kuangliu/pytorch-cifar. The
Intra-Fusion paper prunes exactly this VGG11-BN via Optimal Transport, so it
is a natural real-world target for this project's ot_compression/ stage.

The `_half` / `_doub` / `_quad` VGG11 variants scale every conv layer's width
by 0.5x / 2x / 4x — handy as fixed reference points when comparing against
this project's own OT-compressed widths.
"""

import torch
import torch.nn as nn

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG11_quad": [64, "M", 512, "M", 1024, 1024, "M", 2048, 2048, "M", 2048, 512, "M"],
    "VGG11_doub": [64, "M", 256, "M", 512, 512, "M", 1024, 1024, "M", 1024, 512, "M"],
    "VGG11_half": [64, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M",
              512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, batch_norm=True, bias=True, relu_inplace=True):
        super(VGG, self).__init__()
        self.batch_norm = batch_norm
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name], relu_inplace=relu_inplace)
        self.classifier = nn.Linear(512, num_classes, bias=self.bias)

    def forward(self, x, return_activations: bool = False):
        if not return_activations:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

        # Hook every ReLU so the reference architecture/forward pass stays
        # untouched, while still exposing per-layer post-activations in the
        # shape activation_weight_matrices/extract_activations.py expects.
        post_acts = []
        handles = []
        for m in self.features:
            if isinstance(m, nn.ReLU):
                handles.append(m.register_forward_hook(lambda mod, inp, out: post_acts.append(out)))
        try:
            out = self.features(x)
        finally:
            for h in handles:
                h.remove()
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits, post_acts

    def _make_layers(self, cfg, relu_inplace=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.batch_norm:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=relu_inplace),
                    ]
                else:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
                        nn.ReLU(inplace=relu_inplace),
                    ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG("VGG11")
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
