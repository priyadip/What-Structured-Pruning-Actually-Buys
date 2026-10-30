"""Configurable MLP (2-3 hidden layers) used for MNIST/FashionMNIST and as the
target of neuron-level OT compression.

Named width configurations (`cfg` below) follow the same pattern as
models/vgg.py's `cfg` dict: a base architecture plus `_half` / `_doub` /
`_quad` variants that scale every hidden layer's width by 0.5x / 2x / 4x, so
MLP layers have the same fixed reference points for OT-compressed-width
comparisons that the VGG family does.
"""

import torch.nn as nn

cfg = {
    "MLP2": (256, 128),
    "MLP2_half": (128, 64),
    "MLP2_doub": (512, 256),
    "MLP2_quad": (1024, 512),
    "MLP2_oct": (2048, 1024),
    "MLP3": (256, 128, 64),
    "MLP3_half": (128, 64, 32),
    "MLP3_doub": (512, 256, 128),
    "MLP3_quad": (1024, 512, 256),
    "MLP3_oct": (2048, 1024, 512),
}


class MLP(nn.Module):
    """Feed-forward classifier with a configurable number of hidden layers.

    Each hidden layer's pre-activation (Z) and post-activation (A) tensors are
    exposed via `return_activations=True` so activation_weight_matrices/ can
    extract them without hooks.
    """

    def __init__(self, in_dim: int, hidden_dims=(256, 128), num_classes: int = 10,
                 activation: str = "relu", batch_norm: bool = False, dropout: float = 0.0):
        super().__init__()
        act_layer = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]

        self.in_dim = in_dim
        self.hidden_dims = list(hidden_dims)
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        dims = [in_dim] + self.hidden_dims
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(d_in, d_out))
            self.bns.append(nn.BatchNorm1d(d_out) if batch_norm else nn.Identity())
        self.act = act_layer()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x, return_activations: bool = False):
        x = x.view(x.size(0), -1)
        pre_acts, post_acts = [], []
        for layer, bn in zip(self.layers, self.bns):
            z = bn(layer(x))  # pre-activation: input to the nonlinearity, after optional BN
            a = self.act(z)
            pre_acts.append(z)
            post_acts.append(a)
            x = self.drop(a)
        logits = self.classifier(x)
        if return_activations:
            return logits, pre_acts, post_acts
        return logits


def build_mlp(name: str, in_dim: int, num_classes: int = 10, batch_norm: bool = False, **kwargs) -> MLP:
    """Construct an MLP from a named width configuration in `cfg`, mirroring
    models.vgg.VGG(vgg_name, ...)."""
    if name not in cfg:
        raise ValueError(f"Unknown MLP variant '{name}'. Options: {sorted(cfg)}")
    return MLP(in_dim=in_dim, hidden_dims=cfg[name], num_classes=num_classes, batch_norm=batch_norm, **kwargs)
