"""Returns a model instance by name/architecture so trainer/ and
activation_weight_matrices/ never need to import concrete model classes."""

from models.cnn import SmallCNN
from models.lstm import LSTMClassifier
from models.mlp import MLP, build_mlp
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.vgg import VGG

_DEFAULT_INPUT_DIMS = {
    "MNIST": 28 * 28,
    "FashionMNIST": 28 * 28,
    "CIFAR10": 32 * 32 * 3,
    "CIFAR100": 32 * 32 * 3,
    "ImageNet32": 32 * 32 * 3,
}

_DEFAULT_NUM_CLASSES = {
    "MNIST": 10,
    "FashionMNIST": 10,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "ImageNet32": 1000,
}

# 3-channel-only datasets: VGG/ResNet below (ported from alexandertheus/Intra-Fusion)
# hardcode a 3-channel 32x32 stem, matching the CIFAR-only scope of that paper.
_THREE_CHANNEL_DATASETS = ("CIFAR10", "CIFAR100", "ImageNet32")

_CONV_MODEL_PREFIXES = ("vgg", "resnet")


def is_conv_model(model_name: str) -> bool:
    """True for the ported VGG/ResNet families (which need a 3x32x32 input)."""
    return model_name.startswith(_CONV_MODEL_PREFIXES)


def conv_input_spec(model_name: str, dataset_name: str):
    """Returns (force_rgb, image_size) the data loader should use so that
    `dataset_name` feeds `model_name` a valid input. For conv models on a
    1-channel dataset this is (True, 32); otherwise (False, None) — i.e. no
    adaptation, use the dataset's native format."""
    if is_conv_model(model_name) and dataset_name not in _THREE_CHANNEL_DATASETS:
        return True, 32
    return False, None

# MLP variant name -> cfg key in models.mlp.cfg ("_bn" suffix toggles BatchNorm1d,
# same convention as the VGG "_bn" suffix below). Bare "mlp" is kept as a direct
# alias to MLP's own default hidden_dims for backward compatibility with existing
# checkpoints trained before named variants existed.
_MLP_CFG_KEYS = {
    "mlp2": "MLP2",
    "mlp2_half": "MLP2_half",
    "mlp2_doub": "MLP2_doub",
    "mlp2_quad": "MLP2_quad",
    "mlp2_oct": "MLP2_oct",
    "mlp3": "MLP3",
    "mlp3_half": "MLP3_half",
    "mlp3_doub": "MLP3_doub",
    "mlp3_quad": "MLP3_quad",
    "mlp3_oct": "MLP3_oct",
}

# VGG variant name -> cfg key in models.vgg.cfg (batch_norm is handled separately
# via a "_bn" suffix, exactly as in Intra-Fusion's own model factory).
_VGG_CFG_KEYS = {
    "vgg11": "VGG11",
    "vgg11_half": "VGG11_half",
    "vgg11_doub": "VGG11_doub",
    "vgg11_quad": "VGG11_quad",
    "vgg13": "VGG13",
    "vgg16": "VGG16",
    "vgg19": "VGG19",
}

_RESNET_BUILDERS = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
}


def _build_mlp(model_name: str, dataset_name: str, num_classes: int, **kwargs):
    in_dim = kwargs.pop("in_dim", _DEFAULT_INPUT_DIMS.get(dataset_name))

    if model_name == "mlp":
        return MLP(in_dim=in_dim, num_classes=num_classes, **kwargs)

    batch_norm = model_name.endswith("_bn")
    base_name = model_name[: -len("_bn")] if batch_norm else model_name
    if base_name not in _MLP_CFG_KEYS:
        raise ValueError(f"Unknown MLP variant '{model_name}'. Options: "
                          f"{sorted(_MLP_CFG_KEYS)} (each with an optional '_bn' suffix)")
    return build_mlp(_MLP_CFG_KEYS[base_name], in_dim=in_dim, num_classes=num_classes,
                      batch_norm=batch_norm, **kwargs)


def _build_vgg(model_name: str, num_classes: int, **kwargs):
    batch_norm = model_name.endswith("_bn")
    base_name = model_name[: -len("_bn")] if batch_norm else model_name
    if base_name not in _VGG_CFG_KEYS:
        raise ValueError(f"Unknown VGG variant '{model_name}'. Options: "
                          f"{sorted(_VGG_CFG_KEYS)} (each with an optional '_bn' suffix)")
    return VGG(_VGG_CFG_KEYS[base_name], num_classes=num_classes, batch_norm=batch_norm, **kwargs)


def _is_mlp_name(model_name: str) -> bool:
    if model_name == "mlp":
        return True
    base_name = model_name[: -len("_bn")] if model_name.endswith("_bn") else model_name
    return base_name in _MLP_CFG_KEYS


def build_model(model_name: str, dataset_name: str, **kwargs):
    """model_name: 'mlp' (or a named variant, e.g. 'mlp3_doub_bn' — see
    models.mlp.cfg), 'cnn', 'lstm', a VGG variant (e.g. 'vgg11_bn',
    'vgg11_half_bn', 'vgg16'), or a ResNet variant ('resnet18' ... 'resnet152')."""
    num_classes = kwargs.pop("num_classes", _DEFAULT_NUM_CLASSES.get(dataset_name, 10))

    if _is_mlp_name(model_name):
        return _build_mlp(model_name, dataset_name, num_classes, **kwargs)

    if model_name == "cnn":
        in_channels = kwargs.pop("in_channels", 1 if dataset_name in ("MNIST", "FashionMNIST") else 3)
        return SmallCNN(in_channels=in_channels, num_classes=num_classes, **kwargs)

    if model_name == "lstm":
        return LSTMClassifier(num_classes=num_classes, **kwargs)

    if is_conv_model(model_name):
        # These hardcode a 3-channel 32x32 input stem (ported from Intra-Fusion's
        # CIFAR-only VGG/ResNet). 1-channel datasets (MNIST/FashionMNIST) are
        # handled by adapting the DATA to 3x32x32 in the data loader (see
        # utils.data_loader force_rgb/image_size), so any dataset is allowed here
        # — the trainer is responsible for requesting the adapted input via
        # conv_input_spec() below.
        if model_name.startswith("vgg"):
            return _build_vgg(model_name, num_classes, **kwargs)
        return _RESNET_BUILDERS[model_name](num_classes=num_classes, **kwargs)

    mlp_options = sorted(_MLP_CFG_KEYS) + [f"{name}_bn" for name in sorted(_MLP_CFG_KEYS)]
    vgg_options = sorted(_VGG_CFG_KEYS) + [f"{name}_bn" for name in sorted(_VGG_CFG_KEYS)]
    raise ValueError(
        f"Unknown model_name '{model_name}'. Options: mlp, cnn, lstm, "
        f"{mlp_options}, {vgg_options}, {sorted(_RESNET_BUILDERS)}"
    )
