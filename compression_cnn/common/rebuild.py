"""Save a compressed CNN, and rebuild it from disk through an independent path.

The rebuild deliberately shares no code with the compression driver: it reads
the architecture spec, reconstructs the modules from their recorded shapes, and
loads the weights `strict=True`. A shape or bookkeeping mismatch therefore
raises instead of silently producing a model that evaluates to something
plausible. In the MLP study three of the four real bugs surfaced only this way,
which is why nothing here is quoted until `verify.py` has re-derived it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_factory import build_model  # noqa: E402

from .factored import factored_from_specs     # noqa: E402
from .foldbn import fold_bn                   # noqa: E402
from .surgery import get_module, set_module   # noqa: E402


def module_spec(path: str, m: nn.Module) -> dict:
    if isinstance(m, nn.Conv2d):
        return {"path": path, "type": "Conv2d", "in": m.in_channels, "out": m.out_channels,
                "kernel": list(m.kernel_size), "stride": list(m.stride),
                "padding": list(m.padding), "dilation": list(m.dilation),
                "groups": m.groups, "bias": m.bias is not None}
    if isinstance(m, nn.Linear):
        return {"path": path, "type": "Linear", "in": m.in_features, "out": m.out_features,
                "bias": m.bias is not None}
    raise TypeError(type(m))


def arch_spec(model: nn.Module) -> list[dict]:
    return [module_spec(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def _make(spec: dict) -> nn.Module:
    if spec["type"] == "Conv2d":
        return nn.Conv2d(spec["in"], spec["out"], tuple(spec["kernel"]),
                         tuple(spec["stride"]), tuple(spec["padding"]),
                         tuple(spec["dilation"]), spec["groups"], bias=spec["bias"])
    return nn.Linear(spec["in"], spec["out"], bias=spec["bias"])


def save_compressed(model: nn.Module, meta: dict, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["modules"] = arch_spec(model)
    sd = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(out_dir / f"{tag}.safetensors"))
    (out_dir / f"{tag}.arch.json").write_text(json.dumps(meta, indent=2))
    return out_dir / f"{tag}.safetensors"


def load_compressed(arch_path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    meta = json.loads(Path(arch_path).read_text())
    model = build_model(meta["model_name"], meta["dataset"])
    model = fold_bn(model, meta["family"])            # BN -> Identity, same object graph

    # A patch-space factorisation replaces one conv by a `proj`/`mix` pair, so the
    # spec names children that do not exist in the dense skeleton. Install those
    # parents first, at the rank recorded for `proj`, before fixing any shapes.
    by_path = {s["path"]: s for s in meta["modules"]}
    factored = set()
    for path, spec in by_path.items():
        if not path.endswith(".proj"):
            continue
        parent = path[: -len(".proj")]
        if f"{parent}.mix" not in by_path:
            continue
        set_module(model, parent, factored_from_specs(spec, by_path[f"{parent}.mix"]))
        factored.add(parent)

    for spec in meta["modules"]:
        if any(spec["path"].startswith(p + ".") for p in factored):
            continue                                  # shapes fixed by empty_factored
        cur = get_module(model, spec["path"])
        if isinstance(cur, nn.Conv2d) and (cur.in_channels, cur.out_channels) == \
                (spec["in"], spec["out"]):
            continue
        if isinstance(cur, nn.Linear) and (cur.in_features, cur.out_features) == \
                (spec["in"], spec["out"]):
            continue
        set_module(model, spec["path"], _make(spec))
    w = Path(arch_path).with_name(Path(arch_path).name.replace(".arch.json", ".safetensors"))
    model.load_state_dict(load_file(str(w), device="cpu"), strict=True)
    return model.to(device).eval(), meta


# ---------------------------------------------------------------------------
# cost accounting
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, input_shape: tuple[int, ...], device: str = "cpu") -> int:
    """Multiply-accumulates for one forward pass.

    Parameters are the wrong headline number for a CNN: a 3x3 conv's weights are
    reused at every spatial position, so a layer with 1% of the parameters can
    still dominate the compute. Both are reported, and where they disagree the
    disagreement is the finding.
    """
    total = 0
    hs = []

    def hook(mod, inp, out):
        nonlocal total
        if isinstance(mod, nn.Conv2d):
            total += int(out.numel() / out.shape[0]) * (
                mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, nn.Linear):
            total += mod.in_features * mod.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hs.append(m.register_forward_hook(hook))
    try:
        model.to(device).eval()(torch.zeros(1, *input_shape, device=device))
    finally:
        for h in hs:
            h.remove()
    return total
