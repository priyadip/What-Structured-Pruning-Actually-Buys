"""Loading the compressed models, folding in a reduced input, and evaluating.

Only two things are ever taken from a compressed checkpoint: its **first-layer
weights** `(W_1, b_1)` — which is all the embedding objective is allowed to see —
and the frozen remainder of the network, used solely to score a candidate. No
weight past layer 0 is read, written, or differentiated.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mlp import MLP  # noqa: E402
from utils.config import DATASETS  # noqa: E402
from utils.data_loader import get_dataloaders  # noqa: E402

EXPLORE_RESULTS = PROJECT_ROOT / "compression_mlp" / "results"


@dataclass
class CompressedModel:
    tag: str                 # e.g. cifar10_mlp3_quad_bn_frontier
    dataset: str
    model_name: str
    spec: dict
    model: MLP               # the compressed network, frozen
    W1: torch.Tensor         # [M1, D]
    b1: torch.Tensor         # [M1]

    @property
    def dense_acc(self) -> float:
        return float(self.spec["original_acc"])

    @property
    def compressed_acc(self) -> float:
        return float(self.spec["compressed_acc"])

    @property
    def in_dim(self) -> int:
        return int(self.spec["in_dim"])


def discover_compressed(kind: str = "frontier", results_dir: Path = EXPLORE_RESULTS
                        ) -> list[tuple[Path, str]]:
    """Every `*_{kind}.arch.json` produced by the hidden-layer study."""
    out = []
    for p in sorted(results_dir.glob(f"*_{kind}.arch.json")):
        out.append((p, p.name[: -len(".arch.json")]))
    return out


def load_compressed(spec_path: Path, device: str) -> CompressedModel:
    spec = json.loads(spec_path.read_text())
    weights = spec_path.with_name(spec_path.name.replace(".arch.json", ".safetensors"))
    model = MLP(in_dim=spec["in_dim"], hidden_dims=spec["hidden_dims"],
                num_classes=spec["num_classes"], activation=spec["activation"],
                batch_norm=spec["batch_norm"])
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():           # belt and braces: nothing here trains
        p.requires_grad_(False)

    src = spec["source_checkpoint"]
    dataset = next(d for d in DATASETS if src.lower().startswith(d.lower() + "_"))
    model_name = Path(src).stem[len(dataset) + 1:]
    W1 = model.layers[0].weight.detach().clone()
    b1 = model.layers[0].bias.detach().clone()
    return CompressedModel(spec_path.name[: -len(".arch.json")], dataset, model_name,
                           spec, model, W1, b1)


@torch.no_grad()
def load_flat_data(dataset: str, device: str, batch_size: int = 1024,
                   max_train: int | None = None, seed: int = 0):
    """Flattened (X_train, y_train, X_test, y_test) with the test-time transform.

    The project's train loader is built with `shuffle=True`, so draining it
    returns the rows in a different order on every run. That silently made `X_tr`
    -- and therefore the GRIDE subsample, `d_input`, and the entire `r` grid --
    non-reproducible: the same configuration came out at r=79 on one run and r=81
    on the next. Seeding immediately before the drain pins the permutation.
    """
    train_loader, test_loader = get_dataloaders(dataset, batch_size=batch_size,
                                                num_workers=4, augment=False)
    torch.manual_seed(seed)
    def drain(loader, cap):
        xs, ys, n = [], [], 0
        for x, y in loader:
            xs.append(x.view(x.size(0), -1).to(device))
            ys.append(y.to(device))
            n += x.size(0)
            if cap is not None and n >= cap:
                break
        X, Y = torch.cat(xs), torch.cat(ys)
        return (X[:cap], Y[:cap]) if cap is not None else (X, Y)
    Xtr, ytr = drain(train_loader, max_train)
    Xte, yte = drain(test_loader, None)
    return Xtr, ytr, Xte, yte


def build_reduced_model(cm: CompressedModel, A: torch.Tensor, c: torch.Tensor,
                        device: str) -> MLP:
    """Compressed net with its first layer replaced by `z -> A z + (b1 + c)`.

    The reconstruction target is `X W1^T` *without* the bias, so the fitted
    intercept `c` is added to the original `b1` rather than replacing it.
    """
    spec = cm.spec
    r = A.shape[1]
    m = MLP(in_dim=r, hidden_dims=spec["hidden_dims"], num_classes=spec["num_classes"],
            activation=spec["activation"], batch_norm=False)
    sd = cm.model.state_dict()
    with torch.no_grad():
        for k, v in sd.items():
            if k.startswith("layers.0."):
                continue
            m.state_dict()[k].copy_(v)
        m.layers[0].weight.copy_(A)
        m.layers[0].bias.copy_(cm.b1 + c)
    return m.to(device).eval()


@torch.no_grad()
def accuracy_on_codes(model: nn.Module, Z: torch.Tensor, y: torch.Tensor,
                      batch: int = 4096) -> float:
    ok = 0
    for i in range(0, Z.shape[0], batch):
        ok += (model(Z[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return 100.0 * ok / max(Z.shape[0], 1)


@torch.no_grad()
def fit_A_from_codes(Z: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-6):
    """LS fit of the new first layer given codes `Z [n,r]` and targets `Y [n,M1]`.

    Every encoder — srGW-MDS or baseline — gets its `A` this way, so the
    comparison isolates the *encoder* and never the fitting procedure.
    """
    Zd, Yd = Z.double(), Y.double()
    n = Zd.shape[0]
    Phi = torch.cat([Zd, torch.ones(n, 1, dtype=Zd.dtype, device=Zd.device)], 1)
    G = Phi.t() @ Phi
    lam = ridge * (G.diagonal()[:-1].mean() + 1e-12)
    reg = torch.zeros(G.shape[0], dtype=Zd.dtype, device=Zd.device)
    reg[:-1] = lam
    Th = torch.linalg.solve(G + torch.diag(reg), Phi.t() @ Yd)
    return Th[:-1].t().float(), Th[-1].float()


def save_reduced(model: MLP, encoder, cm: CompressedModel, meta: dict, out_dir: Path,
                 kind: str = "best"):
    """Save the fully compressed model plus the encoder needed to run it.

    Two artefacts are worth keeping: `best` (highest accuracy) and `frontier`
    (smallest `r` that still clears the window). The frontier one is the result
    worth quoting; the most accurate one is the least compressed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if kind == "best" else f"_{kind}"
    tag = f"final_fully_compressed_{cm.tag}{suffix}"
    sd = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    if getattr(encoder, "is_linear", False):
        sd["encoder.P"] = encoder.P.contiguous().cpu()
        sd["encoder.q"] = encoder.q.contiguous().cpu()
    else:
        sd["encoder.landmarks"] = encoder.Xl.contiguous().cpu()
        sd["encoder.Z"] = (encoder.Zc + encoder.mu).contiguous().cpu()
    save_file(sd, str(out_dir / f"{tag}.safetensors"))
    (out_dir / f"{tag}.arch.json").write_text(json.dumps(meta, indent=2))
    return out_dir / f"{tag}.safetensors"
