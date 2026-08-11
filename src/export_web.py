"""Export the trained SignNet MLP to a plain-JSON model for the browser demo.

Why JSON instead of ONNX/TF.js? The network is a ~60K-parameter MLP — small
enough to ship as a ~250 KB JSON file and run the forward pass in ~30 lines
of JavaScript. No heavy runtime, works on any static host.

Trick: BatchNorm in eval mode is just a per-feature affine transform, so we
*fold* it into the preceding Linear layer:

    y = gamma * (Wx + b - mean) / sqrt(var + eps) + beta
      = (D @ W) x + (D @ (b - mean) + beta)      where D = diag(gamma / sqrt(var + eps))

The exported model is therefore a pure stack of Linear+ReLU layers — the JS
forward pass needs no BatchNorm code at all.

Usage:
    python src/export_web.py            # writes web/model.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.model import SignNet  # noqa: E402

WEB_DIR = config.ROOT_DIR / "web"
OUT_PATH = WEB_DIR / "model.json"


def fold_and_export() -> None:
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)  # {"0": "a", ...}
    labels = [label_map[str(i)] for i in range(len(label_map))]

    ckpt = torch.load(config.MODEL_PATH, map_location="cpu")
    model = SignNet(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    seq = list(model.net)  # [Linear, BN, ReLU, Dropout, ..., Linear]
    layers = []
    i = 0
    while i < len(seq):
        m = seq[i]
        if isinstance(m, torch.nn.Linear):
            W = m.weight.detach().numpy().astype(np.float64)
            b = m.bias.detach().numpy().astype(np.float64)
            # fold following BatchNorm1d if present
            if i + 1 < len(seq) and isinstance(seq[i + 1], torch.nn.BatchNorm1d):
                bn = seq[i + 1]
                gamma = bn.weight.detach().numpy()
                beta = bn.bias.detach().numpy()
                mean = bn.running_mean.detach().numpy()
                var = bn.running_var.detach().numpy()
                d = gamma / np.sqrt(var + bn.eps)
                W = W * d[:, None]
                b = d * (b - mean) + beta
                i += 1  # consume the BN
            has_relu = any(
                isinstance(seq[j], torch.nn.ReLU) for j in (i + 1, i + 2) if j < len(seq)
            )
            layers.append(
                {
                    "W": np.round(W, 6).tolist(),
                    "b": np.round(b, 6).tolist(),
                    "relu": has_relu,
                }
            )
        i += 1

    # sanity check: folded forward must match the PyTorch model
    x = torch.randn(8, config.INPUT_SIZE)
    with torch.no_grad():
        ref = model(x).numpy()
    h = x.numpy().astype(np.float64)
    for L in layers:
        h = h @ np.array(L["W"]).T + np.array(L["b"])
        if L["relu"]:
            h = np.maximum(h, 0.0)
    err = np.abs(h - ref).max()
    assert err < 1e-3, f"folded model diverges from PyTorch (max err {err})"

    WEB_DIR.mkdir(exist_ok=True)
    payload = {
        "labels": labels,
        "input_size": config.INPUT_SIZE,
        "layers": layers,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "smoothing_window": config.SMOOTHING_WINDOW,
    }
    OUT_PATH.write_text(json.dumps(payload))
    kb = OUT_PATH.stat().st_size / 1024
    print(f"Exported {len(layers)} layers, {len(labels)} classes -> {OUT_PATH} ({kb:.0f} KB)")
    print(f"Folded-vs-PyTorch max error: {err:.2e}  ✓")


if __name__ == "__main__":
    fold_and_export()
