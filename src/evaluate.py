"""Evaluate the trained SignNet: classification report + confusion matrix."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.dataset import SignDataset  # noqa: E402
from src.model import SignNet  # noqa: E402
from src.train import prepare_splits  # noqa: E402
from src.utils import set_seed  # noqa: E402


def main() -> None:
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run src/train.py first.")

    _, _, (x_test, y_test), encoder = prepare_splits()
    class_names = list(encoder.classes_)

    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    model = SignNet(ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(SignDataset(x_test, y_test), batch_size=config.BATCH_SIZE)
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(np.asarray(yb))

    print(classification_report(targets, preds, target_names=class_names, digits=3))

    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, class_names)

    report = classification_report(
        targets, preds, target_names=class_names, output_dict=True
    )
    with open(config.OUTPUT_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report → {config.OUTPUT_DIR / 'classification_report.json'}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    size = max(8, len(class_names) * 0.45)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(
        cm,
        annot=len(class_names) <= 30,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SignNet — Confusion Matrix (test set)")
    plt.tight_layout()
    out = config.OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
