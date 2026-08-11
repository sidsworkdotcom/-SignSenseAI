"""Train SignNet on the collected landmark dataset.

Pipeline
--------
1. Load data/landmarks.csv, encode labels, stratified train/val/test split.
2. Train with Adam + weight decay, ReduceLROnPlateau, early stopping.
3. Save the best checkpoint (highest val accuracy) + label map.
4. Plot loss/accuracy curves to outputs/training_curves.png.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.dataset import SignDataset, load_dataframe  # noqa: E402
from src.model import SignNet, count_parameters  # noqa: E402
from src.utils import set_seed  # noqa: E402


def prepare_splits():
    df = load_dataframe()
    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"].to_numpy())

    # hold out test set first, then carve val from the remainder
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=config.TEST_SPLIT, stratify=y, random_state=config.SEED
    )
    val_frac = config.VAL_SPLIT / (1 - config.TEST_SPLIT)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=config.SEED
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), encoder


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
    return total_loss / seen, correct / seen


def plot_curves(history: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_title("Loss"), ax1.set_xlabel("epoch"), ax1.legend(), ax1.grid(alpha=0.3)
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    ax2.set_title("Accuracy"), ax2.set_xlabel("epoch"), ax2.legend(), ax2.grid(alpha=0.3)
    fig.suptitle("SignNet training curves")
    fig.tight_layout()
    out = config.OUTPUT_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def main() -> None:
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    (xtr, ytr), (xva, yva), (xte, yte), encoder = prepare_splits()
    num_classes = len(encoder.classes_)
    print(
        f"Classes ({num_classes}): {list(encoder.classes_)}\n"
        f"Train {len(ytr)} | Val {len(yva)} | Test {len(yte)}"
    )

    train_dl = DataLoader(
        SignDataset(xtr, ytr, train=True), batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_dl = DataLoader(SignDataset(xva, yva), batch_size=config.BATCH_SIZE)
    test_dl = DataLoader(SignDataset(xte, yte), batch_size=config.BATCH_SIZE)

    model = SignNet(num_classes).to(device)
    print(f"SignNet parameters: {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val_acc, epochs_without_improvement = 0.0, 0
    start = time.time()

    for epoch in tqdm(range(1, config.EPOCHS + 1), desc="Training"):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device=device)
        scheduler.step(va_acc)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_without_improvement = 0
            torch.save(
                {"state_dict": model.state_dict(), "num_classes": num_classes},
                config.MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            tqdm.write(
                f"epoch {epoch:3d} | train {tr_acc:.3f} | val {va_acc:.3f} "
                f"| lr {optimizer.param_groups[0]['lr']:.1e}"
            )

    print(f"\nTraining time: {time.time() - start:.1f}s | best val acc {best_val_acc:.4f}")

    # final test-set evaluation with the best checkpoint
    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, te_acc = run_epoch(model, test_dl, criterion, device=device)
    print(f"TEST accuracy: {te_acc:.4f} (loss {te_loss:.4f})")

    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump({int(i): c for i, c in enumerate(encoder.classes_)}, f, indent=2)
    print(f"Saved model → {config.MODEL_PATH}\nSaved labels → {config.LABEL_MAP_PATH}")

    plot_curves(history)


if __name__ == "__main__":
    main()
