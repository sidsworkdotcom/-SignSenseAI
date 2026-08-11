# SignSense AI — Kaggle GPU training script (word signs, LSTM)
# ============================================================
# HOW TO USE (on kaggle.com):
#   1. Open the "Google - Isolated Sign Language Recognition" competition
#   2. Click "Code" tab -> "New Notebook"  (dataset is auto-attached, 0 GB downloaded)
#   3. Right sidebar: Session options -> Accelerator -> GPU P100 (or any GPU)
#   4. Delete the default cell content, paste this ENTIRE file, press Run All
#   5. When it finishes (~30-45 min), download from the Output panel:
#        signnet_lstm.pth  +  lstm_label_map.json  +  sequences_asl_signs.npz
#      and put the .pth + .json into your local models/ folder.
#
# Tune here:
NUM_SIGNS = 25        # start with 25; raise later (max 250)
LIMIT_PER_SIGN = 400  # sequences per sign
# ============================================================

from collections import Counter
from pathlib import Path
import json, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm, trange

DATASET = Path("/kaggle/input/asl-signs")
OUT = Path("/kaggle/working")
SEQUENCE_LENGTH = 30
NUM_LANDMARKS, FEATS = 21, 3
INPUT_SIZE = NUM_LANDMARKS * FEATS
SEED = 42
BATCH_SIZE = 128
EPOCHS = 150
LR, WEIGHT_DECAY = 1e-3, 1e-4
EARLY_STOP, LR_PATIENCE, LR_FACTOR = 15, 7, 0.5
LSTM_HIDDEN, LSTM_LAYERS, DROPOUT = 128, 2, 0.3
AUG_JITTER, AUG_ROT_DEG, AUG_SCALE = 0.01, 12, (0.9, 1.1)

torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---------------------------------------------------------------- convert ----
def normalize_frame(lm):
    lm = lm - lm[0]
    s = np.max(np.linalg.norm(lm, axis=1))
    return (lm / s if s > 1e-6 else lm).flatten()

def resample(seq, target=SEQUENCE_LENGTH):
    if len(seq) == target: return seq
    idx = np.linspace(0, len(seq) - 1, target)
    lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
    frac = (idx - lo)[:, None]
    return seq[lo] * (1 - frac) + seq[hi] * frac

def extract(parquet_path):
    df = pd.read_parquet(parquet_path, columns=["frame","type","landmark_index","x","y","z"])
    best = None
    for hand in ("right_hand", "left_hand"):
        h = df[df["type"] == hand].dropna(subset=["x"])
        if h.empty: continue
        per_frame = []
        for _, g in h.groupby("frame"):
            if len(g) != NUM_LANDMARKS: continue
            lm = g.sort_values("landmark_index")[["x","y","z"]].to_numpy(np.float32).copy()
            lm[:, 2] = 0.0   # z is unreliable (per the competition docs) — drop it
            per_frame.append(normalize_frame(lm))
        if len(per_frame) >= 4 and (best is None or len(per_frame) > len(best)):
            best = np.stack(per_frame)
    return best

train_csv = pd.read_csv(DATASET / "train.csv")
keep = [s for s, _ in Counter(train_csv["sign"]).most_common(NUM_SIGNS)]
train_csv = train_csv[train_csv["sign"].isin(keep)]
labels = sorted(keep)
lab2idx = {s: i for i, s in enumerate(labels)}
print(f"Signs ({len(labels)}):", labels)

X, y, skipped = [], [], 0
for sign in labels:
    rows = train_csv[train_csv["sign"] == sign].head(LIMIT_PER_SIGN)
    for _, r in tqdm(list(rows.iterrows()), desc=sign, leave=False):
        seq = extract(DATASET / r["path"])
        if seq is None: skipped += 1; continue
        X.append(resample(seq).astype(np.float32)); y.append(lab2idx[sign])
X = np.stack(X); y = np.array(y, dtype=np.int64)
np.savez_compressed(OUT / "sequences_asl_signs.npz", X=X, y=y, labels=np.array(labels))
print(f"Converted: {X.shape}  ({skipped} skipped — no usable hand)")

# ------------------------------------------------------------------ model ----
class SignNetLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, LSTM_HIDDEN, LSTM_LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, 64), nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT), nn.Linear(64, num_classes))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])

def augment(x):
    b, t, f = x.shape
    lm = x.view(b, t, NUM_LANDMARKS, 3).clone()
    # random horizontal flip per sample — makes the model handedness/mirror proof
    flip = (torch.rand(b, device=x.device) < 0.5).float() * -2 + 1  # ±1
    lm[..., 0] *= flip[:, None, None]
    lm += torch.randn_like(lm) * AUG_JITTER
    ang = (torch.rand(b, device=x.device) * 2 - 1) * np.deg2rad(AUG_ROT_DEG)
    cos, sin = torch.cos(ang), torch.sin(ang)
    xr = lm[...,0]*cos[:,None,None] - lm[...,1]*sin[:,None,None]
    yr = lm[...,0]*sin[:,None,None] + lm[...,1]*cos[:,None,None]
    lm[...,0], lm[...,1] = xr, yr
    lm *= torch.empty(b, device=x.device).uniform_(*AUG_SCALE)[:,None,None,None]
    return lm.view(b, t, f)

# ------------------------------------------------------------------ train ----
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
print(f"Train {len(X_tr)} | Val {len(X_va)} | Test {len(X_te)}")
mk = lambda a, b, sh: DataLoader(TensorDataset(torch.tensor(a), torch.tensor(b)), batch_size=BATCH_SIZE, shuffle=sh)
tr, va, te = mk(X_tr, y_tr, True), mk(X_va, y_va, False), mk(X_te, y_te, False)

model = SignNetLSTM(len(labels)).to(device)
print("Params:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=LR_FACTOR, patience=LR_PATIENCE)
crit = nn.CrossEntropyLoss()

best_acc, best_state, patience, hist = 0.0, None, 0, {"train": [], "val": []}
t0 = time.time()
for epoch in trange(EPOCHS, desc="Training"):
    model.train(); c = n = 0
    for xb, yb in tr:
        xb, yb = augment(xb.to(device)), yb.to(device)
        opt.zero_grad(); out = model(xb); loss = crit(out, yb)
        loss.backward(); opt.step()
        c += (out.argmax(1) == yb).sum().item(); n += len(yb)
    tr_acc = c / n
    model.eval(); c = n = 0
    with torch.no_grad():
        for xb, yb in va:
            out = model(xb.to(device)); yb = yb.to(device)
            c += (out.argmax(1) == yb).sum().item(); n += len(yb)
    va_acc = c / n
    hist["train"].append(tr_acc); hist["val"].append(va_acc); sched.step(1 - va_acc)
    if va_acc > best_acc:
        best_acc, patience = va_acc, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience += 1
        if patience >= EARLY_STOP: print(f"Early stopping at epoch {epoch+1}"); break
    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1:3d} | train {tr_acc:.3f} | val {va_acc:.3f}")

model.load_state_dict(best_state); model.eval(); c = n = 0
with torch.no_grad():
    for xb, yb in te:
        out = model(xb.to(device)); yb = yb.to(device)
        c += (out.argmax(1) == yb).sum().item(); n += len(yb)
print(f"\nTime {time.time()-t0:.0f}s | best val {best_acc:.4f} | TEST accuracy: {c/n:.4f}")

torch.save({"state_dict": model.state_dict(), "num_classes": len(labels)}, OUT / "signnet_lstm.pth")
(OUT / "lstm_label_map.json").write_text(json.dumps({str(i): s for i, s in enumerate(labels)}))
print("Saved: signnet_lstm.pth, lstm_label_map.json  →  Output panel (right side)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(7,4)); plt.plot(hist["train"], label="train"); plt.plot(hist["val"], label="val")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "lstm_training_curves.png", dpi=120)
print("Saved: lstm_training_curves.png")
