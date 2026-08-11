"""PyTorch Dataset for landmark vectors + geometric data augmentation.

Augmentation happens on the fly, directly in landmark space (much cheaper
than image augmentation and just as effective here):

    * Gaussian jitter  — simulates MediaPipe detection noise
    * 2-D rotation     — simulates a tilted hand / camera
    * Uniform scaling  — simulates slight distance changes
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def load_dataframe(csv_path: Path = config.LANDMARK_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No dataset found at {csv_path}. Run src/data_collection.py first."
        )
    df = pd.read_csv(csv_path, header=None)
    df = df.rename(columns={0: "label"})
    return df


def augment(vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random geometric augmentation to one 63-dim landmark vector."""
    lm = vec.reshape(config.NUM_LANDMARKS, 3).copy()

    # jitter
    lm += rng.normal(0.0, config.AUG_JITTER_STD, lm.shape).astype(np.float32)

    # 2-D rotation around the wrist in the x-y plane
    theta = np.deg2rad(rng.uniform(-config.AUG_ROTATION_DEG, config.AUG_ROTATION_DEG))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    lm[:, :2] = lm[:, :2] @ rot.T

    # scaling
    lm *= rng.uniform(*config.AUG_SCALE_RANGE)

    return lm.flatten()


class SignDataset(Dataset):
    """Landmark dataset. Set train=True to enable augmentation."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train: bool = False,
        seed: int = config.SEED,
    ):
        self.x = features.astype(np.float32)
        self.y = labels.astype(np.int64)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        vec = self.x[idx]
        if self.train:
            vec = augment(vec, self.rng)
        return torch.from_numpy(np.ascontiguousarray(vec)), int(self.y[idx])
