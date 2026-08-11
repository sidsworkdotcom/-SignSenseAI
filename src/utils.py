"""Shared utilities: MediaPipe hand wrapper + landmark normalization.

The normalization step is the secret sauce of this project. Raw MediaPipe
coordinates depend on where the hand is in the frame and how far it is from
the camera. We remove both effects so the network only sees *hand shape*:

    1. Translate: subtract the wrist (landmark 0) → position invariant.
    2. Scale:     divide by the max distance from the wrist → size invariant.

The result is a 63-dim vector in roughly [-1, 1] that describes pure geometry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize a (21, 3) landmark array to be translation & scale invariant.

    Args:
        landmarks: array of shape (21, 3) with raw MediaPipe (x, y, z).

    Returns:
        Flat float32 array of shape (63,).
    """
    lm = np.asarray(landmarks, dtype=np.float32).reshape(
        config.NUM_LANDMARKS, config.FEATURES_PER_LANDMARK
    )
    # 1. translation invariance — wrist becomes the origin
    lm = lm - lm[0]
    # 2. scale invariance — normalize by the largest wrist distance
    scale = np.max(np.linalg.norm(lm, axis=1))
    if scale > 1e-6:
        lm = lm / scale
    return lm.flatten()


def resample_sequence(seq: np.ndarray, target: int) -> np.ndarray:
    """Linearly resample a (T, F) sequence to (target, F) frames."""
    if len(seq) == target:
        return seq
    idx = np.linspace(0, len(seq) - 1, target)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    frac = (idx - lo)[:, None]
    return (seq[lo] * (1 - frac) + seq[hi] * frac).astype(np.float32)


def landmarks_from_mediapipe(hand_landmarks) -> np.ndarray:
    """Convert a MediaPipe NormalizedLandmarkList to a (21, 3) numpy array."""
    return np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32
    )


class HandDetector:
    """Thin wrapper around MediaPipe Hands so all scripts share one config."""

    def __init__(self, static_mode: bool = False):
        import mediapipe as mp  # local import: heavy dependency

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self.hands = self._mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr):
        """Run detection on a BGR frame. Returns MediaPipe results object."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self.hands.process(rgb)

    def draw(self, frame_bgr, hand_landmarks) -> None:
        """Draw the landmark skeleton on the frame in-place."""
        self._mp_draw.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_styles.get_default_hand_landmarks_style(),
            self._mp_styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        self.hands.close()


def set_seed(seed: int = config.SEED) -> None:
    """Make experiments reproducible."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
