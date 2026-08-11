"""SignSense AI — UI toolkit.

A small design system for the OpenCV windows so every screen shares the same
look: graphite translucent panels, one accent color, real typography (DejaVu
Sans via PIL instead of OpenCV's Hershey fonts), rounded corners, progress
bars and status dots.

All colors are BGR (OpenCV order).
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------ theme ----
INK = (234, 232, 228)          # near-white text
INK_DIM = (150, 148, 145)      # secondary text
INK_FAINT = (95, 94, 92)       # tertiary text / labels
PANEL = (26, 22, 20)           # graphite panel base
PANEL_EDGE = (52, 48, 45)      # 1px panel border
ACCENT = (151, 220, 61)        # mint  — success / confidence  (BGR)
ACCENT_WARM = (32, 176, 255)   # amber — recording / attention (BGR)
DANGER = (86, 76, 235)         # soft red
TRACK = (48, 44, 42)           # empty progress-bar track

PANEL_ALPHA = 0.82
RADIUS = 10
PAD = 14


@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load DejaVu Sans (ships with matplotlib) with a safe fallback."""
    try:
        from matplotlib import font_manager

        prop = font_manager.FontProperties(
            family="DejaVu Sans", weight="bold" if bold else "normal"
        )
        return ImageFont.truetype(font_manager.findfont(prop), size)
    except Exception:
        return ImageFont.load_default()


# ------------------------------------------------------------- primitives ----
def rounded_rect(img, p1, p2, color, radius=RADIUS, thickness=-1):
    """Draw a rounded rectangle (filled or outlined) on a BGR image."""
    x1, y1 = p1
    x2, y2 = p2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        for cx, cy, a in (
            (x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
            (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90),
        ):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)


def panel(frame, p1, p2, alpha=PANEL_ALPHA, color=PANEL, edge=PANEL_EDGE):
    """Translucent rounded panel blended onto the frame, with a 1px border."""
    overlay = frame.copy()
    rounded_rect(overlay, p1, p2, color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    rounded_rect(frame, p1, p2, edge, thickness=1)


def text(frame, s, org, size=15, color=INK, bold=False, align="left"):
    """Render text with a real TTF font. Returns rendered text width."""
    font = _font(size, bold)
    # measure
    bbox = font.getbbox(s)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = org
    if align == "right":
        x -= w
    elif align == "center":
        x -= w // 2
    # render only the patch we need (fast)
    x0, y0 = max(x - 2, 0), max(y - 2, 0)
    x1 = min(x + w + 4, frame.shape[1])
    y1 = min(y + h + int(size * 0.4) + 4, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return w
    patch = Image.fromarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(patch).text(
        (x - x0, y - y0 - bbox[1]), s, font=font, fill=(color[2], color[1], color[0])
    )
    frame[y0:y1, x0:x1] = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
    return w


def label(frame, s, org, color=INK_FAINT):
    """Small uppercase tracking label — the 'eyebrow' style."""
    text(frame, s.upper(), org, size=11, color=color, bold=True)


def progress(frame, p1, width, value, height=6, color=ACCENT, track=TRACK):
    """Thin rounded progress bar, value in [0, 1]."""
    x, y = p1
    rounded_rect(frame, (x, y), (x + width, y + height), track, radius=height // 2)
    w = int(width * max(0.0, min(1.0, value)))
    if w > height:
        rounded_rect(frame, (x, y), (x + w, y + height), color, radius=height // 2)


def dot(frame, center, color, r=5, pulse=0.0):
    """Status dot with an optional pulse ring (pulse in [0,1])."""
    if pulse > 0:
        ring = int(r + 4 + pulse * 4)
        overlay = frame.copy()
        cv2.circle(overlay, center, ring, color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6 * (1 - pulse), frame, 1 - 0.6 * (1 - pulse), 0, frame)
    cv2.circle(frame, center, r, color, -1, cv2.LINE_AA)


def keycap(frame, key, desc, org):
    """Keyboard hint like  [ C ] clear.  Returns x after the rendered hint."""
    x, y = org
    w = text(frame, key, (x + 7, y + 3), size=11, color=INK_DIM, bold=True)
    rounded_rect(frame, (x, y), (x + w + 14, y + 20), PANEL_EDGE, radius=5, thickness=1)
    x2 = x + w + 22
    x2 += text(frame, desc, (x2, y + 4), size=12, color=INK_FAINT)
    return x2 + 18


def vignette(frame, strength=0.35):
    """Subtle darkening at the frame edges so overlays pop. In-place."""
    h, w = frame.shape[:2]
    kx = cv2.getGaussianKernel(w, w * 0.7)
    ky = cv2.getGaussianKernel(h, h * 0.7)
    mask = (ky @ kx.T)
    mask = mask / mask.max()
    mask = (1 - strength) + strength * mask
    frame[:] = (frame * mask[..., None]).astype(np.uint8)
