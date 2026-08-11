"""SignSense AI — dataset recorder.

Controls
--------
    a–z / 0–9 : record a burst of samples for that label (3-2-1 countdown)
    SPACE     : pause / resume
    TAB       : toggle the dataset panel
    q / ESC   : quit (data is saved continuously)

Record ~250 samples per sign, varying distance, angle and lighting.
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import ui  # noqa: E402
from src.utils import HandDetector, landmarks_from_mediapipe, normalize_landmarks  # noqa: E402

ALLOWED_LABELS = set("abcdefghijklmnopqrstuvwxyz0123456789")
TARGET_PER_CLASS = 250
COUNTDOWN_S = 1.2


def count_existing_samples() -> dict[str, int]:
    counts: dict[str, int] = {}
    if config.LANDMARK_CSV.exists():
        with open(config.LANDMARK_CSV, newline="") as f:
            for row in csv.reader(f):
                if row:
                    counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


class Recorder:
    def __init__(self):
        self.detector = HandDetector()
        self.counts = count_existing_samples()
        self.label: str | None = None
        self.remaining = 0
        self.countdown_until = 0.0
        self.paused = False
        self.show_panel = True
        self.frame_idx = 0
        self.csv_file = open(config.LANDMARK_CSV, "a", newline="")
        self.writer = csv.writer(self.csv_file)

    # ------------------------------------------------------------ logic ----
    def start_burst(self, label: str, now: float):
        self.label = label
        self.remaining = config.SAMPLES_PER_KEYPRESS
        self.countdown_until = now + COUNTDOWN_S

    def capture(self, hand):
        vec = normalize_landmarks(landmarks_from_mediapipe(hand))
        self.writer.writerow([self.label, *vec.tolist()])
        self.counts[self.label] = self.counts.get(self.label, 0) + 1
        self.remaining -= 1
        if self.remaining == 0:
            self.csv_file.flush()
            self.label = None

    # --------------------------------------------------------------- ui ----
    def draw_header(self, frame, hand_visible: bool, now: float):
        w = frame.shape[1]
        ui.panel(frame, (16, 14), (w - 16, 58))
        recording = self.label is not None and now >= self.countdown_until
        if recording:
            pulse = (math.sin(now * 6) + 1) / 2
            ui.dot(frame, (38, 36), ui.ACCENT_WARM, pulse=pulse)
            ui.text(frame, f"Recording  '{self.label.upper()}'", (54, 26), size=16, bold=True)
            ui.text(frame, f"{self.remaining} left", (230, 29), size=12, color=ui.INK_FAINT)
        elif self.label is not None:
            secs = self.countdown_until - now
            ui.dot(frame, (38, 36), ui.ACCENT_WARM)
            ui.text(frame, f"Get ready — '{self.label.upper()}' in {secs:.1f}s",
                    (54, 26), size=16, bold=True)
        elif self.paused:
            ui.dot(frame, (38, 36), ui.INK_FAINT)
            ui.text(frame, "Paused", (54, 26), size=16, bold=True, color=ui.INK_DIM)
        else:
            ui.dot(frame, (38, 36), ui.ACCENT if hand_visible else ui.INK_FAINT)
            ui.text(frame, "Dataset recorder", (54, 26), size=16, bold=True)
            ui.text(frame, "press a letter or digit to record", (196, 29),
                    size=12, color=ui.INK_FAINT)

        total = sum(self.counts.values())
        ui.text(frame, f"{len(self.counts)} classes  ·  {total} samples",
                (w - 34, 29), size=12, color=ui.INK_DIM, align="right")

    def draw_dataset_panel(self, frame):
        if not self.show_panel or not self.counts:
            return
        h, w = frame.shape[:2]
        rows = sorted(self.counts.items())[:16]
        ph = 46 + len(rows) * 22
        x1, y1 = w - 200 - 16, 74
        ui.panel(frame, (x1, y1), (x1 + 200, min(y1 + ph, h - 90)))
        ui.label(frame, f"dataset · target {TARGET_PER_CLASS}", (x1 + ui.PAD, y1 + 12))
        for i, (lab, n) in enumerate(rows):
            yy = y1 + 34 + i * 22
            if yy > h - 116:
                ui.text(frame, "…", (x1 + ui.PAD, yy), size=12, color=ui.INK_FAINT)
                break
            done = n >= TARGET_PER_CLASS
            ui.text(frame, lab.upper(), (x1 + ui.PAD, yy), size=13,
                    color=ui.ACCENT if done else ui.INK_DIM, bold=done)
            ui.progress(frame, (x1 + ui.PAD + 24, yy + 5), 110,
                        n / TARGET_PER_CLASS, height=4,
                        color=ui.ACCENT if done else ui.INK_FAINT)
            ui.text(frame, str(n), (x1 + 200 - ui.PAD, yy), size=11,
                    color=ui.INK_FAINT, align="right")

    def draw_footer(self, frame):
        h, w = frame.shape[:2]
        y1 = h - 54
        ui.panel(frame, (16, y1), (w - 16, h - 14))
        x = 34
        x = ui.keycap(frame, "A–Z", "record burst", (x, y1 + 10))
        x = ui.keycap(frame, "SPACE", "pause", (x, y1 + 10))
        x = ui.keycap(frame, "TAB", "panel", (x, y1 + 10))
        ui.keycap(frame, "Q", "quit", (x, y1 + 10))
        ui.text(frame, "vary distance · angle · lighting", (w - 34, y1 + 14),
                size=12, color=ui.INK_FAINT, align="right")

    # -------------------------------------------------------------- run ----
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Recorder running — press label keys to record, q to quit.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                now = time.time()
                self.frame_idx += 1

                results = self.detector.process(frame)
                hand_visible = bool(results.multi_hand_landmarks)
                ui.vignette(frame, 0.25)
                if hand_visible:
                    hand = results.multi_hand_landmarks[0]
                    self.detector.draw(frame, hand)
                    if (
                        self.label
                        and not self.paused
                        and now >= self.countdown_until
                        and self.remaining > 0
                        and self.frame_idx % config.CAPTURE_EVERY_N_FRAMES == 0
                    ):
                        self.capture(hand)

                self.draw_header(frame, hand_visible, now)
                self.draw_dataset_panel(frame)
                self.draw_footer(frame)
                cv2.imshow("SignSense AI — Recorder", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    self.paused = not self.paused
                elif key == 9:  # TAB
                    self.show_panel = not self.show_panel
                elif key != 255 and chr(key) in ALLOWED_LABELS and self.label is None:
                    pressed = chr(key)
                    # digit keys record whole-word gestures (config.WORD_GESTURES)
                    self.start_burst(config.WORD_GESTURES.get(pressed, pressed), now)
        finally:
            self.csv_file.close()
            cap.release()
            self.detector.close()
            cv2.destroyAllWindows()
            print("\nDataset summary:")
            for lab in sorted(self.counts):
                print(f"  {lab}: {self.counts[lab]}")


if __name__ == "__main__":
    Recorder().run()
