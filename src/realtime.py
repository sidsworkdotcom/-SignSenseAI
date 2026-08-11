"""SignSense AI — real-time recognition app.

Controls
--------
    SPACE      : insert a space into the sentence
    BACKSPACE  : delete last character
    c          : clear the sentence
    q / ESC    : quit

UI layout
---------
    ┌──────────────────────────────────────────────┐
    │ ● SignSense AI            fps · infer · time │   header
    │                                              │
    │                      ┌──────────────┐        │
    │        camera        │  A     94%   │        │   prediction card
    │                      │  ▓▓▓▓▓▓▓░░   │        │   + top-3 + stability
    │                      │  top-3 bars  │        │
    │                      └──────────────┘        │
    │                                              │
    │ > HELLO WORL_                    [c] [⌫] [q] │   sentence bar
    └──────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import ui  # noqa: E402
from src.model import SignNet, SignNetLSTM  # noqa: E402
from src.speech import Speaker  # noqa: E402
from src.utils import HandDetector, landmarks_from_mediapipe, normalize_landmarks, resample_sequence  # noqa: E402


# ----------------------------------------------------------------- model ----
def load_model(device: str):
    if not config.MODEL_PATH.exists() or not config.LABEL_MAP_PATH.exists():
        raise FileNotFoundError("Train a model first: python src/train.py")
    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    model = SignNet(ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with open(config.LABEL_MAP_PATH) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    return model, label_map


def load_lstm(device: str):
    """Optional word-sign model (Google asl-signs). Returns (None, None) if absent."""
    if not config.LSTM_MODEL_PATH.exists() or not config.LSTM_LABEL_MAP_PATH.exists():
        return None, None
    ckpt = torch.load(config.LSTM_MODEL_PATH, map_location=device)
    model = SignNetLSTM(ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with open(config.LSTM_LABEL_MAP_PATH) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    return model, label_map


# ------------------------------------------------------------------- app ----
class App:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.label_map = load_model(self.device)
        self.lstm, self.lstm_label_map = load_lstm(self.device)
        self.mode = "letters"                       # "letters" | "words" (M key)
        # word-mode recording: idle -> (R) countdown -> recording -> classify
        self.word_state = "idle"
        self.word_t0 = 0.0
        self.capture: list[np.ndarray] = []
        self.last_word_top3: list[tuple[str, float]] = []
        self.detector = HandDetector()

        self.window: deque[str] = deque(maxlen=config.SMOOTHING_WINDOW)
        self.sentence = ""
        self.last_commit = 0.0
        self.commit_flash = 0.0          # 1 → 0 fade after a letter commits
        self.session_start = time.time()
        self.letters_committed = 0
        self.fps = 0.0
        self.infer_ms = 0.0
        self.speaker = Speaker(enabled=config.VOICE_ENABLED)
        self.spoken_upto = 0            # index into sentence already spoken

    # ------------------------------------------------------------ logic ----
    def predict(self, hand) -> tuple[str, float, list[tuple[str, float]]]:
        vec = normalize_landmarks(landmarks_from_mediapipe(hand))
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(vec).unsqueeze(0).to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        self.infer_ms = (time.perf_counter() - t0) * 1000
        top = np.argsort(probs)[::-1][:3]
        top3 = [(self.label_map[int(i)], float(probs[i])) for i in top]
        return top3[0][0], top3[0][1], top3

    def predict_sequence(self, clip: np.ndarray) -> list[tuple[str, float]]:
        """Classify one complete captured gesture (T, 63) with the LSTM."""
        seq = resample_sequence(clip, config.SEQUENCE_LENGTH)[None, ...]
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.lstm(torch.from_numpy(seq).to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        self.infer_ms = (time.perf_counter() - t0) * 1000
        top = np.argsort(probs)[::-1][:3]
        return [(self.lstm_label_map[int(i)], float(probs[i])) for i in top]

    def word_mode_step(self, hand, now: float) -> list[tuple[str, float]]:
        """Press-R recording: countdown -> fixed window -> one classification."""
        if self.word_state == "countdown":
            if now - self.word_t0 >= config.WORD_COUNTDOWN_S:
                self.word_state, self.word_t0, self.capture = "recording", now, []
        elif self.word_state == "recording":
            if hand is not None:
                lm = landmarks_from_mediapipe(hand)
                lm[:, 2] = 0.0            # z is unreliable — model trained without it
                self.capture.append(normalize_landmarks(lm))
            if now - self.word_t0 >= config.WORD_RECORD_S:
                clip = self.capture
                self.word_state, self.capture = "idle", []
                if len(clip) >= config.MIN_SIGN_FRAMES:
                    self.last_word_top3 = self.predict_sequence(np.stack(clip))
                    sign, conf = self.last_word_top3[0]
                    print(f"[word] {len(clip)} frames | " + "  ".join(
                        f"{s}:{c*100:.0f}%" for s, c in self.last_word_top3))
                    if conf >= config.WORD_CONFIDENCE_THRESHOLD:
                        self.speak_pending_word()
                        if self.sentence and not self.sentence.endswith(" "):
                            self.sentence += " "
                        self.sentence += sign.upper() + " "
                        self.speaker.say(sign)
                        self.spoken_upto = len(self.sentence)
                        self.letters_committed += 1
                        self.last_commit = now
                        self.commit_flash = 1.0
                else:
                    print(f"[word] only {len(clip)} frames with a hand — keep your "
                          "hand in frame during the recording window")
        return self.last_word_top3

    def speak_pending_word(self) -> None:
        """Speak any complete, not-yet-spoken word in the sentence."""
        pending = self.sentence[self.spoken_upto:]
        self.speaker.say(pending)
        self.spoken_upto = len(self.sentence)

    def maybe_commit(self, now: float) -> None:
        if len(self.window) < self.window.maxlen:
            return
        vote, count = Counter(self.window).most_common(1)[0]
        if count == self.window.maxlen and now - self.last_commit > config.SENTENCE_COOLDOWN_S:
            if len(vote) > 1:
                # word gesture: append as a word with spacing, speak right away
                self.speak_pending_word()              # flush any spelled word
                if self.sentence and not self.sentence.endswith(" "):
                    self.sentence += " "
                self.sentence += vote.upper() + " "
                self.speaker.say(vote)
                self.spoken_upto = len(self.sentence)
            else:
                self.sentence += vote.upper()
            self.letters_committed += 1
            self.last_commit = now
            self.commit_flash = 1.0
            self.window.clear()

    # --------------------------------------------------------------- ui ----
    def draw_hand_brackets(self, frame, hand):
        """Corner brackets around the hand instead of a noisy full box."""
        h, w = frame.shape[:2]
        xs = [p.x for p in hand.landmark]
        ys = [p.y for p in hand.landmark]
        pad = 0.06
        x1 = int(max(min(xs) - pad, 0) * w)
        y1 = int(max(min(ys) - pad, 0) * h)
        x2 = int(min(max(xs) + pad, 1) * w)
        y2 = int(min(max(ys) + pad, 1) * h)
        L = max(14, (x2 - x1) // 6)
        c = ui.ACCENT if self.commit_flash > 0.4 else ui.INK_DIM
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * L, cy), c, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * L), c, 2, cv2.LINE_AA)

    def draw_header(self, frame, hand_visible: bool, now: float):
        w = frame.shape[1]
        ui.panel(frame, (16, 14), (w - 16, 58))
        pulse = (math.sin(now * 3) + 1) / 2 if hand_visible else 0.0
        ui.dot(frame, (38, 36), ui.ACCENT if hand_visible else ui.INK_FAINT, pulse=pulse)
        ui.text(frame, "SignSense AI", (54, 26), size=16, bold=True)
        status = "tracking" if hand_visible else "show a hand"
        if self.mode == "words":
            if self.word_state == "countdown":
                status += "  ·  WORD MODE · GET READY…"
            elif self.word_state == "recording":
                left = config.WORD_RECORD_S - (time.time() - self.word_t0)
                status += f"  ·  WORD MODE · ● SIGN NOW ({max(left,0):.1f}s)"
            else:
                status += "  ·  WORD MODE · press R to sign"
        else:
            status += "  ·  letters"
        if not self.speaker.enabled:
            status += "  ·  muted"
        ui.text(frame, status, (168, 29), size=12, color=ui.INK_FAINT)

        elapsed = int(now - self.session_start)
        right = (
            f"{self.fps:4.0f} fps   ·   {self.infer_ms:4.1f} ms infer   ·   "
            f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        )
        ui.text(frame, right, (w - 34, 29), size=12, color=ui.INK_DIM, align="right")

    def draw_prediction_card(self, frame, top3, now: float):
        w = frame.shape[:2][1]
        cw, ch = 240, 214
        x1, y1 = w - cw - 16, 74
        ui.panel(frame, (x1, y1), (x1 + cw, y1 + ch))

        ui.label(frame, "prediction", (x1 + ui.PAD, y1 + 12))
        if top3:
            sign, conf = top3[0]
            strong = conf >= config.CONFIDENCE_THRESHOLD
            color = ui.ACCENT if strong else ui.INK_DIM
            ui.text(frame, sign.upper(), (x1 + ui.PAD, y1 + 28), size=54, bold=True, color=color)
            ui.text(frame, f"{conf * 100:.0f}%", (x1 + cw - ui.PAD, y1 + 40),
                    size=20, color=ui.INK_DIM, align="right")
            ui.progress(frame, (x1 + ui.PAD, y1 + 96), cw - 2 * ui.PAD, conf, color=color)

            ui.label(frame, "alternatives", (x1 + ui.PAD, y1 + 116))
            for i, (s, p) in enumerate(top3[1:], start=0):
                yy = y1 + 136 + i * 22
                ui.text(frame, s.upper(), (x1 + ui.PAD, yy), size=13, color=ui.INK_DIM)
                ui.progress(frame, (x1 + ui.PAD + 34, yy + 5), cw - 2 * ui.PAD - 76,
                            p, height=4, color=ui.TRACK if p < 0.05 else ui.INK_FAINT)
                ui.text(frame, f"{p * 100:.0f}", (x1 + cw - ui.PAD, yy),
                        size=12, color=ui.INK_FAINT, align="right")
        else:
            ui.text(frame, "—", (x1 + ui.PAD, y1 + 28), size=54, bold=True, color=ui.INK_FAINT)
            ui.text(frame, "waiting for a hand", (x1 + ui.PAD, y1 + 100),
                    size=12, color=ui.INK_FAINT)

        # stability meter — how close the vote window is to committing
        ui.label(frame, "hold to type", (x1 + ui.PAD, y1 + ch - 30))
        fill = len(self.window) / self.window.maxlen
        bar_color = ui.ACCENT if fill == 1.0 else ui.ACCENT_WARM
        ui.progress(frame, (x1 + ui.PAD, y1 + ch - 12), cw - 2 * ui.PAD, fill, color=bar_color)

    def draw_sentence_bar(self, frame, now: float):
        h, w = frame.shape[:2]
        y1 = h - 64
        ui.panel(frame, (16, y1), (w - 16, h - 14))

        # commit flash — brief accent underline when a letter lands
        if self.commit_flash > 0:
            overlay = frame.copy()
            ui.rounded_rect(overlay, (16, y1), (w - 16, h - 14), ui.ACCENT, thickness=1)
            cv2.addWeighted(overlay, self.commit_flash * 0.9, frame,
                            1 - self.commit_flash * 0.9, 0, frame)
            self.commit_flash = max(0.0, self.commit_flash - 0.06)

        shown = self.sentence[-34:]
        cursor = "|" if int(now * 2) % 2 == 0 else " "
        if shown:
            ui.text(frame, shown + cursor, (34, y1 + 16), size=22)
        else:
            ui.text(frame, cursor, (34, y1 + 16), size=22)
            ui.text(frame, "hold a sign steady to type", (52, y1 + 20),
                    size=13, color=ui.INK_FAINT)

        x = w - 512
        x = ui.keycap(frame, "SPACE", "space", (x, y1 + 15))
        x = ui.keycap(frame, "M", "mode", (x, y1 + 15))
        x = ui.keycap(frame, "V", "voice", (x, y1 + 15))
        x = ui.keycap(frame, "BKSP", "undo", (x, y1 + 15))
        x = ui.keycap(frame, "C", "clear", (x, y1 + 15))
        ui.keycap(frame, "Q", "quit", (x, y1 + 15))

    # -------------------------------------------------------------- run ----
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        prev = time.time()
        print("SignSense AI running — q to quit.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - prev, 1e-6))
                prev = now

                top3 = []
                results = self.detector.process(frame)
                hand_visible = bool(results.multi_hand_landmarks)
                ui.vignette(frame, 0.25)
                if self.mode == "words" and self.lstm is not None:
                    hand = results.multi_hand_landmarks[0] if hand_visible else None
                    if hand is not None:
                        self.draw_hand_brackets(frame, hand)
                    top3 = self.word_mode_step(hand, now)
                elif hand_visible:
                    hand = results.multi_hand_landmarks[0]
                    self.draw_hand_brackets(frame, hand)
                    sign, conf, top3 = self.predict(hand)
                    if conf >= config.CONFIDENCE_THRESHOLD:
                        self.window.append(sign)
                    self.maybe_commit(now)
                else:
                    self.window.clear()

                # auto word-break: pause in signing => word is done, speak it
                if (self.sentence[self.spoken_upto:].strip()
                        and now - self.last_commit > config.WORD_PAUSE_S):
                    self.speak_pending_word()

                self.draw_header(frame, hand_visible, now)
                self.draw_prediction_card(frame, top3, now)
                self.draw_sentence_bar(frame, now)
                cv2.imshow("SignSense AI", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    self.sentence = ""
                    self.spoken_upto = 0
                elif key == ord(" "):
                    self.speak_pending_word()
                    self.sentence += " "
                    self.spoken_upto = len(self.sentence)
                elif key == ord("v"):
                    self.speaker.toggle()
                elif key == ord("m"):
                    if self.lstm is None:
                        print("[mode] no word model — put signnet_lstm.pth + "
                              "lstm_label_map.json in models/ (see kaggle_train_lstm.py)")
                    else:
                        self.mode = "words" if self.mode == "letters" else "letters"
                        self.window.clear()
                        self.word_state, self.capture = "idle", []
                        self.last_word_top3 = []
                elif key == ord("r"):
                    if self.mode == "words" and self.word_state == "idle":
                        self.word_state, self.word_t0 = "countdown", now
                elif key in (8, 127):  # backspace / delete
                    self.sentence = self.sentence[:-1]
                    self.spoken_upto = min(self.spoken_upto, len(self.sentence))
        finally:
            cap.release()
            self.detector.close()
            self.speaker.close()
            cv2.destroyAllWindows()
            mins = (time.time() - self.session_start) / 60
            print(f"Session: {mins:.1f} min · {self.letters_committed} letters typed")


if __name__ == "__main__":
    App().run()
