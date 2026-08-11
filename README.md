# SignSense AI 🤟

**An Intelligent Sign Language Recognition System Using Deep Learning and Computer Vision**

SignSense AI recognizes American Sign Language (ASL) hand signs in real time from a webcam feed. It combines **MediaPipe** hand-landmark detection (computer vision) with a **custom neural network built in PyTorch** (deep learning) to classify signs, smooth predictions over time, and build words/sentences live on screen.

---

## 🏗️ Architecture

```
 Webcam Frame
      │
      ▼
 MediaPipe Hands ──► 21 hand landmarks (x, y, z)
      │
      ▼
 Landmark Normalization (translation + scale invariant)
      │
      ▼
 ┌─────────────────────────────┐
 │  SignNet (PyTorch MLP)      │   ← static signs (A–Z, etc.)
 │  63 → 256 → 128 → 64 → C    │
 │  BatchNorm + Dropout        │
 └─────────────────────────────┘
      │
      ▼
 Temporal smoothing (majority vote over sliding window)
      │
      ▼
 On-screen prediction + sentence builder
```

An optional **LSTM head (SignNetLSTM)** is included for dynamic gestures (signs that involve motion, e.g., "J", "Z", "hello"), trained on sequences of 30 frames of landmarks.

## ✨ Features

**Recognition engine**
- Real-time recognition (30+ FPS on CPU — no GPU required, <1 ms inference)
- Custom PyTorch MLP with BatchNorm, Dropout, early stopping, LR scheduling
- Optional LSTM model for motion-based signs
- Landmark normalization → robust to hand position, distance from camera
- On-the-fly data augmentation (jitter, rotation, scaling of landmarks)
- Full evaluation suite: accuracy/loss curves, confusion matrix, per-class report

**Real-time app UI** (custom design system in `src/ui.py` — real TTF typography,
translucent rounded panels, single accent color)
- Live prediction card with confidence meter and **top-3 alternatives**
- "Hold to type" stability meter showing how close a letter is to committing
- Sentence bar with blinking cursor, SPACE / BACKSPACE / clear support
- Commit flash animation + pulsing tracking indicator
- Corner-bracket hand framing, FPS / inference-latency / session-time readout

**Dataset recorder UI**
- 3-2-1 countdown before each recording burst
- Live per-class progress bars toward the 250-sample target (TAB toggles panel)
- Pause/resume, continuous saving (nothing lost on quit), session summary

## 📦 Installation

```bash
git clone https://github.com/sidsworkdotcom/-SignSenseAI.git
cd -SignSenseAI
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Python 3.9–3.11 recommended (MediaPipe compatibility).

## 🚀 Usage

### 1. Collect your dataset

```bash
python src/data_collection.py
```

- Press the key of the sign label you want to record (e.g. `a`, `b`, `c` …)
- Hold the sign in front of the camera; samples are recorded automatically
- Press `SPACE` to pause/resume, `q` to quit
- Landmarks are appended to `data/landmarks.csv`

Aim for **200–300 samples per sign**, varying distance, angle and lighting.

### 2. Train the model

```bash
python src/train.py
```

Produces:
- `models/signnet.pth` — best model checkpoint (by validation accuracy)
- `models/label_map.json` — class index ↔ label mapping
- `outputs/training_curves.png` — loss & accuracy curves

### 3. Evaluate

```bash
python src/evaluate.py
```

Produces a classification report and `outputs/confusion_matrix.png`.

### 4. Run real-time recognition

```bash
python src/realtime.py
```

- Predicted sign + confidence shown live
- Stable predictions get appended to the sentence bar
- `c` clears the sentence, `q` quits

## 📁 Project Structure

```
SignSenseAI/
├── config.py              # All hyperparameters & paths in one place
├── requirements.txt
├── src/
│   ├── utils.py           # Landmark extraction & normalization
│   ├── data_collection.py # Webcam dataset recorder
│   ├── dataset.py         # PyTorch Dataset + augmentation
│   ├── model.py           # SignNet (MLP) + SignNetLSTM
│   ├── train.py           # Training loop w/ early stopping
│   ├── evaluate.py        # Confusion matrix + metrics
│   └── realtime.py        # Live webcam inference app
├── data/                  # Collected landmark CSVs (gitignored)
├── models/                # Trained checkpoints
├── outputs/               # Plots & reports
└── docs/
    └── WEEKLY_PLAN.md     # Development roadmap
```

## 🧠 Why landmarks instead of raw images?

Feeding raw pixels into a CNN needs huge datasets and GPUs, and the model learns background/lighting instead of the hand shape. MediaPipe gives us 21 precise 3-D keypoints per hand; after normalization the input is only **63 numbers** that describe pure hand geometry. A compact MLP trained on a few thousand samples then reaches >98% accuracy and runs in real time on CPU — a much better engineering trade-off, and the neural network is still fully ours.

## 📊 Results (fill in after training)

| Model       | Val Accuracy | Test Accuracy | Params |
|-------------|--------------|---------------|--------|
| SignNet MLP | –            | –             | ~60K   |
| SignNetLSTM | –            | –             | ~230K  |

## 🔮 Roadmap

- [ ] Two-hand sign support
- [ ] Dynamic gesture vocabulary via LSTM
- [ ] Text-to-speech output for recognized sentences
- [ ] Web demo (Streamlit)

## 📚 References

- Zhang et al., *MediaPipe Hands: On-device Real-time Hand Tracking* (2020)
- Goodfellow, Bengio, Courville, *Deep Learning* — MLP regularization
- Hochreiter & Schmidhuber, *Long Short-Term Memory* (1997)
