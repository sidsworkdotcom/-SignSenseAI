# SignSense AI — Weekly Development Plan

Development roadmap for the semester. Each week has a concrete deliverable
that gets committed, plus a short log entry describing what was done, what
worked, and what didn't.

## Week 1 — Project Setup & Research
- Initialize repo: README, requirements.txt, .gitignore, project structure
- Literature review: compare raw-image CNN vs. landmark-based approaches
- Decision log: chose MediaPipe landmarks + custom MLP (documented in README)
- **Commit:** repo skeleton + README with architecture diagram

## Week 2 — Hand Detection & Landmark Pipeline
- Integrate MediaPipe Hands (`src/utils.py`)
- Implement + test landmark normalization (translation & scale invariance)
- Verify detection quality across lighting conditions
- **Commit:** `utils.py` + notes on normalization math

## Week 3 — Data Collection Tool
- Build interactive webcam recorder (`src/data_collection.py`)
- HUD showing per-class sample counts, pause/resume
- Record first pilot dataset (5 signs × 100 samples)
- **Commit:** `data_collection.py` + dataset statistics in log

## Week 4 — Dataset & Augmentation
- PyTorch `Dataset` class (`src/dataset.py`)
- Landmark-space augmentation: jitter, rotation, scaling
- Stratified train/val/test splitting
- **Commit:** `dataset.py` + before/after augmentation visualization

## Week 5 — Model Architecture
- Implement SignNet MLP (`src/model.py`): BatchNorm, Dropout, ~60K params
- Sanity checks: shapes, parameter count, overfit on tiny batch
- **Commit:** `model.py` + architecture justification in log

## Week 6 — Training Pipeline
- Full training loop (`src/train.py`): Adam, weight decay, LR scheduling,
  early stopping, best-checkpoint saving
- First real training run on pilot dataset
- **Commit:** `train.py` + training curves PNG

## Week 7 — Full Dataset & Retraining
- Expand dataset to full ASL alphabet (24 static letters), ~250 samples each
- Retrain; log accuracy improvements vs. pilot dataset
- **Commit:** updated label map, new training curves, results table in README

## Week 8 — Evaluation & Error Analysis
- Evaluation suite (`src/evaluate.py`): confusion matrix, per-class F1
- Error analysis: which sign pairs get confused (e.g., M/N, A/S)? Why?
- Targeted extra data collection for weak classes
- **Commit:** `evaluate.py` + confusion matrix + written error analysis

## Week 9 — Real-time Application
- Live inference app (`src/realtime.py`)
- Prediction smoothing (majority-vote window), confidence threshold,
  sentence builder with cooldown
- **Commit:** `realtime.py` + demo GIF/video

## Week 10 — Hyperparameter Tuning
- Ablations: hidden sizes, dropout, augmentation on/off, window size
- Document each experiment's result in a table
- **Commit:** tuning results table + final chosen config

## Week 11 — Stretch Goal: Dynamic Gestures (LSTM)
- Sequence recorder + SignNetLSTM training on motion signs (J, Z, "hello")
- **Commit:** LSTM branch + comparison MLP vs. LSTM

## Week 12 — Final Polish & Report
- Code cleanup, docstrings, final README with results
- Record demo video, prepare presentation slides
- **Commit:** final release tag `v1.0`
