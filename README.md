# SignSense AI

Real-time American Sign Language recognition from a webcam. MediaPipe extracts
21 hand landmarks per frame; custom PyTorch networks classify them — an MLP for
static signs (fingerspelling) and an LSTM for motion signs — and the app builds
sentences on screen with text-to-speech output. A browser version runs the same
model fully client-side, so the web demo needs no server and no video ever
leaves the device.

## How it works

```
webcam -> MediaPipe Hands -> 21 landmarks -> normalize (wrist origin, scale)
       -> SignNet MLP (static signs)          -> smoothing -> sentence + TTS
       -> SignNetLSTM (motion signs, 30-frame clips) ->
```

Landmarks instead of raw pixels is the core design decision: after
normalization the input is 63 numbers describing pure hand geometry, so a ~60K
parameter MLP trains in under a minute on CPU and runs at <1 ms per inference.
A CNN on pixels would need far more data and compute, and would partly learn
background and lighting instead of hand shape.

## Results

| Model       | Task                          | Test acc. | Params | Training data |
|-------------|-------------------------------|-----------|--------|---------------|
| SignNet MLP | 28 static signs (A-Z, del, space) | 99.86%    | 60K    | Kaggle ASL Alphabet, converted to landmarks (9,398 samples) |
| SignNetLSTM | 25 motion signs               | 78.99%    | 241K   | Google asl-signs (9,486 sequences, hand landmarks only) |

Live accuracy is lower than test accuracy for both models due to domain shift
(different signer, camera, and frame rate than the training data). Notable
findings so far: MediaPipe z-coordinates hurt more than they help (removed);
horizontal-flip augmentation makes the LSTM robust to the mirrored selfie view
at a cost of 1.3 points of test accuracy; the LSTM has no out-of-vocabulary
rejection, so unknown motions map to the nearest known sign.

## Setup

Python 3.9-3.11 (MediaPipe requirement — 3.12+ ships an incompatible API).

```
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Static signs, trained from the Kaggle ASL Alphabet dataset:

```
python src/kaggle_to_landmarks.py --dataset <path to A/B/C... folders> --limit 400
python src/train.py
python src/evaluate.py         # confusion matrix, per-class F1
```

Or record your own signs with the webcam (letter keys record letters, digit
keys record the word gestures defined in config.py):

```
python src/data_collection.py
```

Motion signs train on Google's asl-signs dataset. Run `kaggle_train_lstm.py`
in a Kaggle notebook with the competition dataset attached (nothing to
download; free GPU), then place the resulting `signnet_lstm.pth` and
`lstm_label_map.json` in `models/`.

Live recognition:

```
python src/realtime.py
```

Keys: `M` toggles letter/word mode, `R` records a word sign (2 s window),
`SPACE` word break, `BACKSPACE` undo, `V` voice on/off, `C` clear, `Q` quit.

Browser demo: `python src/export_web.py` writes `web/model.json`; serve
`web/index.html` and `web/model.json` from any static host with HTTPS.

## Structure

```
config.py                  hyperparameters and paths
kaggle_train_lstm.py       self-contained Kaggle notebook script (LSTM)
src/
  utils.py                 hand detection, normalization, resampling
  data_collection.py       webcam dataset recorder
  kaggle_to_landmarks.py   Kaggle ASL Alphabet -> landmarks.csv
  prepare_asl_signs.py     asl-signs parquet -> LSTM sequences (local variant)
  dataset.py               PyTorch dataset + landmark augmentation
  model.py                 SignNet MLP, SignNetLSTM
  train.py / train_lstm.py training loops
  evaluate.py              metrics and confusion matrix
  realtime.py              live app (letter + word modes)
  export_web.py            folds BatchNorm into weights, exports model.json
  speech.py, ui.py         TTS thread, drawing helpers
web/index.html             browser demo (MediaPipe JS + JS forward pass)
docs/                      development notes, logs
```

## Planned

- Threshold tuning and larger vocabulary (50-100 signs) for word mode
- LSTM inference in the browser demo
- Two-hand and pose features to support signs the hand-only model cannot separate
- Deployment of the browser demo at signsenseai.siddheshgupta.com

## References

- Zhang et al., MediaPipe Hands: On-device Real-time Hand Tracking, 2020
- Hochreiter & Schmidhuber, Long Short-Term Memory, 1997
- Google - Isolated Sign Language Recognition, Kaggle, 2023
