"""Central configuration for SignSense AI.

Every tunable knob of the project lives here so experiments are reproducible
and nothing is hard-coded across scripts.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

LANDMARK_CSV = DATA_DIR / "landmarks.csv"        # static-sign dataset
SEQUENCE_DIR = DATA_DIR / "sequences"            # dynamic-gesture dataset
MODEL_PATH = MODEL_DIR / "signnet.pth"
LSTM_MODEL_PATH = MODEL_DIR / "signnet_lstm.pth"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"

for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- mediapipe ----
NUM_LANDMARKS = 21                # MediaPipe hand keypoints
FEATURES_PER_LANDMARK = 3         # x, y, z
INPUT_SIZE = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6

# -------------------------------------------------------- data collection ----
SAMPLES_PER_KEYPRESS = 25         # samples captured per recording burst
CAPTURE_EVERY_N_FRAMES = 2        # skip frames so samples aren't near-identical

# --------------------------------------------------------------- training ----
SEED = 42
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4               # L2 regularization
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_PATIENCE = 7
LR_SCHEDULER_FACTOR = 0.5

# MLP architecture
HIDDEN_SIZES = [256, 128, 64]
DROPOUT = 0.3

# --------------------------------------------------------- augmentation ----
AUG_JITTER_STD = 0.01             # gaussian noise on normalized landmarks
AUG_ROTATION_DEG = 12             # random 2-D rotation range (±)
AUG_SCALE_RANGE = (0.9, 1.1)      # random uniform scaling

# ------------------------------------------------------------ LSTM branch ----
SEQUENCE_LENGTH = 30              # frames per dynamic-gesture sample
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2

# --------------------------------------------------------------- realtime ----
SMOOTHING_WINDOW = 12             # sliding window for majority vote
CONFIDENCE_THRESHOLD = 0.80      # ignore low-confidence predictions
SENTENCE_COOLDOWN_S = 1.2         # min seconds between appended letters
VOICE_ENABLED = True              # speak recognized words (pyttsx3)
WORD_PAUSE_S = 2.5                # pause in signing that ends a word

# ---------------------------------------------------------- word gestures ----
# In the recorder, pressing a DIGIT key records samples for a whole-word sign
# instead of a letter. Committed word labels are appended to the sentence with
# spaces and spoken immediately. Edit freely — the model learns whatever you
# record.
WORD_GESTURES = {
    "1": "hello",
    "2": "thanks",
    "3": "iloveyou",
    "4": "yes",
    "5": "no",
}

# ------------------------------------------------------- LSTM word mode ----
# Word mode (M key). Press R to record: short get-ready pause, then a fixed
# recording window while you perform the sign, then one classification.
WORD_CONFIDENCE_THRESHOLD = 0.60  # accept a recorded sign above this
WORD_COUNTDOWN_S = 0.8            # get-ready pause after pressing R
WORD_RECORD_S = 2.0               # recording window length (seconds)
MIN_SIGN_FRAMES = 8               # need at least this many frames with a hand
LSTM_LABEL_MAP_PATH = MODEL_DIR / "lstm_label_map.json"
