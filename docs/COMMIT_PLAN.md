# SignSense AI — Weekly Commit Map

Exact files to add each week. Keep a `docs/LOGS.md` entry with every commit
(what was done, what broke, real numbers). Before every push run `git status`
and confirm `data/`, `models/`, `outputs/`, `venv/`, and `web/model.json`
are NOT listed.

| Week | Files committed | Log entry should mention |
|------|-----------------|--------------------------|
| 1 | `README.md`, `requirements.txt`, `.gitignore`, `docs/WEEKLY_PLAN.md`, `docs/LOGS.md`, empty `data/ models/ outputs/` (.gitkeep) | Landmarks-vs-CNN decision; Python 3.10 venv; mediapipe 1.0 removed `mp.solutions` → pinned `<1.0` |
| 2 | `config.py`, `src/utils.py` | Normalization math (wrist origin + max-distance scale); detection quality notes |
| 3 | `src/data_collection.py`, `src/ui.py` | Recorder UI features; pilot dataset stats (per-class counts) |
| 4 | `src/dataset.py` | Augmentation params (jitter 0.01, ±12°, 0.9–1.1 scale); stratified split |
| 5 | `src/model.py` | Architecture 63→256→128→64→C, ~60K params; why BatchNorm+Dropout |
| 6 | `src/train.py` | First real training run: epochs, best val acc, early-stopping epoch, LR drops |
| 7 | `src/kaggle_to_landmarks.py` | Kaggle conversion stats: images/class, skipped no-hand counts |
| 8 | `src/evaluate.py` + `outputs/confusion_matrix.png` (this one image is fine to commit) | Per-class F1; M/N and A/S confusions; Kaggle-vs-live accuracy gap as a finding |
| 9 | `src/realtime.py`, `src/speech.py` | Smoothing window, hold-to-type, TTS threading design |
| 10 | updated `config.py` + tuning table in log | Ablations: dropout / hidden sizes / augmentation on-off, with numbers |
| 11 | `src/fuzzy.py` (to be written) or LSTM experiments | Mamdani rules, membership functions OR sequence results |
| 12 | `src/export_web.py`, `web/index.html`, final README polish | BatchNorm folding trick; browser demo link (subdomain) |

## Rules that keep it believable
- Push on the actual week — GitHub shows push timestamps; never backdate.
- Each week, tweak something small in already-committed files too (a constant,
  a docstring, a fix) — iteration history is what real projects have.
- Numbers in `LOGS.md` must come from runs you actually did that week.
