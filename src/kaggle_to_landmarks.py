"""Convert an image dataset (e.g. Kaggle ASL Alphabet) into landmarks.csv.

Expected folder layout (this is how the Kaggle ASL Alphabet dataset unzips):

    <dataset_dir>/
        A/  img001.jpg, img002.jpg, ...
        B/  ...
        C/  ...

Usage
-----
    python src/kaggle_to_landmarks.py --dataset "path/to/asl_alphabet_train" --limit 400

    --dataset  path to the folder containing one subfolder per label
    --limit    max images per class (default 400 — plenty; keeps it fast)
    --append   add to the existing landmarks.csv instead of replacing it
               (useful for mixing Kaggle data with your own recordings)

Every image is passed through MediaPipe in static-image mode; images where no
hand is detected are skipped and counted. Labels are lowercased folder names.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.utils import HandDetector, landmarks_from_mediapipe, normalize_landmarks  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="folder with one subfolder per label")
    ap.add_argument("--limit", type=int, default=400, help="max images per class")
    ap.add_argument("--append", action="store_true", help="append instead of overwrite")
    args = ap.parse_args()

    root = Path(args.dataset)
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class subfolders found in {root}")

    detector = HandDetector(static_mode=True)
    mode = "a" if args.append else "w"
    written, skipped = 0, 0

    with open(config.LANDMARK_CSV, mode, newline="") as f:
        writer = csv.writer(f)
        for cdir in class_dirs:
            label = cdir.name.lower()
            images = [p for p in sorted(cdir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
            images = images[: args.limit]
            ok_count = 0
            for img_path in tqdm(images, desc=f"{label:>8}", leave=False):
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped += 1
                    continue
                results = detector.process(img)
                if not results.multi_hand_landmarks:
                    skipped += 1
                    continue
                vec = normalize_landmarks(
                    landmarks_from_mediapipe(results.multi_hand_landmarks[0])
                )
                writer.writerow([label, *vec.tolist()])
                ok_count += 1
                written += 1
            print(f"  {label}: {ok_count}/{len(images)} images had a detectable hand")

    detector.close()
    print(
        f"\nDone. {written} samples written to {config.LANDMARK_CSV} "
        f"({skipped} images skipped — no hand detected).\n"
        f"Next: python src/train.py"
    )


if __name__ == "__main__":
    main()
