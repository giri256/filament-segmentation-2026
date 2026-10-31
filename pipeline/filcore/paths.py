"""Where the competition data and the preprocessing cache live.

FILAMENT_DATA_ROOT names the competition's MAGFiLO_1.0_Kaggle_2026 folder (train/ and test/ inside it) and
FILAMENT_CACHE_DIR, optionally, the folder preprocess.py fills (flat/, mask/, consensus/, disk.json). Without them
both default to folders under the repository's data/. Every script also takes explicit paths on its command line.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FILAMENT_DATA_ROOT", REPO / "data" / "MAGFiLO_1.0_Kaggle_2026"))
TRAIN_IMAGES = DATA_ROOT / "train" / "train_images"
TRAIN_COCO = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TEST_IMAGES = DATA_ROOT / "test" / "test_images"
CACHE = Path(os.environ.get("FILAMENT_CACHE_DIR", REPO / "data" / "cache"))

#: every GONG frame of the competition is FRAME x FRAME pixels; the submission's RLE strings rely on it
FRAME = 2048
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_files(folder: str | Path) -> list[Path]:
    """The frames in `folder` (.jpg, .jpeg or .png in any letter case), sorted by path."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def image_stems(folder: str | Path) -> list[str]:
    return sorted(p.stem for p in image_files(folder))
