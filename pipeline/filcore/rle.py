"""COCO run-length encoding of masks, pairwise IoU, and the submission CSV.

pycocotools reads masks column-major, so masks are handed over as Fortran-ordered uint8 arrays; it returns the counts
as bytes, kept here as ASCII text. A submission row is `<stem>_<n>,<counts>` with n counting from 1 within a frame;
the frame size (2048 x 2048) is implied and not written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pycocotools import mask as coco_mask

from .paths import FRAME

CSV_HEADER = "filament_id,segmentation_rle"
#: the counts alphabet is ASCII 48..111, which holds none of these; checked anyway, since any of them would force
#: CSV quoting and corrupt the row
CSV_UNSAFE = frozenset(',"\'\r\n')


def encode(mask: np.ndarray) -> dict:
    """A boolean or 0/1 mask as a COCO RLE dict {"size": [h, w], "counts": bytes}."""
    return coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))


def decode(rle: dict) -> np.ndarray:
    return coco_mask.decode(rle).astype(bool)


def text(rle: dict) -> str:
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def from_text(counts: str, height: int = FRAME, width: int = FRAME) -> dict:
    return {"size": [height, width], "counts": counts.encode("ascii")}


def text_to_mask(counts: str, height: int = FRAME, width: int = FRAME) -> np.ndarray:
    return decode(from_text(counts, height, width))


def mask_to_text(mask: np.ndarray) -> str:
    return text(encode(mask))


def polygons_to_rle(polygons: Sequence[Sequence[float]], height: int, width: int) -> dict:
    """A COCO `segmentation` (a list of flat x, y polygons) rasterised as one RLE, the union of its parts."""
    parts = coco_mask.frPyObjects([[float(v) for v in poly] for poly in polygons], height, width)
    return coco_mask.merge(parts)


def label_rles(labels: np.ndarray) -> list[dict]:
    """One RLE per instance of an integer label map (0 = background), in increasing label order."""
    return [encode(labels == k) for k in np.unique(labels) if k != 0]


def iou(gt: Sequence[dict], pred: Sequence[dict]) -> np.ndarray:
    """IoU of every ground-truth / prediction pair, shape (len(gt), len(pred))."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    return np.asarray(coco_mask.iou(list(gt), list(pred), [0] * len(pred)), dtype=np.float64)


def write_csv(predictions: dict[str, Iterable[dict]], path: str | Path) -> Path:
    """The submission: frames in sorted order, each frame's masks numbered from 1 in list order (frames without a
    mask add no row)."""
    rows, seen = [CSV_HEADER], set()
    for stem in sorted(predictions):
        for n, rle in enumerate(predictions[stem], start=1):
            counts = text(rle)
            if CSV_UNSAFE.intersection(counts):
                raise ValueError(f"RLE counts of {stem}_{n} hold CSV-unsafe characters")
            ident = f"{stem}_{n}"
            if ident in seen:
                raise ValueError(f"duplicate filament_id {ident}")
            seen.add(ident)
            rows.append(f"{ident},{counts}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="") as fh:
        fh.write("\n".join(rows) + "\n")
    return path


def read_csv(path: str | Path) -> dict[str, list[dict]]:
    """A submission CSV back as {stem: [rle, ...]}."""
    out: dict[str, list[dict]] = {}
    with open(path, encoding="ascii") as fh:
        if next(fh).strip() != CSV_HEADER:
            raise ValueError(f"{path}: not a submission file")
        for line in fh:
            if line.strip():
                ident, counts = line.rstrip("\r\n").split(",", 1)
                out.setdefault(ident.rsplit("_", 1)[0], []).append(from_text(counts))
    return out
