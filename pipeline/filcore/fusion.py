"""Detector instances: claim regions, claiming dense-map pixels, and merging several detectors' lists.

A detection is a confidence and a full-frame mask as RLE counts text. Its mask is coarse (predicted at a fraction of
the input resolution), so it is grown by a few pixels and used to claim the full-resolution foreground of the dense
map: detections in descending confidence each take the foreground pixels inside their grown mask that no earlier
detection took, which gives non-overlapping instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from .morph import disc
from .rle import from_text, text_to_mask


@dataclass(frozen=True)
class Detection:
    score: float
    counts: str


@dataclass(frozen=True)
class Claim:
    """A detection's grown mask, cut to its bounding box [y0, y1) x [x0, x1)."""
    score: float
    y0: int
    y1: int
    x0: int
    x1: int
    mask: np.ndarray


def claims(detections: Sequence[Detection], grow: int, shape: tuple[int, int], min_score: float = 0.0) -> list[Claim]:
    """Detections of at least `min_score`, best first (ties keep their input order), each mask dilated by
    disc(grow) inside its bounding box padded by `grow`."""
    height, width = shape
    element = disc(grow) if grow > 0 else None
    out = []
    for det in sorted(detections, key=lambda d: -d.score):
        if det.score < min_score:
            continue
        mask = text_to_mask(det.counts, height, width)
        ys, xs = np.nonzero(mask)
        if not ys.size:
            continue
        y0, y1 = max(int(ys.min()) - grow, 0), min(int(ys.max()) + grow + 1, height)
        x0, x1 = max(int(xs.min()) - grow, 0), min(int(xs.max()) + grow + 1, width)
        box = mask[y0:y1, x0:x1].astype(np.uint8)
        if element is not None:
            box = cv2.dilate(box, element)
        out.append(Claim(det.score, y0, y1, x0, x1, box.astype(bool)))
    return out


def claim_pixels(foreground: np.ndarray, regions: Sequence[Claim], min_score: float, min_area: int) -> np.ndarray:
    """Integer instance labels: claims in the given order (best first; the first one below `min_score` ends it)
    each take the foreground pixels in their region not taken yet; a claim of fewer than `min_area` pixels takes
    nothing. Labels count from 1 in claiming order."""
    labels = np.zeros(foreground.shape, dtype=np.int32)
    taken = np.zeros(foreground.shape, dtype=bool)
    n = 0
    for region in regions:
        if region.score < min_score:
            break
        win = (slice(region.y0, region.y1), slice(region.x0, region.x1))
        mine = foreground[win] & region.mask & ~taken[win]
        if int(mine.sum()) < min_area:
            continue
        n += 1
        labels[win][mine] = n
        taken[win] |= mine
    return labels


def merge_lists(lists: Sequence[Sequence[dict]], iou_threshold: float = 0.5) -> list[dict]:
    """Detections of one frame from several sources (detector folds, or flipped copies of the frame) as one list.

    All detections are pooled in descending score order. Each one not yet grouped opens a group and adds, in
    descending IoU with it, every ungrouped detection with IoU above `iou_threshold` from a source the group does
    not hold yet. A group keeps its opener's mask; its score is the mean over all sources (a source that missed it
    counts 0). Also kept: the best member score (`score_max`), how many detections it merged (`n_hit`, summed when
    the inputs are merged lists themselves) and the score per source (`vscores`). Input dicts carry "score" and
    "counts" (and optionally "score_max", "n_hit").
    """
    pool = sorted(((det, src) for src, dets in enumerate(lists) for det in dets), key=lambda ds: -ds[0]["score"])
    if not pool:
        return []
    rles = [from_text(det["counts"]) for det, _ in pool]
    overlap = np.asarray(coco_mask.iou(rles, rles, [0] * len(rles)))
    grouped = np.zeros(len(pool), dtype=bool)
    merged = []
    for i in range(len(pool)):
        if grouped[i]:
            continue
        members, sources = [i], {pool[i][1]}
        for j in np.argsort(-overlap[i], kind="stable"):
            if overlap[i, j] <= iou_threshold:
                break
            if grouped[j] or pool[j][1] in sources:
                continue
            members.append(j)
            sources.add(pool[j][1])
        grouped[members] = True
        per_source = [0.0] * len(lists)
        for m in members:
            per_source[pool[m][1]] = pool[m][0]["score"]
        # added left to right in member order: builtin sum() is compensated from Python 3.12 on and can differ in
        # the last bit, which moves a mean that sits on the 4-place rounding boundary of the detection JSONs
        total = pool[members[0]][0]["score"]
        for m in members[1:]:
            total += pool[m][0]["score"]
        merged.append({"score": total / len(lists),
                       "score_max": max(pool[m][0].get("score_max", pool[m][0]["score"]) for m in members),
                       "n_hit": sum(pool[m][0].get("n_hit", 1) for m in members),
                       "vscores": per_source, "counts": pool[i][0]["counts"]})
    return merged
