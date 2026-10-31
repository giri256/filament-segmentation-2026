"""Panoptic Quality as the competition pools it.

A prediction matches a ground-truth filament when their IoU exceeds 0.5 (such a match is unique when the predictions
do not overlap). A frame's predictions are matched against each of its annotated views separately, and TP, FP, FN
and the matched-IoU sum are pooled over all views of all frames:

    PQ = sum of matched IoUs / (TP + FP / 2 + FN / 2)

Per frame the six totals below are kept (summed over its views), so pooled PQ of any subset or bootstrap resample
of frames is exact.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .rle import iou

FIELDS = ("n_gt", "n_pred", "tp", "fp", "fn", "iou_sum")
IOU_MATCH = 0.5


def matched_ious(gt: Sequence[dict], pred: Sequence[dict], threshold: float = IOU_MATCH) -> list[float]:
    """IoUs of the matched pairs, largest first: pairs above `threshold` taken greedily in descending IoU, each
    filament and each prediction at most once."""
    m = iou(gt, pred)
    if m.size == 0:
        return []
    g, p = np.nonzero(m > threshold)
    order = np.argsort(-m[g, p])
    out, used_g, used_p = [], set(), set()
    for k in order:
        a, b = int(g[k]), int(p[k])
        if a in used_g or b in used_p:
            continue
        used_g.add(a)
        used_p.add(b)
        out.append(float(m[a, b]))
    return out


def frame_totals(view_ids: Sequence[str], gt: dict[str, list[dict]], pred: list[dict]) -> np.ndarray:
    """One frame's FIELDS totals: its predictions against each of its views, summed."""
    out = np.zeros(len(FIELDS), dtype=np.float64)
    for vid in view_ids:
        hits = matched_ious(gt[vid], pred)
        tp = len(hits)
        out += (len(gt[vid]), len(pred), tp, len(pred) - tp, len(gt[vid]) - tp, float(sum(hits)))
    return out


def pooled_pq(totals: np.ndarray) -> np.ndarray:
    """PQ of summed totals (last axis = FIELDS); 0 where there is nothing to count."""
    tp, fp, fn, iou_sum = (totals[..., FIELDS.index(k)] for k in ("tp", "fp", "fn", "iou_sum"))
    den = tp + 0.5 * fp + 0.5 * fn
    return np.divide(iou_sum, den, out=np.zeros_like(iou_sum), where=den > 0)
