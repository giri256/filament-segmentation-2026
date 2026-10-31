"""The organisers' self-evaluation statistics (kaggle.com/code/azimahmadzadeh/self-evaluation-notebook), which the
rubric's quantitative part scores besides PQ.

Per annotator view, from the (n_gt, n_pred) IoU matrix:
  PQ        TP = pairs with IoU > 0.5; pooled sum IoU_TP / (TP + FP/2 + FN/2)
  IoU, Dice histograms over every GT x prediction pair with IoU > 0 (Dice = 2 IoU / (1 + IoU) for a pair),
            so a grazing touch counts as a low-IoU pair and missed filaments do not count at all
  degrees   per GT: how many predictions overlap it (1:0 missed, 1:n fragmented); per prediction: how many
            GTs it overlaps (0:1 false, n:1 merged)
Also (ours, not the notebook's): how many predicted instances consist of more than one 8-connected part
(add_masks). Annotators draw one connected region per filament (99.2% of 2,743 train instances), so a
multi-part prediction looks like a defect when the judges inspect masks.

Accumulators are plain dicts of numbers and numpy arrays, so worker results can be summed with add().
"""

from __future__ import annotations

import cv2
import numpy as np

BINS = np.linspace(0.0, 1.0, 51)
DEG = 6  # degrees >= DEG share the last bin


def empty():
    return {"tp": 0, "fp": 0, "fn": 0, "tp_iou": 0.0, "n_pairs": 0, "iou_sum": 0.0, "dice_sum": 0.0,
            "iou_hist": np.zeros(50, np.int64), "dice_hist": np.zeros(50, np.int64),
            "gt_deg": np.zeros(DEG + 1, np.int64), "pred_deg": np.zeros(DEG + 1, np.int64),
            "n_inst": 0, "multi_part": 0}


def add_view(acc, iou, n_gt, n_pred):
    """Add one annotator view; iou has shape (n_gt, n_pred) (any shape works when either is 0)."""
    m = np.asarray(iou, np.float64).reshape(n_gt, n_pred)
    hit = m > 0.5
    acc["tp"] += int(hit.sum())
    acc["tp_iou"] += float(m[hit].sum())
    acc["fp"] += int((hit.sum(axis=0) == 0).sum())
    acc["fn"] += int((hit.sum(axis=1) == 0).sum())
    pos = m[m > 0]
    dice = 2 * pos / (1 + pos)
    acc["n_pairs"] += int(pos.size)
    acc["iou_sum"] += float(pos.sum())
    acc["dice_sum"] += float(dice.sum())
    acc["iou_hist"] += np.histogram(pos, BINS)[0]
    acc["dice_hist"] += np.histogram(dice, BINS)[0]
    acc["gt_deg"] += np.bincount(np.minimum((m > 0).sum(axis=1), DEG), minlength=DEG + 1)
    acc["pred_deg"] += np.bincount(np.minimum((m > 0).sum(axis=0), DEG), minlength=DEG + 1)
    return acc


def add_masks(acc, masks):
    """Count predicted instances (boolean full-frame masks) and those with more than one 8-connected part."""
    for m in masks:
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
        acc["n_inst"] += 1
        acc["multi_part"] += int(cv2.connectedComponents(crop, connectivity=8)[0] - 1 > 1)
    return acc


def add(a, b):
    return {k: a[k] + b[k] for k in a}


def summary(acc):
    gd, pd = acc["gt_deg"], acc["pred_deg"]
    n = max(acc["n_pairs"], 1)
    return {"pq": acc["tp_iou"] / max(acc["tp"] + 0.5 * acc["fp"] + 0.5 * acc["fn"], 1e-9),
            "tp": acc["tp"], "fp": acc["fp"], "fn": acc["fn"], "n_pairs": acc["n_pairs"],
            "iou_mean": acc["iou_sum"] / n, "dice_mean": acc["dice_sum"] / n,
            "low_pairs_lt_0.1": int(acc["iou_hist"][:5].sum()),
            "n_inst": acc["n_inst"], "multi_part": acc["multi_part"],
            "gt_1to0": int(gd[0]), "gt_1to1": int(gd[1]), "gt_1toN": int(gd[2:].sum()),
            "pred_0to1": int(pd[0]), "pred_1to1": int(pd[1]), "pred_Nto1": int(pd[2:].sum()),
            "gt_degree": gd.tolist(), "pred_degree": pd.tolist(),
            "iou_hist": acc["iou_hist"].tolist(), "dice_hist": acc["dice_hist"].tolist()}


def line(name, s):
    return (f"{name}: org PQ {s['pq']:.5f} | pairs {s['n_pairs']} meanIoU {s['iou_mean']:.4f} meanDice {s['dice_mean']:.4f} "
            f"(IoU<0.1 pairs {s['low_pairs_lt_0.1']}) | GT 1:0 {s['gt_1to0']} 1:1 {s['gt_1to1']} 1:n {s['gt_1toN']} "
            f"| pred 0:1 {s['pred_0to1']} 1:1 {s['pred_1to1']} n:1 {s['pred_Nto1']} | multi-part {s['multi_part']}/{s['n_inst']}")
