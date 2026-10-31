"""Baseline for the scorer gate: the fixed detector-guided fusion on the cv5 out-of-fold maps.

Each OOF observation is fused (filcore.fusion, the fixed fusion of filament-seg by A. Pawar: threshold,
closing, detections in score order claim U-Net pixels inside their grown masks) with its own fold's merged
detections, at every setting of a small grid. The setting is cross-fitted over the cv5 folds: each fold is
scored with the setting that is best on the other folds' observations. The per-observation totals of that cross-fitted run are the
base the scorer's paired bootstrap compares against (scorer.py --base); the organisers' self-evaluation
statistics of the same run (orgstats.py) go to <out>.org.json.

    FILAMENT_DATA_ROOT=... python pipeline/oof_fusion.py --maps DIR --dets DIR --disks disk.json \
        --tags r34soft --detectors s1024,m1280 --out base_totals.json [--folds 0] [--workers 2]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bank as B  # noqa: E402
import orgstats as O  # noqa: E402
from filcore.coco import Coco  # noqa: E402
from filcore.fusion import Detection, claim_pixels, claims  # noqa: E402
from filcore.limb import load_disks  # noqa: E402
from filcore.morph import map_foreground  # noqa: E402
from filcore.pq import frame_totals, pooled_pq  # noqa: E402
from filcore.rle import iou, label_rles  # noqa: E402

#: det2 used threshold 0.40 / min score 0.25, filament-seg's fusion defaults 0.5 / 0.30; on the cv5 fold-0 maps grow 16 was
#: worst (-0.007 vs grow 4), so the grid spends its budget on thresholds instead
THRESHOLDS = (0.35, 0.40, 0.45, 0.50)
GROWS = (4, 8)
MIN_SCORES = (0.25, 0.30)
CLOSE, MIN_AREA = 3, 200
GRID = list(itertools.product(THRESHOLDS, GROWS, MIN_SCORES))
_C: dict = {}


def key(setting):
    return "thr%.2f_g%d_s%.2f" % setting


def _init(ctx):
    global _C
    cv2.setNumThreads(1)
    _C = ctx


def _inputs(stem):
    c = _C
    fold = c["fold_of"][stem]
    p = B.load_maps(c, stem, fold)[0]
    logits = np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
    disk = c["disks"][stem].mask(p.shape)
    dets = [Detection(d["score"], d["counts"]) for d in c["dets"][fold].get(stem, [])]
    views = c["views"][stem]
    return p, logits, disk, dets, views, {v: c["gt"][v] for v in views}


def org_stem(item):
    """Organisers' statistics of one observation at one setting."""
    stem, (thr, grow, ms) = item
    p, logits, disk, dets, views, gt = _inputs(stem)
    labels = claim_pixels(map_foreground(logits, disk, thr, CLOSE), claims(dets, grow, p.shape, min(MIN_SCORES)), ms,
                          MIN_AREA)
    rles = label_rles(labels)
    acc = O.empty()
    for v in views:
        O.add_view(acc, iou(gt[v], rles), len(gt[v]), len(rles))
    O.add_masks(acc, [labels == k for k in np.unique(labels) if k > 0])
    return acc


def fuse_stem(stem):
    p, logits, disk, dets, views, gt = _inputs(stem)
    out = {}
    for thr in THRESHOLDS:
        binary = map_foreground(logits, disk, thr, CLOSE)
        for grow in GROWS:
            regions = claims(dets, grow, p.shape, min(MIN_SCORES))
            for ms in MIN_SCORES:
                labels = claim_pixels(binary, regions, ms, MIN_AREA)
                out[key((thr, grow, ms))] = frame_totals(views, gt, label_rles(labels)).tolist()
    return stem, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tags", default="r34soft")
    ap.add_argument("--detectors", default="s1024,m1280")
    ap.add_argument("--folds-file", default=str(HERE / "cv5_folds.json"))
    ap.add_argument("--folds", default="", help="comma list of folds to run (default: all)")
    ap.add_argument("--disks", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True, help="cross-fitted base totals JSON {stem: totals}")
    args = ap.parse_args()

    folds = json.loads(Path(args.folds_file).read_text())
    fold_of, k = folds["fold_of_stem"], folds["k"]
    run_folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(k))
    detectors = args.detectors.split(",")
    dets = {f: B.merged_dets(args.dets, detectors, f, "oof") for f in run_folds}
    coco = Coco()
    views = coco.by_stem()
    stems = sorted(s for s in views if fold_of[s] in run_folds)
    ctx = {"mode": "oof", "maps": args.maps, "tags": args.tags.split(","), "fold_of": fold_of, "dets": dets,
           "disks": load_disks(args.disks),
           "views": views, "gt": coco.rles_of(i for s in stems for i in views[s])}
    print(f"{len(stems)} obs in folds {run_folds}; tags {ctx['tags']}; detectors {detectors}; {len(GRID)} settings",
          flush=True)
    t0 = time.time()
    per = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(ctx,)) as pool:
        for n, (stem, out) in enumerate(pool.map(fuse_stem, stems, chunksize=4), start=1):
            per[stem] = out
            if n % 50 == 0 or n == len(stems):
                print(f"  {n}/{len(stems)} {time.time() - t0:.0f}s", flush=True)

    keys = [key(s) for s in GRID]
    tot = {kk: {f: np.sum([per[s][kk] for s in stems if fold_of[s] == f], axis=0) for f in run_folds} for kk in keys}
    for kk in keys:
        print(f"  {kk}: pooled PQ {float(pooled_pq(sum(tot[kk].values()))):.5f}", flush=True)
    base, chosen = {}, {}
    for f in run_folds:
        others = [g for g in run_folds if g != f] or [f]
        best = max(keys, key=lambda kk: float(pooled_pq(sum(tot[kk][g] for g in others))))
        chosen[f] = best
        base.update({s: per[s][best] for s in stems if fold_of[s] == f})
    crossfit = float(pooled_pq(np.sum(list(base.values()), axis=0)))
    print(f"cross-fitted base PQ {crossfit:.5f} over {len(base)} obs; settings per fold {chosen}", flush=True)
    Path(args.out).write_text(json.dumps(base))
    Path(args.out).with_suffix(".grid.json").write_text(json.dumps(per))

    setting = {key(s): s for s in GRID}
    acc = O.empty()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(ctx,)) as pool:
        for a in pool.map(org_stem, [(s, setting[chosen[fold_of[s]]]) for s in stems], chunksize=4):
            acc = O.add(acc, a)
    summ = O.summary(acc)
    print(O.line("cross-fitted base", summ), flush=True)
    Path(args.out).with_suffix(".org.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
