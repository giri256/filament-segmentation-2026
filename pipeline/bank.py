"""Candidate bank: a 15-level component tree per observation, node descriptors, per-view IoU targets.

The tree: the dense probability map p is thresholded at LEVELS, each level is
closed (radius 3), hole-filled and cut to the disk (filcore.morph's
map_foreground steps), levels are forced to nest, and components of at least
MIN_AREA pixels become nodes; a node's parent is the component one level down
that contains it. Selections are antichains, so masks never overlap and IoU >
0.5 matches are unique: any selection is scored exactly from the per-node,
per-view IoUs stored here.

Two modes, with OOF/test parity in mind:
* oof:  observation in fold k; p, s (D4 spread) and d (|U-Net - UNet++|) come
        from fold k's models (averaged over the model tags), detections from
        fold k's detectors (merged across detectors, filcore.fusion.merge_lists).
* test: the tree is built on p averaged over all folds and tags, but every
        evidence descriptor (probability, spread, disagreement, detector) is
        computed separately with each fold's maps and detections on the same
        node masks and then averaged, so it is distributed like the
        single-fold OOF descriptors (filseg's E15 lesson).

    FILAMENT_DATA_ROOT=... python pipeline/bank.py --mode oof --maps DIR --dets DIR \
        --tags r34soft,ppsoft --detectors s1024,m1280 --out DIR
Maps: DIR/{oof|test}_<tag>_f<k>/<stem>.npz (p, s as uint8); detections:
DIR/dets_<det>_f<k>_{oof|test}.json.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one thread per worker process; parallelism comes from --workers

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402
from scipy import ndimage  # noqa: E402

from filcore import paths  # noqa: E402
from filcore.coco import Coco  # noqa: E402
from filcore.fusion import Detection, claim_pixels, claims, merge_lists  # noqa: E402
from filcore.limb import flat_field, load_disks, load_gray  # noqa: E402
from filcore.morph import close, fill_holes  # noqa: E402
from filcore.rle import from_text, text_to_mask  # noqa: E402

LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80)
MIN_AREA = 50
CLOSE = 3
DET_KEEP = 0.25
SITES = "BCLMTU"
#: detector-carved pieces: the fusion's claiming rule (detections in score order claim the level's
#: pixels inside their mask grown by PIECE_GROW) at these levels, kept as alternative decompositions
PIECE_GROW = 4
PIECE_MIN_SCORE = 0.10
#: a separate detector (--aux-detector, e.g. RF-DETR from rfdetr_det.py) enters as rf_* descriptors of its own and is
#: never merged with the YOLO lists: merge_lists' mean counts a source's miss as 0, so a third, differently calibrated
#: source would move every merged score that DET_KEEP, the carving and the det_* descriptors were set on
RF_KEEP = 0.30
#: its lists come without NMS (a DETR predictor emits overlapping query masks: on the fold-0 pilot 57 % of the
#: detections overlap a higher-scoring one at mask IoU > 0.6, nearly all below score 0.2), so they are thinned the
#: way the YOLO lists are (ultralytics NMS at IoU 0.6) before any rf_* descriptor is read
RF_NMS = 0.6
AUX_NAMES = ["rf_best_iou", "rf_best_score", "rf_best_spill", "rf_precision", "rf_cover", "rf_n_contained",
             "rf_max_touch_score", "rf_wscore"]
_CTX: dict = {}


def _init(ctx):
    global _CTX
    cv2.setNumThreads(1)
    _CTX = ctx


def fg_positions(fg0, idx):
    if fg0.size == 0 or idx.size == 0:
        return np.zeros(0, np.int64)
    pos = np.clip(np.searchsorted(fg0, idx), 0, len(fg0) - 1)
    return pos[fg0[pos] == idx]


def level_sets(p, dmask):
    """Nested closed/hole-filled level sets of p; labels of the level-0 foreground per level."""
    prev, fg0, lab, perim, n_lab = None, None, [], [], []
    for t in LEVELS:
        b = ((p > t) & dmask).astype(np.uint8)
        b = fill_holes(close(b, CLOSE)).astype(bool) & dmask
        if prev is not None:
            b &= prev
        b8 = b.astype(np.uint8)
        n, L = cv2.connectedComponents(b8, connectivity=8, ltype=cv2.CV_32S)
        if fg0 is None:
            fg0 = np.flatnonzero(b)
        lab.append(L.ravel()[fg0])
        edge = b & ~cv2.erode(b8, np.ones((3, 3), np.uint8)).astype(bool)
        perim.append(np.bincount(L[edge], minlength=n))
        n_lab.append(n)
        prev = b
    return fg0, lab, perim, n_lab


def build_tree(fg0, lab, perim, n_lab, w, disk):
    """Nodes (level, label, parent) and label-free geometric/tree descriptors."""
    ys = (fg0 // w).astype(np.float64)
    xs = (fg0 % w).astype(np.float64)
    rr = np.hypot(xs - disk.cx, ys - disk.cy) / disk.radius
    nodes, geo = [], []
    key_to_node = {}
    for k in range(len(LEVELS)):
        lk = lab[k]
        m = lk > 0
        ll = lk[m]
        n = n_lab[k]
        area = np.bincount(ll, minlength=n).astype(np.float64)
        keep = np.flatnonzero(area >= MIN_AREA)
        keep = keep[keep > 0]
        if keep.size == 0:
            continue
        sums = {name: np.bincount(ll, v[m], minlength=n) for name, v in
                (("y", ys), ("x", xs), ("yy", ys * ys), ("xx", xs * xs), ("xy", ys * xs), ("r", rr))}
        bb_y0 = ndimage.minimum(ys[m], ll, keep)
        bb_y1 = ndimage.maximum(ys[m], ll, keep)
        bb_x0 = ndimage.minimum(xs[m], ll, keep)
        bb_x1 = ndimage.maximum(xs[m], ll, keep)
        if k > 0:
            par = np.zeros(n, np.int64)
            par[ll] = lab[k - 1][m]
        if k + 1 < len(LEVELS):
            nl = lab[k + 1][m]
            mm = nl > 0
            n2 = n_lab[k + 1]
            child_area = np.bincount(nl[mm], minlength=n2).astype(np.float64)
            child_par = np.zeros(n2, np.int64)
            child_par[nl[mm]] = ll[mm]
            big = child_area >= MIN_AREA
            n_child = np.bincount(child_par[big], minlength=n)
            sum_child = np.bincount(child_par[big], child_area[big], minlength=n)
            max_child = np.zeros(n)
            np.maximum.at(max_child, child_par[big], child_area[big])
        else:
            n_child = sum_child = max_child = np.zeros(n)
        for j, lbl in enumerate(keep):
            a = area[lbl]
            my, mx = sums["y"][lbl] / a, sums["x"][lbl] / a
            vyy = sums["yy"][lbl] / a - my * my
            vxx = sums["xx"][lbl] / a - mx * mx
            vxy = sums["xy"][lbl] / a - my * mx
            tr, det = vyy + vxx, vyy * vxx - vxy * vxy
            disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
            ev1, ev2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
            major = 4 * np.sqrt(max(ev1, 1e-6))
            bbox = (bb_y1[j] - bb_y0[j] + 1) * (bb_x1[j] - bb_x0[j] + 1)
            key_to_node[(k, int(lbl))] = len(nodes)
            nodes.append((k, int(lbl), (k - 1, int(par[lbl])) if k > 0 else None))
            geo.append({
                "level": k, "thr": LEVELS[k], "log_area": np.log(a), "perim_ratio": perim[k][lbl] / a,
                "elong": np.sqrt(ev1 / ev2), "major": major, "width": a / max(major, 1.0),
                "bbox_fill": a / bbox, "r_rel": sums["r"][lbl] / a, "lat_rel": (my - disk.cy) / disk.radius,
                "n_child": n_child[lbl], "child_frac": sum_child[lbl] / a, "max_child_frac": max_child[lbl] / a,
            })
    parent = np.array([key_to_node.get(pk, -1) if pk is not None else -1 for _, _, pk in nodes], np.int64)
    nn = len(nodes)
    child_count = np.bincount(parent[parent >= 0], minlength=nn) if nn else np.zeros(0, int)
    up, down = np.zeros(nn), np.zeros(nn)
    for i in range(nn):
        pi = parent[i]
        geo[i]["parent_area_ratio"] = float(np.exp(geo[i]["log_area"] - geo[pi]["log_area"])) if pi >= 0 else 1.0
        geo[i]["n_sib"] = float(child_count[pi] - 1) if pi >= 0 else 0.0
        up[i] = up[pi] + 1 if pi >= 0 and child_count[pi] == 1 else 0
    for i in range(nn - 1, -1, -1):
        pi = parent[i]
        if pi >= 0 and child_count[pi] == 1:
            down[pi] = down[i] + 1
    for i in range(nn):
        geo[i]["chain_up"], geo[i]["chain_down"] = up[i], down[i]
    return nodes, parent, geo


def decode_dets(dets, fg0, h, w):
    pos, area, score, smax, nhit, gs = [], [], [], [], [], []
    covered = np.zeros(len(fg0), bool)
    for d in dets:
        idx = np.flatnonzero(text_to_mask(d["counts"], h, w))
        if idx.size == 0:
            continue
        q = fg_positions(fg0, idx)
        pos.append(q)
        area.append(idx.size)
        score.append(d["score"])
        smax.append(d.get("score_max", d["score"]))
        nhit.append(d.get("n_hit", 1))
        gs.append(d.get("gscores", []))
        if d["score"] >= DET_KEEP:
            covered[q] = True
    return (pos, np.array(area, float), np.array(score, float), covered, np.array(smax, float), np.array(nhit, float),
            gs)


_DISC = {r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)) for r in (1, 2, 3, 5)}


def _ring_maps(Lk):
    """Ring/border pixel sets of one label image (labels < 65536): each background pixel within
    3 (resp. 1-5) px of a candidate is assigned to the largest neighbouring label; the inner border is
    the candidate's pixels within 2 px of its edge."""
    L16 = Lk.astype(np.uint16)
    fg = L16 > 0
    d1 = cv2.dilate(L16, _DISC[1])
    d3 = cv2.dilate(L16, _DISC[3])
    d5 = cv2.dilate(L16, _DISC[5])
    er = cv2.erode(fg.astype(np.uint8), _DISC[2]).astype(bool)
    r3 = ~fg & (d3 > 0)
    r5 = ~fg & (d5 > 0) & (d1 == 0)
    bd = fg & ~er
    return fg.ravel(), r3.ravel(), d3.ravel(), r5.ravel(), d5.ravel(), bd.ravel()


def ring_features(items, all_lab, all_n, fg0, h, w, map_sets):
    """Boundary/ring descriptors per candidate, for each dict of full-resolution maps in map_sets:
    "p" gives p_ring3 (mean p in the 1-3 px outer ring), p_edge_drop (inner 2 px border minus ring)
    and p_core_frac (share above 0.5); the spine map "q" (a thin line, ~0 off the spine) gives q_in,
    q_out_mass (spine pixels, q > 0.3, in the outer ring: the spine continues past a truncated
    fragment) and q_out_max (max q in the 1-3 px ring); any other map X gives X_in, X_ring_diff
    (1-5 px ring minus inside), X_ring_z (that over the ring's std) and X_edge_diff (ring3 minus border)."""
    outs = [[dict() for _ in items] for _ in map_sets]
    by_k = {}
    for i, (k, lbl, _) in enumerate(items):
        by_k.setdefault(k, []).append((i, lbl))
    for k, its in by_k.items():
        n = all_n[k]
        L = np.zeros(h * w, np.int32)
        L[fg0] = all_lab[k]
        fg, r3, d3, r5, d5, bd = _ring_maps(L.reshape(h, w))
        lin, l3, l5, lbd = L[fg], d3[r3].astype(np.int64), d5[r5].astype(np.int64), L[bd]
        c_in = np.maximum(np.bincount(lin, minlength=n), 1)
        c3 = np.maximum(np.bincount(l3, minlength=n), 1)
        c5 = np.maximum(np.bincount(l5, minlength=n), 1)
        cb = np.maximum(np.bincount(lbd, minlength=n), 1)
        for out, maps in zip(outs, map_sets):
            for name, v in maps.items():
                vf = v.ravel()
                m_in = np.bincount(lin, vf[fg], minlength=n) / c_in
                m3 = np.bincount(l3, vf[r3], minlength=n) / c3
                mb = np.bincount(lbd, vf[bd], minlength=n) / cb
                if name == "p":
                    core = np.bincount(lin, (vf[fg] > 0.5).astype(np.float64), minlength=n) / c_in
                    for i, lbl in its:
                        out[i].update({"p_ring3": m3[lbl], "p_edge_drop": mb[lbl] - m3[lbl], "p_core_frac": core[lbl]})
                elif name == "q":
                    o_mass = np.bincount(l5, (vf[r5] > 0.3).astype(np.float64), minlength=n)
                    o_max = np.zeros(n)
                    np.maximum.at(o_max, l3, vf[r3])
                    for i, lbl in its:
                        out[i].update({"q_in": m_in[lbl], "q_out_mass": o_mass[lbl], "q_out_max": o_max[lbl]})
                else:
                    m5 = np.bincount(l5, vf[r5], minlength=n) / c5
                    sd5 = np.sqrt(np.maximum(np.bincount(l5, vf[r5] ** 2, minlength=n) / c5 - m5 ** 2, 0.0))
                    for i, lbl in its:
                        out[i].update({f"{name}_in": m_in[lbl], f"{name}_ring_diff": m5[lbl] - m_in[lbl],
                                       f"{name}_ring_z": (m5[lbl] - m_in[lbl]) / (sd5[lbl] + 1e-3),
                                       f"{name}_edge_diff": m3[lbl] - mb[lbl]})
    return outs


_K3 = np.ones((3, 3), np.uint8)


def ridge_features(items, all_lab, all_n, fg0, h, w, p_map, s_map, q_map=None):
    """Medial-ridge descriptors per candidate from its distance transform: ridge length (px), mean width,
    width CV, and mean p and D4 spread s along the ridge (curved ribbons, junction merges, leaky halos);
    with a spine map q also q_ridge (mean q along the ridge), q_ridge_cov (share of the ridge with q > 0.35) and
    q_n_spines (separate q > 0.35 components of >= 15 px inside the candidate: a merge of several filaments)."""
    out = [None] * len(items)
    pf, sf = p_map.ravel(), s_map.ravel()
    qf = q_map.ravel() if q_map is not None else None
    by_k = {}
    for i, (k, lbl, _) in enumerate(items):
        by_k.setdefault(k, []).append((i, lbl))
    for k, its in by_k.items():
        lk = all_lab[k]
        order = np.argsort(lk, kind="stable")
        starts = np.concatenate([[0], np.cumsum(np.bincount(lk, minlength=all_n[k]))])
        for i, lbl in its:
            pos = fg0[order[starts[lbl]:starts[lbl + 1]]]
            ys, xs = pos // w, pos % w
            y0, x0 = int(ys.min()) - 1, int(xs.min()) - 1
            crop = np.zeros((int(ys.max()) - y0 + 2, int(xs.max()) - x0 + 2), np.uint8)
            crop[ys - y0, xs - x0] = 1
            edt = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
            ry, rx = np.nonzero((crop > 0) & (edt >= cv2.dilate(edt, _K3) - 0.4))
            ev = edt[ry, rx]
            g = (ry + y0) * w + (rx + x0)
            out[i] = {"ridge_len": float(ry.size), "ridge_width": float(2 * ev.mean()),
                      "ridge_wcv": float(ev.std() / (ev.mean() + 1e-4)), "p_ridge": float(pf[g].mean()),
                      "s_ridge": float(sf[g].mean())}
            if qf is not None:
                qr = qf[g]
                qc = np.zeros_like(crop)
                qc[ys - y0, xs - x0] = qf[pos] > 0.35
                nc, _, st, _ = cv2.connectedComponentsWithStats(qc, connectivity=8)
                out[i].update({"q_ridge": float(qr.mean()), "q_ridge_cov": float((qr > 0.35).mean()),
                               "q_n_spines": float((st[1:, cv2.CC_STAT_AREA] >= 15).sum())})
    return out


def evidence(nodes, lab, n_lab, fg0, pv, sv, dv, dets, h, w, n_groups=1):
    """Per-node evidence descriptors from one set of maps (values on fg0) and one detection list.
    With several detectors (n_groups > 1), det_g<j> is detector j's own score for the best-IoU detection."""
    det_pos, det_area, det_score, covered, det_smax, det_nhit, det_gs = decode_dets(dets, fg0, h, w)
    out = [None] * len(nodes)
    by_level = {}
    for i, (k, lbl, _) in enumerate(nodes):
        by_level.setdefault(k, []).append((i, lbl))
    for k, items in by_level.items():
        lk = lab[k]
        m = lk > 0
        ll = lk[m]
        n = n_lab[k]
        idx = np.array([lbl for _, lbl in items])
        area = np.bincount(ll, minlength=n).astype(np.float64)
        sp = np.bincount(ll, pv[m], minlength=n)
        sp2 = np.bincount(ll, pv[m] ** 2, minlength=n)
        pmax = np.asarray(ndimage.maximum(pv[m], ll, idx))
        ss = np.bincount(ll, sv[m], minlength=n) if sv is not None else None
        sd = np.bincount(ll, dv[m], minlength=n) if dv is not None else None
        cov = np.bincount(ll, covered[m].astype(np.float64), minlength=n)
        inter_d = (np.stack([np.bincount(lk[q][lk[q] > 0], minlength=n) for q in det_pos]).astype(np.float64)
                   if det_pos else np.zeros((0, n)))
        for j, (i, lbl) in enumerate(items):
            a = area[lbl]
            f = {"p_mean": sp[lbl] / a, "p_std": np.sqrt(max(sp2[lbl] / a - (sp[lbl] / a) ** 2, 0.0)),
                 "p_max": float(pmax[j]), "det_cover": cov[lbl] / a}
            if ss is not None:
                f["s_mean"] = ss[lbl] / a
            if sd is not None:
                f["d_mean"] = sd[lbl] / a
            if inter_d.shape[0]:
                col = inter_d[:, lbl]
                iou_d = col / (a + det_area - col)
                b = int(np.argmax(iou_d))
                contained = (col / det_area >= 0.5) & (det_score >= DET_KEEP)
                cs = np.sort(det_score[contained])[::-1]
                touch = col > 0
                f.update({"det_precision": col[b] / a, "det_split_gap": cov[lbl] / a - col[b] / a,
                          "det_best_smax": det_smax[b] if col[b] > 0 else 0.0,
                          "det_best_nhit": det_nhit[b] if col[b] > 0 else 0.0})
                f.update({"det_best_iou": iou_d[b], "det_best_score": det_score[b] if col[b] > 0 else 0.0,
                          "det_best_spill": 1 - col[b] / det_area[b] if col[b] > 0 else 1.0,
                          "det_n_contained": float(contained.sum()),
                          "det_second_contained": cs[1] if cs.size > 1 else 0.0,
                          "det_max_touch_score": det_score[touch].max() if touch.any() else 0.0,
                          "det_wscore": float((det_score * col).sum() / a)})
                if n_groups > 1:
                    g = det_gs[b] if col[b] > 0 and len(det_gs[b]) == n_groups else [0.0] * n_groups
                    f.update({f"det_g{q}": float(g[q]) for q in range(n_groups)})
            else:
                if n_groups > 1:
                    f.update({f"det_g{q}": 0.0 for q in range(n_groups)})
                f.update({"det_precision": 0.0, "det_split_gap": cov[lbl] / a, "det_best_smax": 0.0,
                          "det_best_nhit": 0.0})
                f.update({"det_best_iou": 0.0, "det_best_score": 0.0, "det_best_spill": 1.0, "det_n_contained": 0.0,
                          "det_second_contained": 0.0, "det_max_touch_score": 0.0, "det_wscore": 0.0})
            out[i] = f
    return out


def nms_dets(dets, iou_threshold=RF_NMS):
    """Greedy mask NMS: in descending score order (ties keep their order), drop a detection whose mask IoU with an
    already kept one exceeds iou_threshold."""
    dets = sorted(dets, key=lambda d: -d["score"])
    if len(dets) < 2:
        return dets
    rles = [from_text(d["counts"]) for d in dets]
    iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))
    kept = []
    for i in range(len(dets)):
        if not kept or iou[i, kept].max() <= iou_threshold:
            kept.append(i)
    return [dets[i] for i in kept]


def aux_evidence(nodes, lab, n_lab, fg0, dets, h, w):
    """Per-node rf_* descriptors from the separate detector's list, defined like their det_* namesakes (best-IoU
    detection's IoU, score, spill and precision; cover and containment by detections of at least RF_KEEP; best
    touching score; area-weighted score)."""
    pos, area, score = [], [], []
    covered = np.zeros(len(fg0), bool)
    for d in dets:
        idx = np.flatnonzero(text_to_mask(d["counts"], h, w))
        if idx.size == 0:
            continue
        q = fg_positions(fg0, idx)
        pos.append(q)
        area.append(idx.size)
        score.append(d["score"])
        if d["score"] >= RF_KEEP:
            covered[q] = True
    area, score = np.array(area, float), np.array(score, float)
    out = [None] * len(nodes)
    by_level = {}
    for i, (k, lbl, _) in enumerate(nodes):
        by_level.setdefault(k, []).append((i, lbl))
    for k, items in by_level.items():
        lk = lab[k]
        m = lk > 0
        ll = lk[m]
        n = n_lab[k]
        node_area = np.bincount(ll, minlength=n).astype(np.float64)
        cov = np.bincount(ll, covered[m].astype(np.float64), minlength=n)
        inter = (np.stack([np.bincount(lk[q][lk[q] > 0], minlength=n) for q in pos]).astype(np.float64)
                 if pos else np.zeros((0, n)))
        for i, lbl in items:
            a = node_area[lbl]
            col = inter[:, lbl]
            f = {"rf_cover": cov[lbl] / a}
            if col.size and col.max() > 0:
                iou = col / (a + area - col)
                b = int(np.argmax(iou))
                f.update({"rf_best_iou": iou[b], "rf_best_score": score[b], "rf_best_spill": 1 - col[b] / area[b],
                          "rf_precision": col[b] / a,
                          "rf_n_contained": float(((col / area >= 0.5) & (score >= RF_KEEP)).sum()),
                          "rf_max_touch_score": score[col > 0].max(), "rf_wscore": float((score * col).sum() / a)})
            else:
                f.update({"rf_best_iou": 0.0, "rf_best_score": 0.0, "rf_best_spill": 1.0, "rf_precision": 0.0,
                          "rf_n_contained": 0.0, "rf_max_touch_score": 0.0, "rf_wscore": 0.0})
            out[i] = f
    return out


def carve_pieces(fg0, lab, n_lab, nodes, dets, h, w, disk, piece_levels):
    """Detector-carved pieces per level: label arrays on fg0, piece rows, containers.

    Returns (lab_ext, n_ext, perim_ext, pieces) where pieces is a list of dicts with the
    lab_ext index, label, carving level index, claiming score/rank and the chain of tree
    nodes containing the whole piece (root first).
    """
    key_to_node = {(k, lbl): i for i, (k, lbl, _) in enumerate(nodes)}
    lab_ext, n_ext, perim_ext, pieces = [], [], [], []
    if not dets:
        return lab_ext, n_ext, perim_ext, pieces
    ordered = sorted(dets, key=lambda d: -d["score"])
    regions = claims([Detection(d["score"], d["counts"]) for d in ordered], PIECE_GROW, (h, w), PIECE_MIN_SCORE)
    for li, t in enumerate(piece_levels):
        k_lvl = LEVELS.index(t)
        flat = np.zeros(h * w, bool)
        flat[fg0[lab[k_lvl] > 0]] = True
        binary = flat.reshape(h, w)
        labels = claim_pixels(binary, regions, PIECE_MIN_SCORE, MIN_AREA)
        # claim_pixels() numbers kept regions in claim order; recover each label's score and rank
        claimed = np.zeros(binary.shape, bool)
        info = []
        for rank, region in enumerate(regions):
            window = (slice(region.y0, region.y1), slice(region.x0, region.x1))
            pix = binary[window] & region.mask & ~claimed[window]
            if int(pix.sum()) < MIN_AREA:
                continue
            claimed[window] |= pix
            info.append((region.score, rank))
        n = int(labels.max()) + 1
        assert len(info) == n - 1, (len(info), n)
        pl = labels.ravel()[fg0]
        b8 = (labels > 0).astype(np.uint8)
        edge = (labels > 0) & ~cv2.erode(b8, np.ones((3, 3), np.uint8)).astype(bool)
        # a piece's border also runs between touching pieces: count pixels whose label differs from a neighbour
        er_same = np.zeros_like(edge)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            er_same |= (labels != np.roll(np.roll(labels, dy, 0), dx, 1)) & (labels > 0)
        perim_ext.append(np.bincount(labels[edge | er_same], minlength=n))
        lab_ext.append(pl)
        n_ext.append(n)
        if n == 1:
            continue
        m = pl > 0
        ids = np.arange(1, n)
        chains = [[] for _ in ids]
        alive = np.ones(len(ids), bool)
        for kk in range(k_lvl + 1):
            if not alive.any():
                break
            lo = np.asarray(ndimage.minimum(lab[kk][m], pl[m], ids))
            hi = np.asarray(ndimage.maximum(lab[kk][m], pl[m], ids))
            single = alive & (lo == hi) & (lo > 0)
            for j in np.flatnonzero(single):
                node = key_to_node.get((kk, int(lo[j])))
                if node is not None:
                    chains[j].append(node)
            alive &= single
        for j, lbl in enumerate(ids):
            pieces.append({"ext": li, "label": int(lbl), "li": li, "k": k_lvl, "score": info[j][0],
                           "rank": info[j][1], "chain": chains[j]})
    return lab_ext, n_ext, perim_ext, pieces


def piece_geometry(pieces, lab_ext, n_ext, perim_ext, fg0, w, disk, geo):
    ys = (fg0 // w).astype(np.float64)
    xs = (fg0 % w).astype(np.float64)
    rr = np.hypot(xs - disk.cx, ys - disk.cy) / disk.radius
    stats = {}
    for li, pl in enumerate(lab_ext):
        m = pl > 0
        ll = pl[m]
        n = n_ext[li]
        if n == 1:
            continue
        sums = {name: np.bincount(ll, v[m], minlength=n) for name, v in
                (("a", np.ones_like(ys)), ("y", ys), ("x", xs), ("yy", ys * ys), ("xx", xs * xs), ("xy", ys * xs),
                 ("r", rr))}
        ids = np.arange(1, n)
        bb = [np.asarray(f(v[m], ll, ids)) for f, v in ((ndimage.minimum, ys), (ndimage.maximum, ys),
                                                         (ndimage.minimum, xs), (ndimage.maximum, xs))]
        stats[li] = (sums, bb)
    sib = {}
    for pc in pieces:
        key = (pc["li"], pc["chain"][-1] if pc["chain"] else -1)
        sib[key] = sib.get(key, 0) + 1
    out = []
    for pc in pieces:
        sums, bb = stats[pc["li"]]
        lbl = pc["label"]
        a = sums["a"][lbl]
        my, mx = sums["y"][lbl] / a, sums["x"][lbl] / a
        vyy = sums["yy"][lbl] / a - my * my
        vxx = sums["xx"][lbl] / a - mx * mx
        vxy = sums["xy"][lbl] / a - my * mx
        tr, det = vyy + vxx, vyy * vxx - vxy * vxy
        disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
        ev1, ev2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
        major = 4 * np.sqrt(max(ev1, 1e-6))
        j = lbl - 1
        bbox = (bb[1][j] - bb[0][j] + 1) * (bb[3][j] - bb[2][j] + 1)
        deep = pc["chain"][-1] if pc["chain"] else -1
        out.append({
            "level": pc["k"], "thr": LEVELS[pc["k"]], "log_area": np.log(a), "perim_ratio": perim_ext[pc["li"]][lbl] / a,
            "elong": np.sqrt(ev1 / ev2), "major": major, "width": a / max(major, 1.0), "bbox_fill": a / bbox,
            "r_rel": sums["r"][lbl] / a, "lat_rel": (my - disk.cy) / disk.radius,
            "n_child": np.nan, "child_frac": np.nan, "max_child_frac": np.nan,
            "parent_area_ratio": float(np.exp(np.log(a) - geo[deep]["log_area"])) if deep >= 0 else np.nan,
            "n_sib": float(sib[(pc["li"], deep)] - 1), "chain_up": float(len(pc["chain"])), "chain_down": np.nan,
            "is_piece": 1.0, "piece_score": pc["score"], "piece_rank": float(pc["rank"]),
        })
    return out


def load_maps(c, stem, fold):
    """(p, s, d, q) float32 maps of one fold's models for one observation; q is the spine map of
    c["spine_tag"]'s model (None without one)."""
    sub = "oof" if c["mode"] == "oof" else "test"
    ps, ss = [], []
    for tag in c["tags"]:
        z = np.load(Path(c["maps"]) / f"{sub}_{tag}_f{fold}" / f"{stem}.npz")
        ps.append(z["p"].astype(np.float32) / 255.0)
        ss.append(z["s"].astype(np.float32) / 1020.0)
    p = np.mean(ps, axis=0)
    s = np.mean(ss, axis=0)
    # mean pairwise |difference| between the dense models (= |p0 - p1| for two tags)
    d = np.mean([np.abs(ps[i] - ps[j]) for i in range(len(ps)) for j in range(i + 1, len(ps))], axis=0) \
        if len(ps) > 1 else None
    q = None
    if c.get("spine_tag"):
        z = np.load(Path(c["maps"]) / f"{sub}_{c['spine_tag']}_f{fold}" / f"{stem}.npz")
        q = z["q"].astype(np.float32) / 255.0
    return p, s, d, q


def build_stem(stem):
    t0 = time.time()
    c = _CTX
    disk = c["disks"][stem]
    folds = [c["fold_of"][stem]] if c["mode"] == "oof" else list(range(c["k"]))
    maps = {k: load_maps(c, stem, k) for k in folds}
    p_tree = np.mean([maps[k][0] for k in folds], axis=0)
    h, w = p_tree.shape
    dmask = disk.mask(p_tree.shape)
    fg0, lab, perim, n_lab = level_sets(p_tree, dmask)
    nodes, parent, geo = build_tree(fg0, lab, perim, n_lab, w, disk)
    for g in geo:
        g.update({"is_piece": 0.0, "piece_score": np.nan, "piece_rank": np.nan})
    lab_ext, n_ext, perim_ext, pieces = carve_pieces(fg0, lab, n_lab, nodes, c["carve"].get(stem, []), h, w, disk,
                                                     c["piece_levels"])
    geo = geo + piece_geometry(pieces, lab_ext, n_ext, perim_ext, fg0, w, disk, geo)
    nl = len(LEVELS)
    all_lab, all_n = lab + lab_ext, n_lab + n_ext
    items = [(k, lbl, None) for k, lbl, _ in nodes] + [(nl + pc["li"], pc["label"], None) for pc in pieces]
    s_avg = np.mean([maps[k][1] for k in folds], axis=0)
    q_avg = np.mean([maps[k][3] for k in folds], axis=0) if c.get("spine_tag") else None
    for g, rf in zip(geo, ridge_features(items, all_lab, all_n, fg0, h, w, p_tree, s_avg, q_avg)):
        g.update(rf)
    image_path = next(Path(c["image_dir"]).glob(f"{stem}.*"))
    flat = flat_field(load_gray(image_path), disk)
    rings = ring_features(items, all_lab, all_n, fg0, h, w,
                          [{"p": maps[k][0], **({"I": flat} if j == 0 else {}),
                            **({"q": maps[k][3]} if maps[k][3] is not None else {})} for j, k in enumerate(folds)])
    ev_all = []
    for j, k in enumerate(folds):
        p, s, d, _ = maps[k]
        dets = c["dets"][k].get(stem, [])
        ev = evidence(items, all_lab, all_n, fg0, p.ravel()[fg0], s.ravel()[fg0],
                      None if d is None else d.ravel()[fg0], dets, h, w, c["n_groups"])
        aux = aux_evidence(items, all_lab, all_n, fg0, c["aux"][k].get(stem, []), h, w) if c["aux"] else None
        for i, e in enumerate(ev):
            e.update(rings[j][i])
            e.update({key: val for key, val in rings[0][i].items() if key.startswith("I_")})
            if aux is not None:
                e.update(aux[i])
        ev_all.append(ev)
    rows = []
    in_disk = p_tree[dmask]
    k40 = LEVELS.index(0.40)
    a40 = np.bincount(lab[k40][lab[k40] > 0]) if (lab[k40] > 0).any() else np.zeros(1)
    site = stem[14:16][:1]
    n_det = np.mean([sum(1 for dd in c["dets"][k].get(stem, []) if dd["score"] >= DET_KEEP) for k in folds])
    frame = {"f_n40": float((a40 >= 200).sum()), "f_area40": float((lab[k40] > 0).sum() / dmask.sum()),
             "f_pmean": float(in_disk.mean()), "f_radius": float(disk.radius), "f_ndet": float(n_det),
             "f_year": float(stem[:4]), "f_site": float(SITES.index(site)) if site in SITES else -1.0}
    for i in range(len(items)):
        r = dict(geo[i])
        for key in ev_all[0][i]:
            r[key] = float(np.mean([ev[i][key] for ev in ev_all]))
        r.update(frame)
        rows.append(r)
    names = c["names"]
    X = np.array([[r[n] for n in names] for r in rows], np.float32).reshape(len(rows), len(names))
    n_nodes = len(nodes)
    alt = [(n_nodes + j, -1) for j in range(len(pieces))] + \
        [(n_nodes + j, v) for j, pc in enumerate(pieces) for v in pc["chain"]]
    out = {"X": X, "names": np.array(names),
           "parent": np.concatenate([parent, np.full(len(pieces), -1, np.int64)]),
           "is_piece": np.array([0] * n_nodes + [1] * len(pieces), np.int8),
           "piece_level": np.array([-1] * n_nodes + [pc["li"] for pc in pieces], np.int64),
           "alt_pairs": np.array(alt, np.int64).reshape(-1, 2),
           "level": np.array([it[0] for it in items], np.int64), "label": np.array([it[1] for it in items], np.int32)}
    if c["mode"] == "oof":
        views = c["views"][stem]
        vi = np.zeros((len(items), len(views)), np.float32)
        n_gt = []
        node_level = out["level"]
        node_lbl = out["label"].astype(np.int64)
        node_area = np.round(np.exp(X[:, names.index("log_area")].astype(np.float64)))
        per_level = [(kl, np.flatnonzero(node_level == kl)) for kl in range(len(all_lab))]
        per_level = [(kl, sel) for kl, sel in per_level if sel.size]
        for j, v in enumerate(views):
            gts = c["gt"][v]
            n_gt.append(len(gts))
            for rle in gts:
                q = fg_positions(fg0, np.flatnonzero(mask_utils.decode(rle)))
                ga = float(mask_utils.area(rle))
                for kl, sel in per_level:
                    lk = all_lab[kl][q]
                    inter = np.bincount(lk[lk > 0], minlength=all_n[kl]).astype(np.float64)[node_lbl[sel]]
                    iou = inter / (node_area[sel] + ga - inter)
                    vi[sel, j] = np.maximum(vi[sel, j], iou)
        out["view_iou"] = vi
        out["n_gt"] = np.array(n_gt, np.int32)
    np.savez_compressed(Path(c["out"]) / f"{stem}.npz", **out)
    return stem, f"{len(nodes)}+{len(pieces)}", round(time.time() - t0, 1)


def feature_names(n_tags, n_groups=1, spine=False, aux=False):
    geo = ["level", "thr", "log_area", "perim_ratio", "elong", "major", "width", "bbox_fill", "r_rel", "lat_rel",
           "n_child", "child_frac", "max_child_frac", "parent_area_ratio", "n_sib", "chain_up", "chain_down",
           "is_piece", "piece_score", "piece_rank", "ridge_len", "ridge_width", "ridge_wcv", "p_ridge", "s_ridge"]
    ev = ["p_mean", "p_std", "p_max", "det_cover", "s_mean"] + (["d_mean"] if n_tags > 1 else []) + \
        ["det_best_iou", "det_best_score", "det_best_spill", "det_n_contained", "det_second_contained",
         "det_max_touch_score", "det_wscore", "det_precision", "det_split_gap", "det_best_smax", "det_best_nhit",
         "p_ring3", "p_edge_drop", "p_core_frac", "I_in", "I_ring_diff", "I_ring_z", "I_edge_diff"] + \
        ([f"det_g{q}" for q in range(n_groups)] if n_groups > 1 else []) + \
        (["q_ridge", "q_ridge_cov", "q_n_spines", "q_in", "q_out_mass", "q_out_max"] if spine else [])
    fr = ["f_n40", "f_area40", "f_pmean", "f_radius", "f_ndet", "f_year", "f_site"]
    return geo + ev + fr + (AUX_NAMES if aux else [])


def merge_dets(views, iou_threshold=0.5):
    """Several detection lists of one observation as one (filcore.fusion.merge_lists: mean score over the lists
    with a miss counting 0, plus each merged detection's best member score, hit count and per-list scores)."""
    return merge_lists(views, iou_threshold)


def merged_dets(det_root, detectors, fold, sub):
    """One fold's detectors merged per observation. A detector named with a trailing "F" (yolo_flips.py)
    holds four flipped copies per observation, each merged as a view of its own."""
    files = [(d, json.loads((Path(det_root) / f"dets_{d}_f{fold}_{sub}.json").read_text())) for d in detectors]
    stems = sorted(set().union(*(f for _, f in files)))

    def views(stem):
        out = []
        for d, f in files:
            if d.endswith("F"):
                out.extend(f.get(stem, [[], [], [], []]))
            else:
                out.append(f.get(stem, []))
        return out

    sizes = [4 if d.endswith("F") else 1 for d in detectors]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    out = {}
    for stem in stems:
        merged = merge_dets(views(stem))
        for m in merged:  # each detector's own (flip-averaged) score for this merged detection
            m["gscores"] = [float(np.mean(m["vscores"][a:b])) for a, b in zip(starts[:-1], starts[1:])]
        out[stem] = merged
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oof", "test"], required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tags", default="r34soft,ppsoft")
    ap.add_argument("--detectors", default="s1024,m1280")
    ap.add_argument("--aux-detector", default="", help="a detector kept out of the merge, as rf_* descriptors "
                    "(DIR/dets_<name>_f<k>_{oof|test}.json)")
    ap.add_argument("--folds-file", default=str(HERE / "cv5_folds.json"))
    ap.add_argument("--disks", required=True, help="disk.json from preprocessing (train + test stems)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stems", default="", help="JSON list of observations to build (default: all)")
    ap.add_argument("--spine-tag", default="", help="dense tag whose maps carry a spine map q (spine descriptors)")
    ap.add_argument("--piece-levels", default="0.25,0.40,0.55", help="carving levels; empty = no pieces")
    args = ap.parse_args()

    folds = json.loads(Path(args.folds_file).read_text())
    fold_of, k = folds["fold_of_stem"], folds["k"]
    tags = args.tags.split(",")
    detectors = args.detectors.split(",") if args.detectors else []
    sub = "oof" if args.mode == "oof" else "test"
    dets = {f: (merged_dets(args.dets, detectors, f, sub) if detectors else {}) for f in range(k)}
    disks = load_disks(args.disks)
    piece_levels = [float(x) for x in args.piece_levels.split(",") if x]
    if args.mode == "oof":
        carve = {s: dets[fold_of[s]].get(s, []) for s in fold_of}
    else:  # one carving for test: the folds' merged detections merged again across folds
        stems_t = sorted(set().union(*[set(dets[f]) for f in range(k)]))
        carve = {}
        for s in stems_t:
            carve[s] = merge_dets([dets[f].get(s, []) for f in range(k)])
    ctx = {"mode": args.mode, "maps": args.maps, "tags": tags, "k": k, "fold_of": fold_of, "dets": dets,
           "disks": disks, "out": args.out, "names": feature_names(len(tags), len(detectors), bool(args.spine_tag), bool(args.aux_detector)),
           "carve": carve,
           "spine_tag": args.spine_tag,
           "n_groups": len(detectors),
           "piece_levels": piece_levels,
           "image_dir": str(paths.TRAIN_IMAGES if args.mode == "oof" else paths.TEST_IMAGES)}
    if args.mode == "oof":
        coco = Coco()
        views = coco.by_stem()
        stems = sorted(views)
        ctx["views"] = views
        ctx["gt"] = coco.rles_of(i for v in views.values() for i in v)
    else:
        stems = paths.image_stems(paths.TEST_IMAGES)
    if args.stems:
        wanted = set(json.loads(Path(args.stems).read_text()))
        stems = [s for s in stems if s in wanted]
    if args.limit:
        stems = stems[: args.limit]
    # the separate detector's lists of the folds these observations need (oof: their own folds; test: all)
    need = sorted({fold_of[s] for s in stems}) if args.mode == "oof" else list(range(k))
    aux = {f: {s: nms_dets(d) for s, d in
               json.loads((Path(args.dets) / f"dets_{args.aux_detector}_f{f}_{sub}.json").read_text()).items()}
           for f in need} if args.aux_detector else {}
    ctx["aux"] = aux
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "oof":
        (out / "views.json").write_text(json.dumps({s: ctx["views"][s] for s in stems}))
    else:
        (out / "stems.json").write_text(json.dumps(stems))
    todo = [s for s in stems if not (out / f"{s}.npz").exists()]
    print(f"{args.mode}: {len(stems)} observations, {len(todo)} to build; tags {tags}; detectors {detectors}"
          f"{'; aux detector ' + args.aux_detector if aux else ''}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(ctx,)) as pool:
        for n, (stem, n_nodes, secs) in enumerate(pool.map(build_stem, todo), start=1):
            if n % 25 == 0 or n == len(todo) or n <= 2:
                print(f"  {n}/{len(todo)} {stem} nodes={n_nodes} {secs}s {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
