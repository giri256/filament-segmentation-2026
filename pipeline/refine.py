"""Stage-3 crop refiner: re-draw each stage-1 candidate's mask with an instance-conditioned network.

crops: for every observation, the candidates the AND-OR DP keeps at tau_obs (scorer.py --exp dump's cross-fitted
       scores), plus with --tau-delta > 0 those kept at tau_obs - delta that do not overlap the selection (extra
       training examples), are cut out as rotated crops (v2, after filseg v4 §8): the seed's minimum-area rectangle with
       its long axis horizontal, padded 50%, at native 2048-px scale unless it does not fit (isotropic shrink, never
       enlarged), warped bilinearly into CROP_W x CROP_H. 5 channels: flat-field intensity, dense probability p, D4
       spread s, the candidate (seed) mask, and the other selected candidates.
       OOF target: the mean of the best-overlapping GT filament (IoU > 0.1) over the views that drew it, i.e. the
       filament's extent given that it exists (existence is stage 1's job; averaging over all views would scale a
       1-of-3 filament to 1/3 and the threshold would erode it). nhit = how many views drew it.
train: UNet++ R34 (--arch) on the crops with nhit > 0, BCE + 0.5 soft Dice, flip augmentation, cross-fitted over the same date folds
       as the dump; held-out crops predicted with 4-flip TTA (torch; GPU).
eval:  refined masks (threshold, components touching the seed, disk, overlaps resolved by score; a mask that loses
       more than half its area to higher-scoring ones is dropped, not left as a sliver) scored against GT with the
       RLE IoU, for the stage-1 selection with original vs refined masks; refj@t also joins a refined mask's parts
       with 3-px corridors (one region per instance); refjs@t is refj that keeps the seed when the crop was shrunk
       below MIN_SCALE; refjx@t (round 16) fixes the claim order: the refined mask is drawn on the whole disk, then
       loses the already-claimed pixels, so the half-area drop works (refj measures it after claiming and keeps
       slivers), and its parts are joined through unclaimed pixels only; refjxs@t also drops a mask that keeps under
       half its seed; refjxcNN@t also drops a mask whose mean refiner probability is under NN/100;
       <family>@cv picks the threshold on the other cv5 folds;
       also the organisers' self-evaluation statistics per variant (orgstats.py).

    python pipeline/refine.py crops --bank B --scores oof_scores.npz --maps M --dets D --tags t --detectors d --disks J --out C
    python pipeline/refine.py train --crops C --out P
    python pipeline/refine.py eval --bank B --scores oof_scores.npz --crops C --preds P --maps M --dets D ... --disks J

Test path: submit.py --scores-out writes the test selection as a score dump; `crops --mode test` cuts its crops;
`train --fit-all --models P` predicts them with the mean of the saved fold models; `apply --variant V` writes the CSV
with exactly eval's mask procedure (final_masks), so `apply --variant orig` reproduces submit.py's CSV.

    python pipeline/refine.py apply --mode test --bank T --scores test_scores.npz --crops C --preds P --variant refj@0.5 \
        --maps M --dets D --disks J --out submission.csv
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bank as B  # noqa: E402
import orgstats as O  # noqa: E402
import scorer as S  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402
from filcore import paths  # noqa: E402
from filcore.coco import Coco  # noqa: E402
from filcore.limb import flat_field, load_disks, load_gray  # noqa: E402
from filcore.rle import encode as mask_to_rle, write_csv  # noqa: E402

#: v2 crops (filseg v4 §8): the seed's minimum-area rectangle, long axis horizontal, padded 50%, native 2048-px scale
#: unless it does not fit (isotropic shrink, never enlarged); every channel and the target warped bilinearly
CROP_W, CROP_H, PAD = 384, 128, 1.5
#: refjs@t keeps the seed for crops shrunk below this scale (long thin filaments lose width when resampled)
MIN_SCALE = 0.65
_C: dict = {}


def _init(c):
    global _C
    cv2.setNumThreads(1)
    _C = c


def candidate_masks(c, stem):
    """Rebuild every candidate of one observation exactly as bank.py did: (fg0, all_lab, h, w, p, s, disk)."""
    disk = c["disks"][stem]
    folds = [c["fold_of"][stem]] if c["mode"] == "oof" else list(range(c["k"]))
    maps = [B.load_maps(c, stem, f) for f in folds]
    p = np.mean([m[0] for m in maps], axis=0)
    sp = np.mean([m[1] for m in maps], axis=0)
    h, w = p.shape
    fg0, lab, perim, n_lab = B.level_sets(p, disk.mask(p.shape))
    nodes, _, _ = B.build_tree(fg0, lab, perim, n_lab, w, disk)
    lab_ext, _, _, pieces = B.carve_pieces(fg0, lab, n_lab, nodes, c["carve"].get(stem, []), h, w, disk,
                                           c["piece_levels"])
    return fg0, lab + lab_ext, len(nodes) + len(pieces), h, w, p, sp, disk


def row_mask(fg0, all_lab, z, i, h, w):
    m = np.zeros(h * w, bool)
    m[fg0[all_lab[int(z["level"][i])] == int(z["label"][i])]] = True
    return m.reshape(h, w)


def crop_affine(mask):
    """2 x 3 affine from frame (x, y) to crop (u, v) for one seed mask."""
    ys, xs = np.nonzero(mask)
    (cx, cy), (wr, hr), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    if wr < hr:
        wr, hr, ang = hr, wr, ang + 90.0
    scale = min(1.0, CROP_W / max(PAD * wr, 1.0), CROP_H / max(PAD * hr, 1.0))
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(ang), scale)
    M[0, 2] += CROP_W / 2 - cx
    M[1, 2] += CROP_H / 2 - cy
    return M


def warp(img, M):
    """CROP_H x CROP_W crop of a full-frame float map (bilinear, zero outside the frame)."""
    return cv2.warpAffine(img.astype(np.float32), M, (CROP_W, CROP_H), flags=cv2.INTER_LINEAR, borderValue=0)


def paste(P, M, h, w):
    """Inverse warp of a crop-space map into the frame window it covers: (y0, x0, window array)."""
    corners = np.array([[0, 0, 1], [CROP_W, 0, 1], [0, CROP_H, 1], [CROP_W, CROP_H, 1]], float) @ \
        cv2.invertAffineTransform(M).T
    x0, y0 = max(int(np.floor(corners[:, 0].min())), 0), max(int(np.floor(corners[:, 1].min())), 0)
    x1, y1 = min(int(np.ceil(corners[:, 0].max())) + 1, w), min(int(np.ceil(corners[:, 1].max())) + 1, h)
    Ms = M.copy()
    Ms[:, 2] += M[:, :2] @ np.array([x0, y0], float)  # dst (window) pixel -> crop coordinates
    sub = cv2.warpAffine(P.astype(np.float32), Ms, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderValue=0)
    return y0, x0, sub


def join_parts(m, allowed):
    """Join the 8-connected parts of a mask with 3-px corridors along a minimum spanning tree of their shortest gaps
    (filseg v4 join_parts: neutral on PQ, one connected region per instance like every GT mask)."""
    n, L = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    from scipy import ndimage
    parts = list(range(1, n))
    near = {}
    for a in parts:  # nearest pixel of part a to every other part
        _, (iy, ix) = ndimage.distance_transform_edt(L != a, return_indices=True)
        for b in parts:
            if b == a:
                continue
            ys, xs = np.nonzero(L == b)
            d2 = (iy[ys, xs] - ys) ** 2 + (ix[ys, xs] - xs) ** 2
            k = int(np.argmin(d2))
            near[(a, b)] = (float(d2[k]), (int(xs[k]), int(ys[k])), (int(ix[ys[k], xs[k]]), int(iy[ys[k], xs[k]])))
    done, canvas = {parts[0]}, np.zeros(m.shape, np.uint8)
    while len(done) < len(parts):  # Prim
        d, a, b = min((near[(a, b)][0], a, b) for a in done for b in parts if b not in done)
        _, p1, p2 = near[(a, b)]
        cv2.line(canvas, p1, p2, 1, thickness=3)
        done.add(b)
    return m | (canvas.astype(bool) & allowed)


def crops_stem(stem):
    c = _C
    t0 = time.time()
    j = c["stems"].index(stem)
    z = np.load(Path(c["bank"]) / f"{stem}.npz")
    lo, hi = c["off"][j], c["off"][j + 1]
    sel_rows = np.flatnonzero(c["sel"][lo:hi])
    extra = np.flatnonzero(c["pre"][lo:hi] & ~c["sel"][lo:hi])
    if sel_rows.size + extra.size == 0:
        return stem, 0, 0.0
    fg0, all_lab, n_rows, h, w, p, sp, disk = candidate_masks(c, stem)
    assert n_rows == len(z["parent"]), (stem, n_rows, len(z["parent"]))
    flat = flat_field(load_gray(next(Path(c["image_dir"]).glob(f"{stem}.*"))), disk)
    # the "other instances" channel is the selection only, as at eval/test time; relaxed extras (training examples of
    # filaments the selection missed) are kept only if they do not overlap the selection
    union = np.zeros((h, w), bool)
    sel_masks = [row_mask(fg0, all_lab, z, i, h, w) for i in sel_rows]
    for m in sel_masks:
        union |= m
    rows, masks = list(sel_rows), sel_masks
    for i in extra:
        m = row_mask(fg0, all_lab, z, i, h, w)
        if not (m & union).any():
            rows.append(i)
            masks.append(m)
    gts = [c["gt"][v] for v in c["views"][stem]] if c["mode"] == "oof" else []
    base = [np.clip(flat / 1.5, 0, 1), p, np.clip(sp * 4, 0, 1)]
    X, Y, Ms, nhit = [], [], [], []
    for i, m in zip(rows, masks):
        M = crop_affine(m)
        seed = m.astype(np.float32)
        other = (union & ~m).astype(np.float32)
        chans = [warp(ch, M) for ch in (*base, seed, other)]
        X.append(np.stack([np.clip(np.round(ch * 255), 0, 255) for ch in chans]).astype(np.uint8))
        Ms.append(M)
        if gts:
            seed_rle = mask_to_rle(m.astype(np.uint8))
            tgt = np.zeros((h, w), np.float32)
            hit = 0
            for g in gts:
                if not g:
                    continue
                iou = np.asarray(mask_utils.iou(g, [seed_rle], [0])).ravel()
                b = int(np.argmax(iou))
                if iou[b] > 0.1:
                    tgt += mask_utils.decode(g[b]).astype(np.float32)
                    hit += 1
            tgt /= max(hit, 1)
            nhit.append(hit)
            Y.append(np.clip(np.round(warp(tgt, M) * 255), 0, 255).astype(np.uint8))
    out = {"X": np.stack(X), "rows": np.array(rows, np.int64), "M": np.array(Ms, np.float64),
           "is_sel": np.array([r in set(sel_rows.tolist()) for r in rows])}
    if Y:
        out["Y"] = np.stack(Y)
        out["nhit"] = np.array(nhit, np.int64)
    np.savez_compressed(Path(c["out"]) / f"{stem}.npz", **out)
    return stem, len(rows), round(time.time() - t0, 1)


def context(args, scores):
    folds = json.loads(Path(args.folds_file).read_text())
    k = folds["k"]
    tags = args.tags.split(",")
    detectors = args.detectors.split(",") if args.detectors else []
    sub = "oof" if args.mode == "oof" else "test"
    dets = {f: (B.merged_dets(args.dets, detectors, f, sub) if detectors else {}) for f in range(k)}
    if args.mode == "oof":
        carve = {s: dets[folds["fold_of_stem"][s]].get(s, []) for s in folds["fold_of_stem"]}
    else:
        stems_t = sorted(set().union(*[set(dets[f]) for f in range(k)]))
        carve = {s: B.merge_dets([dets[f].get(s, []) for f in range(k)]) for s in stems_t}
    c = {"mode": args.mode, "maps": args.maps, "tags": tags, "k": k, "fold_of": folds["fold_of_stem"],
         "disks": load_disks(args.disks),
         "carve": carve, "piece_levels": [float(x) for x in args.piece_levels.split(",") if x],
         "image_dir": str(paths.TRAIN_IMAGES if args.mode == "oof" else paths.TEST_IMAGES), "bank": args.bank}
    if args.mode == "oof":
        views = json.loads((Path(args.bank) / "views.json").read_text())
        c["views"] = views
        c["gt"] = Coco().rles_of(i for v in views.values() for i in v)
    return c


def cmd_crops(args):
    d = np.load(args.scores, allow_pickle=False)
    F = S.Forest(Path(args.bank))
    assert list(d["stems"]) == F.stems
    score, tau_obs = d["score"], d["tau_obs"]
    pre = d["sel"].astype(bool).copy()
    for t in sorted(set(tau_obs.tolist())):
        m = tau_obs == t
        pre |= F.select(score, t - args.tau_delta, m) & m[F.stem_of]
    off = np.concatenate([[0], np.cumsum(np.bincount(F.stem_of, minlength=F.n_stems))])
    c = context(args, score)
    c.update({"stems": F.stems, "pre": pre, "sel": d["sel"].astype(bool), "off": off, "out": args.out})
    Path(args.out).mkdir(parents=True, exist_ok=True)
    todo = [s for s in F.stems if not (Path(args.out) / f"{s}.npz").exists()]
    print(f"crops: {int(pre.sum())} candidates ({int(d['sel'].sum())} selected) at tau and tau - {args.tau_delta} over {F.n_stems} obs; {len(todo)} obs to do",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(c,)) as pool:
        for n, (stem, nc, secs) in enumerate(pool.map(crops_stem, todo), start=1):
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {stem} {nc} crops {secs}s", flush=True)


# ---------------------------------------------------------------- training (torch)

def cmd_train(args):
    import torch
    import segmentation_models_pytorch as smp

    dev = torch.device("cuda")
    crops = sorted(Path(args.crops).glob("*.npz"))
    data = {p.stem: np.load(p) for p in crops}
    stems = sorted(data)
    folds = S.date_folds(stems, args.k, 0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    has_y = "Y" in data[stems[0]]

    def batches(X, Y, bs, train):
        idx = np.random.permutation(len(X)) if train else np.arange(len(X))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            x = X[b].astype(np.float32) / 255.0
            y = Y[b].astype(np.float32)[:, None] / 255.0 if Y is not None else None
            if train and args.seed_cut > 0:
                cut_seed_tips(x)
            if train:  # the four flips (a transpose is not valid on a non-square crop)
                if np.random.rand() < 0.5:
                    x, y = x[..., ::-1], y[..., ::-1]
                if np.random.rand() < 0.5:
                    x, y = x[..., ::-1, :], y[..., ::-1, :]
            yield b, torch.from_numpy(np.ascontiguousarray(x)).to(dev), \
                (torch.from_numpy(np.ascontiguousarray(y)).to(dev) if y is not None else None)

    def cut_seed_tips(x):
        """--seed-cut: on that share of the batch, erase 15-30 % of the seed's length at one end (crop columns;
        the long axis is horizontal) from the seed channel only, so the target still asks for the full extent."""
        for i in range(len(x)):
            if np.random.rand() >= args.seed_cut:
                continue
            cols = np.flatnonzero(x[i, 3].max(axis=0) > 0.5)
            if cols.size < 8:
                continue
            c0, c1 = int(cols[0]), int(cols[-1])
            n = max(1, int(round(np.random.uniform(0.15, 0.30) * (c1 - c0 + 1))))
            if np.random.rand() < 0.5:
                x[i, 3, :, c0:c0 + n] = 0
            else:
                x[i, 3, :, c1 - n + 1:c1 + 1] = 0

    def make(arch, encoder, weights):
        cls = smp.UnetPlusPlus if arch == "unetpp" else smp.Unet
        return cls(encoder, encoder_weights=weights, in_channels=5, classes=1)

    def loss_fn(logit, y):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        p = torch.sigmoid(logit).flatten(1)
        t = y.flatten(1)
        dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
        return bce + 0.5 * dice.mean()

    def train_model(train_stems, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        # only candidates some annotator drew: the refiner models extent, not existence
        X = np.concatenate([data[s]["X"][data[s]["nhit"] > 0] for s in train_stems])
        Y = np.concatenate([data[s]["Y"][data[s]["nhit"] > 0] for s in train_stems])
        model = make(args.arch, args.encoder, "imagenet").to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = args.epochs * ((len(X) + args.bs - 1) // args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda")
        bufs = list(model.buffers())
        ema = None
        if args.ema > 0:  # --ema: an exponential moving average of the weights (BatchNorm stats copied) is returned
            ema = copy.deepcopy(model)
            for q in ema.parameters():
                q.requires_grad_(False)
        for ep in range(args.epochs):
            model.train()
            tot, n, t0, bad = 0.0, 0, time.time(), 0
            for _, x, y in batches(X, Y, args.bs, True):
                saved = [b.clone() for b in bufs]
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = loss_fn(model(x), y)
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    # an fp16 overflow: undo its BatchNorm running-stat update and drop the batch (the schedule
                    # still advances); persistent overflow aborts instead of training on for hours
                    with torch.no_grad():
                        for b, s_ in zip(bufs, saved):
                            b.copy_(s_)
                    sched.step()
                    bad += 1
                    if bad > args.max_bad:
                        raise SystemExit(f"epoch {ep + 1}: more than {args.max_bad} non-finite batches; aborting")
                    continue
                scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if ema is not None:
                    with torch.no_grad():
                        for qe, qm in zip(ema.parameters(), model.parameters()):
                            qe.mul_(args.ema).add_(qm.detach(), alpha=1 - args.ema)
                        for be, bm in zip(ema.buffers(), model.buffers()):
                            be.copy_(bm)
                tot += loss.item() * len(x)
                n += len(x)
            print(f"    epoch {ep + 1}/{args.epochs} loss {tot / max(n, 1):.4f} ({time.time() - t0:.0f}s, {len(X)} crops)"
                  + (f" skipped {bad} non-finite batches" if bad else ""), flush=True)
        return ema if ema is not None else model

    @torch.no_grad()
    def predict(models, stems_):
        flips = (None, (-1,), (-2,), (-1, -2))
        for s in stems_:
            X = data[s]["X"]
            acc = np.zeros((len(X), CROP_H, CROP_W), np.float32)
            for model in models:
                model.eval()
                for flip in flips:
                    xx = X if flip is None else np.flip(X, axis=flip)
                    for b, x, _ in batches(xx, None, 32, False):
                        with torch.autocast("cuda", dtype=torch.float16):
                            pr = torch.sigmoid(model(x)).float().cpu().numpy()[:, 0]
                        acc[b] += pr if flip is None else np.flip(pr, axis=flip)
            acc /= len(flips) * len(models)
            if not np.isfinite(acc).all():
                raise SystemExit(f"non-finite refiner output for {s}: refusing to write it")
            np.savez_compressed(out / f"{s}.npz", P=np.round(acc * 255).astype(np.uint8))

    if has_y and not args.fit_all:
        for k, fi in enumerate(folds):
            held = [stems[i] for i in fi]
            train = [s for s in stems if s not in set(held)]
            print(f"fold {k}: train {len(train)} obs, predict {len(held)} obs", flush=True)
            model = train_model(train, k)
            predict([model], held)
            if args.save_models:
                torch.save(model.state_dict(), out / f"refiner_f{k}.pt")
    else:  # test crops: average the saved fold models
        models = []
        for k in range(args.k):
            m = make(args.arch, args.encoder, None).to(dev)
            sd = torch.load(Path(args.models) / f"refiner_f{k}.pt", map_location=dev)
            sd = sd["model"] if isinstance(sd.get("model"), dict) else sd  # export_fp16.py payload or a raw state dict
            m.load_state_dict({n: v.float() if v.is_floating_point() else v for n, v in sd.items()})
            models.append(m)
        predict(models, stems)


# ---------------------------------------------------------------- evaluation

def refined_mask(P, M, seed, h, w, dmask, thr, join=False):
    """Paste a crop prediction back: inverse warp into its frame window, threshold, keep the parts touching the seed
    (join=True joins them into one region); falls back to the seed if nothing survives."""
    y0, x0, sub = paste(P.astype(np.float32) / 255.0, M, h, w)
    win = (slice(y0, y0 + sub.shape[0]), slice(x0, x0 + sub.shape[1]))
    b = (sub > thr) & dmask[win]
    n, L = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)
    keep = np.unique(L[seed[win] & b])
    keep = keep[keep > 0]
    if not keep.size:
        return seed
    part = np.isin(L, keep)
    if join and keep.size > 1:
        part = join_parts(part, dmask[win])
    out = np.zeros((h, w), bool)
    out[win] = part
    return out


def refiner_inputs(c, stem):
    """One observation's crops (or None), the mean refiner prediction over the --preds members (0-255 scale) and
    the crop index of every bank row that has a crop."""
    cp = Path(c["crops"]) / f"{stem}.npz" if c["crops"] else None
    crop = np.load(cp) if cp is not None and cp.exists() else None
    # several refiner members (comma-separated --preds): mean of their probabilities, still on the 0-255 scale
    pred = None
    if crop is not None and c["preds"]:
        members = [np.load(Path(d) / f"{stem}.npz")["P"].astype(np.float32) for d in c["preds"].split(",")]
        if c.get("pred_weights"):  # --pred-weights: a weighted mean instead (weights normalised to sum 1)
            wts = np.asarray(c["pred_weights"], np.float32)
            pred = np.tensordot(wts / wts.sum(), np.stack(members), axes=1)
        else:
            pred = np.mean(members, axis=0)
        if c.get("anchor", 0.0) > 0:  # --anchor: blend in the warped stage-1 probability (crop channel 1)
            lam = c["anchor"]
            pred = (1 - lam) * pred + lam * crop["X"][:, 1].astype(np.float32)
    where = {int(r): q for q, r in enumerate(crop["rows"])} if crop is not None else {}
    return crop, pred, where


def _bbox(m, pad=2):
    ys, xs = np.nonzero(m)
    return (slice(max(ys.min() - pad, 0), ys.max() + pad + 1), slice(max(xs.min() - pad, 0), xs.max() + pad + 1))


def final_masks(c, v, z, sel, lo, fg0, all_lab, h, w, dmask, crop, pred, where):
    """The kept masks of one observation under variant v (orig, id@t, ref@t, refj@t, refjs@t, refjx@t, refjxs@t,
    refjxcNN@t): in score order, each (refined) mask loses the pixels already claimed and is dropped under MIN_AREA or
    under half its area."""
    claimed = np.zeros((h, w), bool)
    rles, kept = [], []
    for i in sorted(sel, key=lambda i: -c["score"][lo + i]):
        m = row_mask(fg0, all_lab, z, i, h, w)
        fam = v.split("@")[0]
        if fam.startswith("refjx") and i in where:
            q, thr = where[i], float(v.split("@")[1])
            seed_area = int(m.sum())
            m = refined_mask(pred[q], crop["M"][q], m, h, w, dmask, thr, join=False)  # drawn before any claim
            raw = int(m.sum())
            m = m & ~claimed
            if m.sum() < B.MIN_AREA or m.sum() < 0.5 * raw or (fam == "refjxs" and m.sum() < 0.5 * seed_area):
                continue
            win = _bbox(m)
            m[win] = join_parts(m[win], (dmask & ~claimed)[win])  # one region; corridors through unclaimed pixels
            if fam.startswith("refjxc"):
                y0, x0, sub = paste(pred[q].astype(np.float32) / 255.0, crop["M"][q], h, w)
                inside = m[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
                if inside.any() and sub[inside].mean() < int(fam[len("refjxc"):]) / 100:
                    continue
            claimed |= m
            kept.append(m)
            rles.append(mask_to_rle(m.astype(np.uint8)))
            continue
        if v != "orig" and not fam.startswith("refjx") and i in where:
            q = where[i]
            # id@t pastes back the crop's own seed channel: the crop -> paste-back round trip alone
            P = crop["X"][q, 3] if v.startswith("id@") else pred[q]
            if fam == "refjs" and np.hypot(crop["M"][q][0, 0], crop["M"][q][0, 1]) < MIN_SCALE:
                pass  # crop shrunk too much: resampling would lose the thin ribbon, keep the seed
            else:
                # unclaimed pixels only, so a later claim cannot re-split a joined mask
                m = refined_mask(P, crop["M"][q], m, h, w, dmask & ~claimed, float(v.split("@")[1]),
                                 join=fam in ("refj", "refjs"))
        raw = int(m.sum())
        m = m & ~claimed
        if m.sum() < B.MIN_AREA or m.sum() < 0.5 * raw:
            continue
        claimed |= m
        kept.append(m)
        rles.append(mask_to_rle(m.astype(np.uint8)))
    return kept, rles


def eval_stem(stem):
    c = _C
    j = c["stems"].index(stem)
    z = np.load(Path(c["bank"]) / f"{stem}.npz")
    lo, hi = c["off"][j], c["off"][j + 1]
    sel = np.flatnonzero(c["sel"][lo:hi])
    res, org = {}, {}
    gts = [c["gt"][v] or [] for v in c["views"][stem]]
    if sel.size == 0:  # nothing kept: every GT filament is a miss
        for v in c["variants"]:
            res[v] = [0.0, 0.5 * sum(len(g) for g in gts)]
            org[v] = O.empty()
            for g in gts:
                O.add_view(org[v], np.zeros((len(g), 0)), len(g), 0)
        return stem, res, org
    fg0, all_lab, _, h, w, _, _, disk = candidate_masks(c, stem)
    dmask = disk.mask((h, w))
    crop, pred, where = refiner_inputs(c, stem)
    for v in c["variants"]:
        kept, rles = final_masks(c, v, z, sel, lo, fg0, all_lab, h, w, dmask, crop, pred, where)
        num = 0.0
        org[v] = O.empty()
        for g in gts:
            iou = np.asarray(mask_utils.iou(g, rles, [0] * len(rles))).reshape(len(g), len(rles)) if g and rles \
                else np.zeros((len(g), len(rles)))
            num += float(iou[iou > 0.5].sum())
            O.add_view(org[v], iou, len(g), len(rles))
        den = 0.5 * (len(rles) * len(gts) + sum(len(g) for g in gts))
        res[v] = [num, den]
        O.add_masks(org[v], kept)
    return stem, res, org


def blend_options(args):
    """--pred-weights / --anchor as worker-context entries (defaults: plain member mean, no anchoring)."""
    wts = [float(x) for x in args.pred_weights.split(",")] if args.pred_weights else []
    if wts and len(wts) != len(args.preds.split(",")):
        raise SystemExit(f"--pred-weights has {len(wts)} values for {len(args.preds.split(','))} --preds members")
    return {"pred_weights": wts, "anchor": args.anchor}


def cmd_eval(args):
    d = np.load(args.scores)
    F = S.Forest(Path(args.bank))
    off = np.concatenate([[0], np.cumsum(np.bincount(F.stem_of, minlength=F.n_stems))])
    c = context(args, d["score"])
    families = args.families.split(",")
    variants = ["orig", "id@0.5"] + ([f"{k}@{t}" for k in families for t in args.thr.split(",")]
                                     if args.preds else [])
    c.update({"stems": F.stems, "off": off, "sel": d["sel"], "score": d["score"], "crops": args.crops,
              "preds": args.preds, "variants": variants, **blend_options(args)})
    tot = {v: np.zeros(2) for v in variants}
    per = {v: [] for v in variants}
    org = {v: O.empty() for v in variants}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(c,)) as pool:
        for stem, res, o in pool.map(eval_stem, F.stems):
            for v in variants:
                tot[v] += res[v]
                per[v].append(res[v])
                org[v] = O.add(org[v], o[v])
    fold = np.array([c["fold_of"][s] for s in F.stems])
    for fam in families:  # threshold chosen on the other cv5 folds, per family
        refs = [v for v in variants if v.startswith(f"{fam}@") and not v.endswith("@cv")]
        if not refs:
            continue
        arr = {v: np.array(per[v]) for v in refs}
        cv, picks = np.zeros((len(F.stems), 2)), {}
        for f in sorted(set(fold.tolist())):
            tr = fold != f
            best = max(refs, key=lambda v: arr[v][tr, 0].sum() / arr[v][tr, 1].sum())
            cv[fold == f] = arr[best][fold == f]
            picks[f] = best
        print(f"{fam}@cv thresholds per fold {picks}", flush=True)
        variants = variants + [f"{fam}@cv"]
        per[f"{fam}@cv"], tot[f"{fam}@cv"] = cv.tolist(), cv.sum(0)
    base = np.array(per["orig"])
    boot = np.random.default_rng(0).integers(0, len(base), size=(2000, len(base)))
    for v in variants:
        a = np.array(per[v])
        dl = a[boot, 0].sum(1) / a[boot, 1].sum(1) - base[boot, 0].sum(1) / base[boot, 1].sum(1)
        lo, hi = np.percentile(dl, [2.5, 97.5])
        print(f"{v}: PQ {tot[v][0] / tot[v][1]:.5f}  delta vs orig {tot[v][0] / tot[v][1] - tot['orig'][0] / tot['orig'][1]:+.5f} "
              f"[{lo:+.5f}, {hi:+.5f}]", flush=True)
    summ = {v: O.summary(org[v]) for v in variants if v in org}
    for v in summ:
        print(O.line(v, summ[v]), flush=True)
    if args.org_out:
        Path(args.org_out).write_text(json.dumps(summ, indent=1))
    if args.totals_out:  # {variant: {stem: [num, den]}}, for pairing variants across runs
        Path(args.totals_out).write_text(json.dumps({v: dict(zip(F.stems, map(list, per[v]))) for v in variants}))


def apply_stem(stem):
    """(stem, RLEs of the final masks under c["variant"]), or (stem, None) when nothing is selected."""
    c = _C
    j = c["stems"].index(stem)
    lo, hi = c["off"][j], c["off"][j + 1]
    sel = np.flatnonzero(c["sel"][lo:hi])
    if sel.size == 0:
        return stem, None
    z = np.load(Path(c["bank"]) / f"{stem}.npz")
    fg0, all_lab, n_rows, h, w, _, _, disk = candidate_masks(c, stem)
    if n_rows != len(z["parent"]):
        raise SystemExit(f"{stem}: rebuilt {n_rows} candidates, bank has {len(z['parent'])}")
    crop, pred, where = refiner_inputs(c, stem)
    if c["variant"] != "orig" and sel.size and crop is None:
        raise SystemExit(f"{stem}: {sel.size} selected candidates but no crops in {c['crops']}")
    _, rles = final_masks(c, c["variant"], z, sel, lo, fg0, all_lab, h, w, disk.mask((h, w)), crop, pred, where)
    return stem, rles


def cmd_apply(args):
    if args.variant != "orig" and not args.crops:
        raise SystemExit(f"{args.variant} needs --crops")
    if args.variant.split("@")[0].startswith("ref") and not args.preds:
        raise SystemExit(f"{args.variant} needs --preds")
    d = np.load(args.scores)
    F = S.Forest(Path(args.bank))
    assert list(d["stems"]) == F.stems
    off = np.concatenate([[0], np.cumsum(np.bincount(F.stem_of, minlength=F.n_stems))])
    c = context(args, d["score"])
    c.update({"stems": F.stems, "off": off, "sel": d["sel"].astype(bool), "score": d["score"], "crops": args.crops,
              "preds": args.preds, "variant": args.variant, **blend_options(args)})
    preds = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(c,)) as pool:
        for stem, rles in pool.map(apply_stem, F.stems):
            if rles is not None:
                preds[stem] = rles
    write_csv(preds, args.out)
    print(f"apply {args.variant}: wrote {args.out}: {sum(len(r) for r in preds.values())} instances over {len(preds)} observations "
          f"({int(c['sel'].sum())} selected candidates)", flush=True)


# ---------------------------------------------------------------- rescue (round 19, an experiment)

#: adding a mask pays off when its view-mean matched IoU exceeds PQ / 2 (every view's denominator grows by 1/2);
#: PQ = the final candidate's OOF 0.4475
ORACLE_U = 0.4475 / 2


def _num_den(gts, rles):
    num = 0.0
    for g in gts:
        if g and rles:
            iou = np.asarray(mask_utils.iou(g, rles, [0] * len(rles))).reshape(len(g), len(rles))
            num += float(iou[iou > 0.5].sum())
    return [num, 0.5 * (len(rles) * len(gts) + sum(len(g) for g in gts))]


def rescue_stem(stem):
    """The base variant on the selection, then every relaxed extra (a crop cut at tau - delta outside the selection)
    drawn like refjx: refined on the whole disk, minus the selection's pixels, dropped under MIN_AREA or under half its
    area, joined through unclaimed pixels. Per extra: the mean refiner probability inside its mask and (GT) its
    view-mean matched IoU if added alone. Extras added together claim in stage-1 score order, with the same drop rule.
    Returns [num, den] per variant and one row per extra."""
    c = _C
    j = c["stems"].index(stem)
    z = np.load(Path(c["bank"]) / f"{stem}.npz")
    lo = c["off"][j]
    sel = np.flatnonzero(c["sel"][lo:c["off"][j + 1]])
    gts = [c["gt"][v] or [] for v in c["views"][stem]]
    crop, pred, where = refiner_inputs(c, stem)
    extra = sorted((i for i in where if not c["sel"][lo + i]), key=lambda i: -c["score"][lo + i])
    if sel.size == 0 and not extra:
        return stem, {v: _num_den(gts, []) for v in c["variants"]}, []
    fg0, all_lab, _, h, w, _, _, disk = candidate_masks(c, stem)
    dmask = disk.mask((h, w))
    kept, rles = final_masks(c, c["base"], z, sel, lo, fg0, all_lab, h, w, dmask, crop, pred, where)
    claimed = np.zeros((h, w), bool)
    for m in kept:
        claimed |= m
    cand, rows = [], []
    for i in extra:
        q = where[i]
        seed = row_mask(fg0, all_lab, z, i, h, w)
        m = refined_mask(pred[q], crop["M"][q], seed, h, w, dmask, c["thr"], join=False)
        raw = int(m.sum())
        m = m & ~claimed
        ok = m.sum() >= B.MIN_AREA and m.sum() >= 0.5 * raw
        prob, u = 0.0, 0.0
        if ok:
            win = _bbox(m)
            m[win] = join_parts(m[win], (dmask & ~claimed)[win])
            y0, x0, sub = paste(pred[q].astype(np.float32) / 255.0, crop["M"][q], h, w)
            inside = m[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
            prob = float(sub[inside].mean()) if inside.any() else 0.0
            u = _num_den(gts, [mask_to_rle(m.astype(np.uint8))])[0] / len(gts)
        cand.append((i, m if ok else None, raw, prob, u))
        rows.append({"stem": stem, "row": int(i), "score": float(c["score"][lo + i]), "prob": round(prob, 4),
                     "u": round(u, 4), "raw": raw, "area": int(m.sum()) if ok else 0, "seed": int(seed.sum()),
                     "views": len(gts), "ok": bool(ok)})

    def assemble(keep):
        taken, out = claimed.copy(), list(rles)
        for i, m, raw, _, _ in cand:
            if m is None or i not in keep:
                continue
            m = m & ~taken
            if m.sum() < B.MIN_AREA or m.sum() < 0.5 * raw:
                continue
            taken |= m
            out.append(mask_to_rle(m.astype(np.uint8)))
        return _num_den(gts, out)

    usable = [x for x in cand if x[1] is not None]
    res = {c["base"]: _num_den(gts, rles), "all": assemble({x[0] for x in usable}),
           "oracle": assemble({x[0] for x in usable if x[4] >= ORACLE_U})}
    for t in c["grid"]:
        res[f"resc@{t}"] = assemble({x[0] for x in usable if x[3] >= t})
    return stem, res, rows


def cmd_rescue(args):
    """Would keeping refined relaxed extras beat the selection alone? Variants: the base (e.g. refjx@0.5 on the
    selection, == eval's), all usable extras, the GT oracle (extras whose view-mean matched IoU >= PQ / 2), resc@t
    (extras whose mean refiner probability >= t) and resc@cv (t picked on the other cv5 folds)."""
    if not args.preds or not args.base.startswith("refjx@"):
        raise SystemExit("rescue needs --preds and a refjx@t --base")
    d = np.load(args.scores)
    F = S.Forest(Path(args.bank))
    off = np.concatenate([[0], np.cumsum(np.bincount(F.stem_of, minlength=F.n_stems))])
    c = context(args, d["score"])
    grid = [float(x) for x in args.grid.split(",")]
    variants = [args.base, "all", "oracle"] + [f"resc@{t}" for t in grid]
    c.update({"stems": F.stems, "off": off, "sel": d["sel"], "score": d["score"], "crops": args.crops,
              "preds": args.preds, "variants": variants, "base": args.base, "thr": float(args.base.split("@")[1]),
              "grid": grid, **blend_options(args)})
    per, rows = {v: [] for v in variants}, []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(c,)) as pool:
        for stem, res, r in pool.map(rescue_stem, F.stems):
            for v in variants:
                per[v].append(res[v])
            rows += r
    ok = [r for r in rows if r["ok"]]
    print(f"rescue: {len(rows)} extras with crops, {len(ok)} usable after the selection's claim, "
          f"{sum(r['u'] >= ORACLE_U for r in ok)} with view-mean matched IoU >= {ORACLE_U:.4f}", flush=True)
    for t in grid:
        k = [r for r in ok if r["prob"] >= t]
        print(f"  prob >= {t}: {len(k)} extras, {sum(r['u'] >= ORACLE_U for r in k)} of them worth adding", flush=True)
    fold = np.array([c["fold_of"][s] for s in F.stems])
    refs = [f"resc@{t}" for t in grid]
    arr = {v: np.array(per[v]) for v in refs}
    cv, picks = np.zeros((len(F.stems), 2)), {}
    for f in sorted(set(fold.tolist())):  # threshold chosen on the other cv5 folds
        tr = fold != f
        best = max(refs, key=lambda v: arr[v][tr, 0].sum() / arr[v][tr, 1].sum())
        cv[fold == f] = arr[best][fold == f]
        picks[f] = best
    print(f"resc@cv thresholds per fold {picks}", flush=True)
    variants.append("resc@cv")
    per["resc@cv"] = cv.tolist()
    base = np.array(per[args.base])
    boot = np.random.default_rng(0).integers(0, len(base), size=(2000, len(base)))
    for v in variants:
        a = np.array(per[v])
        dl = a[boot, 0].sum(1) / a[boot, 1].sum(1) - base[boot, 0].sum(1) / base[boot, 1].sum(1)
        lo, hi = np.percentile(dl, [2.5, 97.5])
        pq = a[:, 0].sum() / a[:, 1].sum()
        print(f"{v}: PQ {pq:.5f}  delta vs {args.base} {pq - base[:, 0].sum() / base[:, 1].sum():+.5f} "
              f"[{lo:+.5f}, {hi:+.5f}]", flush=True)
    if args.totals_out:
        Path(args.totals_out).write_text(json.dumps({v: dict(zip(F.stems, map(list, per[v]))) for v in variants}))
    if args.rows_out:
        Path(args.rows_out).write_text(json.dumps(rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("crops", "eval", "apply", "rescue"):
        p = sub.add_parser(name)
        p.add_argument("--bank", required=True)
        p.add_argument("--scores", required=True)
        p.add_argument("--maps", required=True)
        p.add_argument("--dets", required=True)
        p.add_argument("--tags", required=True)
        p.add_argument("--detectors", default="s1024,m1280")
        p.add_argument("--piece-levels", default="0.25,0.40,0.55")
        p.add_argument("--disks", required=True)
        p.add_argument("--folds-file", default=str(HERE / "cv5_folds.json"))
        p.add_argument("--mode", choices=["oof", "test"], default="oof")
        p.add_argument("--workers", type=int, default=4)
        if name != "crops":
            p.add_argument("--pred-weights", default="", help="weights of the --preds members (default: plain mean)")
            p.add_argument("--anchor", type=float, default=0.0,
                           help="blend weight of the stage-1 probability (crop channel 1) into the refiner output")
        if name == "crops":
            p.add_argument("--tau-delta", type=float, default=0.0)
            p.add_argument("--out", required=True)
        elif name == "apply":
            p.add_argument("--crops", default="")
            p.add_argument("--preds", default="", help="refiner prediction dirs (comma-separated = mean of members)")
            p.add_argument("--variant", default="orig", help="orig, or family@threshold, e.g. refj@0.5")
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--crops", required=True)
            p.add_argument("--preds", default="", help="refiner predictions (comma-separated dirs = mean of members); "
                                                       "empty = identity check only")
            p.add_argument("--thr", default="0.4,0.5")
            p.add_argument("--families", default="ref,refj,refjs",
                           help="refined variant families (also refjx, refjxs, refjxcNN)")
            p.add_argument("--org-out", default="", help="write the organisers' statistics per variant (JSON)")
            p.add_argument("--totals-out", default="", help="write per-observation [num, den] per variant (JSON)")
            if name == "rescue":
                p.add_argument("--base", default="refjx@0.5", help="the selection's variant (refjx family)")
                p.add_argument("--grid", default="0.3,0.4,0.5,0.6,0.7,0.8",
                               help="mean refiner probability thresholds for the extras")
                p.add_argument("--rows-out", default="", help="write one row per relaxed extra (JSON)")
    p = sub.add_parser("train")
    p.add_argument("--crops", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--bs", type=int, default=24)
    p.add_argument("--arch", choices=["unetpp", "unet"], default="unetpp")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip-grad", type=float, default=0.0, help="clip the unscaled gradient norm; 0 = off")
    p.add_argument("--max-bad", type=int, default=50, help="abort after this many non-finite batches in an epoch")
    p.add_argument("--seed-cut", type=float, default=0.0,
                   help="share of training crops whose seed loses 15-30 %% of its length at one end; 0 = off")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the returned weights per step; 0 = off")
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--fit-all", action="store_true", help="test crops: predict with saved fold models")
    p.add_argument("--models", default="")
    args = ap.parse_args()
    {"crops": cmd_crops, "train": cmd_train, "eval": cmd_eval, "apply": cmd_apply, "rescue": cmd_rescue}[args.cmd](args)


if __name__ == "__main__":
    main()
