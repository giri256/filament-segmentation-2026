"""Final step: fit the hurdle scorer on the whole OOF bank (or load saved boosters), pick tau out of fold, select on
the test bank, write the CSV.

tau: 5-fold date-grouped CV over the OOF bank gives out-of-fold scores for every
observation; tau maximises (3-point smoothed) pooled PQ of their selection.
Masks are rebuilt deterministically from the same inputs bank.py used (the
fold-averaged probability map's level sets and the cross-fold carving), and
each test observation's candidate count is checked against its bank file.
Kept masks are made disjoint exactly as refine.py eval does for the OOF numbers: in
score order, each mask loses the pixels already claimed and is dropped if what is left
is under MIN_AREA or under half of it.

--save-model DIR writes the fitted boosters (LightGBM text), tau and the feature columns
(config.json); --model DIR loads them instead of fitting, so the OOF bank is not needed.
--scores-out FILE writes the test selection in the score-dump format refine.py reads
(stems, score, tau_obs, sel), for the refiner's test crops and `refine.py apply`.

    FILAMENT_DATA_ROOT=... python pipeline/submit.py --oof-bank DIR --test-bank DIR --maps DIR --dets DIR \
        --disks disk.json --out submission.csv [--sets nofr] [--tau 0.25] [--no-pieces] [--save-model DIR]
    FILAMENT_DATA_ROOT=... python pipeline/submit.py --model DIR --test-bank DIR --maps DIR --dets DIR \
        --disks disk.json --out submission.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bank as B  # noqa: E402
import scorer as S  # noqa: E402
from filcore.limb import load_disks  # noqa: E402
from filcore.rle import encode as mask_to_rle, write_csv  # noqa: E402


def oof_tau(F, cols, k=5, seed=0):
    score = np.zeros(F.n)
    for fi in S.date_folds(F.stems, k, seed):
        train = sorted(set(range(F.n_stems)) - set(fi))
        S.predict(S.fit(F, F.rows_of(train), cols), F, F.rows_of(fi), cols, score)
    curve = np.array([F.pq(F.select(score, t)) for t in S.TAUS])
    smooth = np.convolve(np.pad(curve, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    j = int(np.argmax(smooth))
    return float(S.TAUS[j]), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-bank", default="")
    ap.add_argument("--model", default="", help="load boosters + tau + columns saved with --save-model")
    ap.add_argument("--save-model", default="", help="write the fitted boosters, tau and columns here")
    ap.add_argument("--test-bank", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tags", default="r34soft,ppsoft")
    ap.add_argument("--detectors", default="s1024,m1280")
    ap.add_argument("--piece-levels", default="0.25,0.40,0.55")
    ap.add_argument("--disks", required=True)
    ap.add_argument("--folds-file", default=str(HERE / "cv5_folds.json"))
    ap.add_argument("--sets", default="nofr")
    # the booster variant every OOF number was measured with (scorer.py --params small); sub1 (public 0.38) ran
    # before this flag existed and so used the default parameters
    ap.add_argument("--params", choices=["default", "small", "large"], default="small")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--no-pieces", action="store_true")
    ap.add_argument("--scores-out", default="", help="write the test selection as a refine.py score dump (.npz)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    S.VARIANT.update(params=args.params)

    if args.model:
        cfg = json.loads((Path(args.model) / "config.json").read_text())
        names, col_names, tau = cfg["names"], cfg["columns"], cfg["tau"]
        models = [(lgb.Booster(model_file=str(Path(args.model) / f"h_{b}.txt")),
                   lgb.Booster(model_file=str(Path(args.model) / f"m_{b}.txt"))) for b in range(cfg["bag"])]
        print(f"loaded {len(models)} booster pair(s) from {args.model}; tau {tau}; {len(col_names)} columns", flush=True)
    else:
        F = S.Forest(Path(args.oof_bank), use_pieces=not args.no_pieces)
        names = list(F.names)
        cols = S.feature_sets(F.names)[args.sets]
        col_names = [names[i] for i in cols]
        if args.tau is None:
            tau, curve = oof_tau(F, cols)
            print(f"tau {tau} (OOF PQ at tau {curve[list(S.TAUS).index(tau)]:.5f}; curve max {curve.max():.5f})",
                  flush=True)
        else:
            tau = args.tau
        models = S.fit(F, F.rows_of(range(F.n_stems)), cols)
        if args.save_model:
            d = Path(args.save_model)
            d.mkdir(parents=True, exist_ok=True)
            for b, (bh, bm) in enumerate(models):
                bh.save_model(str(d / f"h_{b}.txt"))
                bm.save_model(str(d / f"m_{b}.txt"))
            (d / "config.json").write_text(json.dumps({"tau": tau, "names": names, "columns": col_names,
                                                       "bag": len(models), "sets": args.sets,
                                                       "variant": S.VARIANT, "pieces": not args.no_pieces}, indent=1))
            print(f"saved {len(models)} booster pair(s), tau and columns to {d}", flush=True)
    T = S.Forest(Path(args.test_bank), use_pieces=not args.no_pieces)
    assert list(T.names) == names, "test bank and scorer have different descriptors"
    cols = [names.index(n) for n in col_names]
    score = np.zeros(T.n)
    S.predict(models, T, T.rows_of(range(T.n_stems)), cols, score)
    sel = T.select(score, tau)
    print(f"selected {int(sel.sum())} candidates ({int((sel & T.is_piece).sum())} pieces) over {T.n_stems} test obs",
          flush=True)
    if args.scores_out:
        np.savez(args.scores_out, stems=np.array(T.stems), score=score, tau_obs=np.full(T.n_stems, tau), sel=sel)
        print(f"wrote the test score dump {args.scores_out}", flush=True)

    folds = json.loads(Path(args.folds_file).read_text())
    k = folds["k"]
    tags = args.tags.split(",")
    detectors = args.detectors.split(",") if args.detectors else []
    dets = {f: (B.merged_dets(args.dets, detectors, f, "test") if detectors else {}) for f in range(k)}
    disks = load_disks(args.disks)
    piece_levels = [float(x) for x in args.piece_levels.split(",") if x]
    c = {"mode": "test", "maps": args.maps, "tags": tags}
    preds = {}
    for j, stem in enumerate(T.stems):
        rows = np.flatnonzero(sel & (T.stem_of == j))
        if rows.size == 0:
            continue
        off = int(np.flatnonzero(T.stem_of == j)[0])
        z = np.load(Path(args.test_bank) / f"{stem}.npz")
        p_tree = np.mean([B.load_maps(c, stem, f)[0] for f in range(k)], axis=0)
        h, w = p_tree.shape
        disk = disks[stem]
        fg0, lab, perim, n_lab = B.level_sets(p_tree, disk.mask(p_tree.shape))
        nodes, _, _ = B.build_tree(fg0, lab, perim, n_lab, w, disk)
        carve = B.merge_dets([dets[f].get(stem, []) for f in range(k)])
        lab_ext, _, _, pieces = B.carve_pieces(fg0, lab, n_lab, nodes, carve, h, w, disk, piece_levels)
        if len(nodes) + len(pieces) != len(z["parent"]):
            raise SystemExit(f"{stem}: rebuilt {len(nodes)}+{len(pieces)} candidates, bank has {len(z['parent'])}")
        all_lab = lab + lab_ext
        rles = []
        claimed = np.zeros((h, w), bool)
        for r in sorted(rows, key=lambda r: -score[r]):  # as refine.py eval: score order, overlaps resolved
            i = r - off
            m = np.zeros(h * w, bool)
            m[fg0[all_lab[int(z["level"][i])] == int(z["label"][i])]] = True
            m = m.reshape(h, w)
            raw = int(m.sum())
            m &= ~claimed
            if m.sum() < B.MIN_AREA or m.sum() < 0.5 * raw:
                continue
            claimed |= m
            rles.append(mask_to_rle(m.astype(np.uint8)))
        preds[stem] = rles
    write_csv(preds, args.out)
    print(f"wrote {args.out}: {sum(len(r) for r in preds.values())} instances over {len(preds)} observations (tau {tau})")


if __name__ == "__main__":
    main()
