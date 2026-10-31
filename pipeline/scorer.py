"""Hurdle scorer + exact AND-OR tree selection over a candidate bank (pipeline/bank.py).

Pooled PQ of disjoint predictions S is sum_S s_i / (0.5 sum_S R_i + 0.5 G), with
s_i = sum_v 1[IoU_iv > 0.5] IoU_iv over the R_i annotator views, so (Dinkelbach)
the best S keeps the disjoint set maximising sum_S (u_i - tau), u_i = s_i / R_i,
for one global tau (= PQ*/2 at the optimum). u is modelled as a hurdle:
h = share of views hit (LightGBM cross-entropy, weight R), m = mean IoU over
hit views (L2, weight = number of hit views); u_hat = h_hat * m_hat.

Candidates form an AND-OR forest: each tree node is kept, or replaced by its
children's best, or by one of its detector-carved alternative decompositions
(the positive pieces of one carving level contained in it); a virtual root per
observation holds every piece. The DP is exact and selections stay disjoint.

    python pipeline/scorer.py --bank DIR --exp cv|curve|importance|oracle [--no-pieces] [--sets ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np

TAUS = np.round(np.arange(0.10, 0.401, 0.01), 3)
PARAMS = dict(max_depth=4, num_leaves=12, min_data_in_leaf=50, learning_rate=0.03, bagging_fraction=0.8,
              bagging_freq=1, feature_fraction=0.7, lambda_l2=5.0, verbose=-1,
              num_threads=int(os.environ.get("LGB_THREADS", "4")), seed=0)
ROUNDS = 200


class Forest:
    """All observations of a bank as one AND-OR forest (global row indices)."""

    def __init__(self, bank: Path, stems=None, use_pieces=True, chain_feats=False):
        bank = Path(bank)
        meta = bank / "views.json"
        self.has_targets = meta.exists()
        stems = stems or sorted(json.loads(meta.read_text()) if self.has_targets
                                else json.loads((bank / "stems.json").read_text()))
        self.stems = list(stems)
        X, parent, level, is_piece, plev, pairs, stem_of = [], [], [], [], [], [], []
        R, sv, tp, G = [], [], [], []
        off = 0
        self.names = None
        for j, s in enumerate(self.stems):
            z = np.load(bank / f"{s}.npz")
            n = len(z["parent"])
            if self.names is None:
                self.names = [str(x) for x in z["names"]]
            X.append(z["X"].reshape(n, len(z["names"])))
            p = z["parent"].astype(np.int64)
            parent.append(np.where(p >= 0, p + off, -1))
            level.append(z["level"])
            ip = z["is_piece"].astype(bool) if "is_piece" in z else np.zeros(n, bool)
            is_piece.append(ip)
            plev.append(z["piece_level"] if "piece_level" in z else np.full(n, -1))
            if "alt_pairs" in z and len(z["alt_pairs"]):
                a = z["alt_pairs"].astype(np.int64)
                pairs.append(np.stack([a[:, 0] + off, np.where(a[:, 1] >= 0, a[:, 1] + off, -1),
                                       np.full(len(a), j)], axis=1))
            stem_of.append(np.full(n, j))
            if self.has_targets:
                vi = z["view_iou"].astype(np.float64).reshape(n, len(z["n_gt"]))
                hit = vi > 0.5
                R.append(np.full(n, vi.shape[1], float))
                sv.append((vi * hit).sum(1))
                tp.append(hit.sum(1).astype(float))
                G.append(float(z["n_gt"].sum()))
            off += n
        F = len(self.names)
        self.X = np.concatenate([x.reshape(-1, F) for x in X]).astype(np.float32)
        self.parent = np.concatenate(parent)
        self.level = np.concatenate(level).astype(np.int64)
        self.is_piece = np.concatenate(is_piece)
        self.piece_level = np.concatenate(plev).astype(np.int64)
        self.pairs = np.concatenate(pairs) if pairs else np.zeros((0, 3), np.int64)
        self.stem_of = np.concatenate(stem_of)
        self.n = len(self.parent)
        self.n_stems = len(self.stems)
        self.n_plev = int(self.piece_level.max()) + 1 if (self.piece_level >= 0).any() else 0
        # a piece that is (nearly) its own level's tree node adds nothing but a second noisy score
        # for the same mask (optimiser's curse): drop it
        self.excluded = np.zeros(self.n, bool)
        if self.is_piece.any():
            col = {nm: self.names.index(nm) for nm in ("level", "chain_up", "parent_area_ratio")}
            same_level = (self.X[:, col["chain_up"]] - 1) == self.X[:, col["level"]]
            self.excluded = self.is_piece & same_level & (self.X[:, col["parent_area_ratio"]] >= 0.95)
            keep = ~self.excluded[self.pairs[:, 0]] if len(self.pairs) else np.zeros(0, bool)
            self.pairs = self.pairs[keep]
        if chain_feats:
            self.add_chain_features()
        if not use_pieces:
            self.pairs = np.zeros((0, 3), np.int64)
            self.n_plev = 0
        self.use_pieces = use_pieces
        tree = ~self.is_piece
        self.tree_levels = [np.flatnonzero(tree & (self.level == k)) for k in range(int(self.level[tree].max()) + 1)]
        if self.has_targets:
            self.R = np.concatenate(R)
            self.sv = np.concatenate(sv)
            self.tp = np.concatenate(tp)
            self.G = np.array(G)
            self.h = self.tp / self.R
            self.m = np.where(self.tp > 0, self.sv / np.maximum(self.tp, 1), 0)

    def add_chain_features(self):
        """Chain-relative columns (cycle 2): a global tree sees absolute values only and cannot tell which level of a
        chain is the local peak. Per root tree (a piece joins the tree of the node it is an alternative to):
        chain_rel_det_iou = det_best_iou minus the tree's max, chain_rel_edge_drop = the same for p_edge_drop,
        parent_det_iou_diff = det_best_iou minus the parent's (NaN at roots and for pieces)."""
        root = np.arange(self.n)
        par = self.parent.copy()
        for a, b, _ in self.pairs:  # pieces have no parent: attach each to (the tree of) one node it replaces
            if b >= 0 and par[a] < 0:
                par[a] = b
        while True:
            nxt = np.where(par[root] >= 0, par[root], root)
            if np.array_equal(nxt, root):
                break
            root = nxt
        new = []
        for name, short in (("det_best_iou", "det_iou"), ("p_edge_drop", "edge_drop")):
            x = self.X[:, self.names.index(name)].astype(np.float64)
            mx = np.full(self.n, -np.inf)
            np.maximum.at(mx, root, x)
            new.append((f"chain_rel_{short}", x - mx[root]))
        x = self.X[:, self.names.index("det_best_iou")].astype(np.float64)
        new.append(("parent_det_iou_diff", np.where(self.parent >= 0, x - x[np.maximum(self.parent, 0)], np.nan)))
        self.X = np.concatenate([self.X, np.stack([v for _, v in new], 1).astype(np.float32)], 1)
        self.names = self.names + [n for n, _ in new]

    def rows_of(self, stem_idx):
        m = np.isin(self.stem_of, stem_idx) & ~self.excluded
        if not self.use_pieces:
            m &= ~self.is_piece
        return m

    def select(self, score, tau, stem_mask=None):
        """Exact AND-OR DP; returns a boolean selection over rows (restricted to stem_mask's stems)."""
        val = score - tau
        if stem_mask is not None:
            val = np.where(stem_mask[self.stem_of], val, -1e9)
        L = max(self.n_plev, 1)
        posp = np.where(self.is_piece, np.maximum(val, 0), 0.0)
        altsum = np.zeros((self.n, L))
        vroot = np.zeros((self.n_stems, L))
        if len(self.pairs):
            pr, cont, st = self.pairs[:, 0], self.pairs[:, 1], self.pairs[:, 2]
            lv = self.piece_level[pr]
            c = cont >= 0
            np.add.at(altsum, (cont[c], lv[c]), posp[pr[c]])
            np.add.at(vroot, (st[~c], lv[~c]), posp[pr[~c]])
        altmax, altarg = altsum.max(1), altsum.argmax(1)
        sumc = np.zeros(self.n)
        best = np.zeros(self.n)
        for idx in reversed(self.tree_levels):
            best[idx] = np.maximum(np.maximum(val[idx], sumc[idx]), altmax[idx])
            p = self.parent[idx]
            ok = p >= 0
            np.add.at(sumc, p[ok], best[idx][ok])
        roots = np.flatnonzero(~self.is_piece & (self.parent < 0))
        roots_sum = np.bincount(self.stem_of[roots], best[roots], minlength=self.n_stems)
        vmax, varg = vroot.max(1), vroot.argmax(1)
        stem_alt = (vmax > roots_sum) & (vmax > 0)
        sel = np.zeros(self.n, bool)
        desc = np.zeros(self.n, bool)
        chosen = np.full(self.n, -1)
        for idx in self.tree_levels:
            p = self.parent[idx]
            active = np.where(p < 0, ~stem_alt[self.stem_of[idx]], desc[np.maximum(p, 0)])
            v = val[idx]
            keep = active & (v > 0) & (v >= sumc[idx]) & (v >= altmax[idx])
            alt = active & ~keep & (altmax[idx] > sumc[idx]) & (altmax[idx] > 0)
            sel[idx] = keep
            chosen[idx[alt]] = altarg[idx[alt]]
            desc[idx] = active & ~keep & ~alt
        if len(self.pairs):
            pr, cont, st = self.pairs[:, 0], self.pairs[:, 1], self.pairs[:, 2]
            lv = self.piece_level[pr]
            take = np.where(cont >= 0, chosen[np.maximum(cont, 0)] == lv, stem_alt[st] & (varg[st] == lv))
            take &= val[pr] > 0
            sel[pr[take]] = True
        return sel

    def stem_totals(self, sel):
        num = np.bincount(self.stem_of[sel], self.sv[sel], minlength=self.n_stems)
        npred = np.bincount(self.stem_of[sel], self.R[sel], minlength=self.n_stems)
        return num, 0.5 * (npred + self.G)

    def pq(self, sel, stem_idx=None):
        num, den = self.stem_totals(sel)
        if stem_idx is not None:
            num, den = num[stem_idx], den[stem_idx]
        return float(num.sum() / den.sum())


def date_folds(stems, k, seed):
    dates = sorted({s[:8] for s in stems})
    rng = np.random.default_rng(seed)
    rng.shuffle(dates)
    fold_of = {d: i % k for i, d in enumerate(dates)}
    return [[i for i, s in enumerate(stems) if fold_of[s[:8]] == f] for f in range(k)]


#: scorer variants (Exp T2): "default" as above; "small" = shallower, more regularised, feature-subsampled
#: booster; MONOTONE gives +1/-1 constraints by descriptor name (more detector agreement / sharper edges / darker
#: ribbon -> more likely annotated); BAG > 1 averages boosters over seeds
VARIANT = {"params": "default", "monotone": False, "bag": 1, "regime": False}
#: regime split (cycle 2): fndiag found 92% of the dropped-but-matchable small filaments have no confident YOLO
#: detection; a global booster pools them with every detector-less fragment, so fit separate hurdle boosters for
#: candidates whose best detection scores below / at least REGIME_SCORE (bank.py's DET_KEEP)
REGIME_SCORE = 0.25
SMALL = dict(max_depth=3, num_leaves=8, min_data_in_leaf=80, feature_fraction=0.5)
#: bank0 (565 obs) learning curve was flat under SMALL: test whether more capacity pays at this size
LARGE = dict(max_depth=6, num_leaves=31, min_data_in_leaf=40, learning_rate=0.05)
MONOTONE = {"det_best_score": 1, "det_best_iou": 1, "det_precision": 1, "p_max": 1, "p_mean": 1, "I_ring_z": 1,
            "p_edge_drop": 1, "p_ring3": -1, "det_split_gap": -1}


def _params(F, cols, seed):
    p = {**PARAMS, "seed": seed}
    if VARIANT["params"] == "small":
        p.update(SMALL)
    elif VARIANT["params"] == "large":
        p.update(LARGE)
    if VARIANT["monotone"]:
        p["monotone_constraints"] = [MONOTONE.get(F.names[c], 0) for c in cols]
        p["monotone_constraints_method"] = "advanced"
    return p


def _low_det(F):
    return F.X[:, F.names.index("det_best_score")] < REGIME_SCORE


def fit(F, rows, cols, params=None, rounds=ROUNDS):
    if VARIANT.get("regime"):
        lo = _low_det(F)
        return {"lo": _fit(F, rows & lo, cols, params, rounds), "hi": _fit(F, rows & ~lo, cols, params, rounds)}
    return _fit(F, rows, cols, params, rounds)


def _fit(F, rows, cols, params=None, rounds=ROUNDS):
    X = F.X[rows][:, cols]
    pos = rows & (F.tp > 0)
    models = []
    for b in range(VARIANT["bag"]):
        p = params or _params(F, cols, b)
        bh = lgb.train({**p, "objective": "cross_entropy"}, lgb.Dataset(X, F.h[rows], weight=F.R[rows]), rounds)
        bm = lgb.train({**p, "objective": "regression"}, lgb.Dataset(F.X[pos][:, cols], F.m[pos], weight=F.tp[pos]),
                       rounds)
        models.append((bh, bm))
    return models


def predict(models, F, rows, cols, out):
    if isinstance(models, dict):
        lo = _low_det(F)
        _predict(models["lo"], F, rows & lo, cols, out)
        _predict(models["hi"], F, rows & ~lo, cols, out)
        return
    _predict(models, F, rows, cols, out)


def _predict(models, F, rows, cols, out):
    if not rows.any():
        return
    X = F.X[rows][:, cols]
    h = np.mean([bh.predict(X) for bh, _ in models], axis=0)
    m = np.mean([bm.predict(X) for _, bm in models], axis=0)
    out[rows] = np.clip(h, 0, 1) * np.clip(m, 0.5, 1.0)


def choose_tau(F, train_idx, cols, seed, inner_k=4):
    oof = np.zeros(F.n)
    stems = [F.stems[i] for i in train_idx]
    for f in date_folds(stems, inner_k, seed + 100):
        fi = [train_idx[i] for i in f]
        rest = sorted(set(train_idx) - set(fi))
        predict(fit(F, F.rows_of(rest), cols), F, F.rows_of(fi), cols, oof)
    mask = np.zeros(F.n_stems, bool)
    mask[train_idx] = True
    curve = np.array([F.pq(F.select(oof, t, mask), train_idx) for t in TAUS])
    smooth = np.convolve(np.pad(curve, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    return float(TAUS[int(np.argmax(smooth))])


def cv(F, cols, repeats, k, frac=1.0, seed0=0, inner_k=4):
    num_sum, den_sum, per_rep = np.zeros(F.n_stems), np.zeros(F.n_stems), []
    for r in range(repeats):
        score = np.zeros(F.n)
        sel = np.zeros(F.n, bool)
        for fi in date_folds(F.stems, k, seed0 + r):
            train = sorted(set(range(F.n_stems)) - set(fi))
            if frac < 1.0:
                tdates = sorted({F.stems[i][:8] for i in train})
                keep = set(np.random.default_rng(1000 + r).choice(tdates, max(4, round(frac * len(tdates))),
                                                                  replace=False))
                train = [i for i in train if F.stems[i][:8] in keep]
            tau = choose_tau(F, train, cols, r, inner_k)
            predict(fit(F, F.rows_of(train), cols), F, F.rows_of(fi), cols, score)
            mask = np.zeros(F.n_stems, bool)
            mask[fi] = True
            sel |= F.select(score, tau, mask) & mask[F.stem_of]
        num, den = F.stem_totals(sel)
        num_sum += num
        den_sum += den
        per_rep.append(float(num.sum() / den.sum()))
    return per_rep, num_sum / repeats, den_sum / repeats


def oracle(F):
    tau = 0.2
    for _ in range(30):
        p = F.pq(F.select(np.where(F.rows_of(range(F.n_stems)), F.sv / F.R, -1), tau))
        if abs(p / 2 - tau) < 1e-7:
            break
        tau = p / 2
    return p, tau


#: never model inputs: the carving detection's score/rank come from cross-fold merged detections on test
#: (not distributed like one fold's), and det_best_score carries the same information parity-safely
NOT_FEATURES = ("piece_score", "piece_rank")


def feature_sets(names):
    ok = [i for i, n in enumerate(names) if n not in NOT_FEATURES]
    return {
        "all": ok,
        "nofr": [i for i in ok if not names[i].startswith("f_")],
        "nodet": [i for i in ok if not names[i].startswith(("f_", "det_"))],
        "noring": [i for i in ok if not names[i].startswith(("f_", "I_", "p_ring", "p_edge", "p_core"))
                   and names[i] not in ("det_precision", "det_split_gap")],
        # bank2 ablations: without the ridge descriptors / without the per-detector scores
        "nofr_noridge": [i for i in ok if not names[i].startswith(("f_", "ridge_")) and names[i] not in
                         ("p_ridge", "s_ridge")],
        "nofr_nodetg": [i for i in ok if not names[i].startswith(("f_", "det_g"))],
        "nofr_old": [i for i in ok if not names[i].startswith(("f_", "det_g", "ridge_")) and names[i] not in
                     ("p_ridge", "s_ridge")],
        # cycle 2: nofr_old without the chain-relative columns (the paired baseline when --chain-feats is on)
        "nofr_old_nochain": [i for i in ok if not names[i].startswith(("f_", "det_g", "ridge_", "chain_rel_"))
                             and names[i] not in ("p_ridge", "s_ridge", "parent_det_iou_diff")],
        # map-side pilot: nofr_old without the spine descriptors (q_*), the paired baseline on a spine bank
        "nofr_old_noq": [i for i in ok if not names[i].startswith(("f_", "det_g", "ridge_", "q_")) and names[i] not in
                         ("p_ridge", "s_ridge")],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--exp", default="cv")
    ap.add_argument("--sets", default="nofr")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-pieces", action="store_true")
    ap.add_argument("--regime-split", action="store_true", help="separate boosters by detector confidence (_low_det)")
    ap.add_argument("--chain-feats", action="store_true", help="add the chain-relative columns (Forest.add_chain_features)")
    ap.add_argument("--base", default="", help="JSON {stem: totals(n_gt,n_pred,tp,fp,fn,iou_sum)} or {stem: [num, den]} "
                                               "(another arm's --totals-out) to pair against")
    ap.add_argument("--totals-out", default="", help="cv: write each set's per-observation [num, den], averaged over "
                                                     "the repeats, to <prefix>.<set>.json (pairs with --base)")
    ap.add_argument("--params", choices=["default", "small", "large"], default="default")
    ap.add_argument("--monotone", action="store_true")
    ap.add_argument("--bag", type=int, default=1)
    ap.add_argument("--out", default="oof_scores.npz", help="--exp dump output")
    ap.add_argument("--tau-offset", type=float, default=0.0,
                    help="dump: select at the nested tau + offset (a lower tau trades misses for false positives)")
    args = ap.parse_args()
    VARIANT.update(params=args.params, monotone=args.monotone, bag=args.bag, regime=args.regime_split)
    F = Forest(Path(args.bank), use_pieces=not args.no_pieces, chain_feats=args.chain_feats)
    print(f"{F.n_stems} obs, {int((~F.is_piece).sum())} nodes, {int(F.is_piece.sum())} pieces "
          f"({int(F.excluded.sum())} trivial dropped) "
          f"(used: {F.use_pieces}), {len(F.names)} features", flush=True)
    base = json.loads(Path(args.base).read_text()) if args.base else None
    if base:
        bt = np.array([base[s] for s in F.stems], float)
        # same date partitions (seeds) on the same stems make two arms' cv totals paired
        bnum, bden = (bt[:, 0], bt[:, 1]) if bt.shape[1] == 2 else (bt[:, 5], bt[:, 2] + 0.5 * bt[:, 3] + 0.5 * bt[:, 4])
        print(f"base PQ {bnum.sum() / bden.sum():.5f}")
    boot = np.random.default_rng(0).integers(0, F.n_stems, size=(2000, F.n_stems))

    def report(tag, per_rep, num, den):
        line = f"[{tag} {VARIANT}] PQ mean {np.mean(per_rep):.5f} sd {np.std(per_rep):.5f} reps {np.round(per_rep, 4).tolist()}"
        if base:
            d = num.sum() / den.sum() - bnum.sum() / bden.sum()
            bd = num[boot].sum(1) / den[boot].sum(1) - bnum[boot].sum(1) / bden[boot].sum(1)
            lo, hi = np.percentile(bd, [2.5, 97.5])
            line += f"; delta {d:+.5f} [{lo:+.5f}, {hi:+.5f}] P>0 {np.mean(bd > 0):.3f}"
        print(line, flush=True)

    sets = feature_sets(F.names)
    if args.exp == "oracle":
        p, t = oracle(F)
        print(f"oracle PQ {p:.5f} at tau {t:.4f}")
        return
    if args.exp == "dump":
        # one cross-fitted pass (date folds, seed 0): OOF score per row and the fold's nested tau per observation
        cols = sets[args.sets.split(",")[0]]
        score, tau_obs = np.zeros(F.n), np.zeros(F.n_stems)
        for fi in date_folds(F.stems, args.k, 0):
            train = sorted(set(range(F.n_stems)) - set(fi))
            tau_obs[fi] = choose_tau(F, train, cols, 0)
            predict(fit(F, F.rows_of(train), cols), F, F.rows_of(fi), cols, score)
        sel = np.zeros(F.n, bool)
        for t in sorted(set(tau_obs)):
            m = tau_obs == t
            sel |= F.select(score, t + args.tau_offset, m) & m[F.stem_of]
        num, den = F.stem_totals(sel)
        print(f"dump: OOF PQ {num.sum() / den.sum():.5f} with nested taus {sorted(set(tau_obs.tolist()))}"
              f"{f' offset {args.tau_offset:+.3f}' if args.tau_offset else ''}; {int(sel.sum())} selected")
        np.savez_compressed(args.out, score=score, tau_obs=tau_obs, stems=np.array(F.stems), sel=sel,
                            stem_of=F.stem_of, cols=np.array(cols), variant=json.dumps(VARIANT))
        return
    for name in args.sets.split(","):
        cols = sets[name]
        if args.exp == "cv":
            per_rep, num, den = cv(F, cols, args.repeats, args.k)
            report(name, per_rep, num, den)
            if args.totals_out:
                Path(f"{args.totals_out}.{name}.json").write_text(
                    json.dumps({s: [float(a), float(b)] for s, a, b in zip(F.stems, num, den)}))
        elif args.exp == "curve":
            for frac in (0.25, 0.5, 0.75, 1.0):
                report(f"{name} frac {frac}", *cv(F, cols, args.repeats, args.k, frac=frac))
        elif args.exp == "importance":
            bh, bm = fit(F, F.rows_of(range(F.n_stems)), cols)[0]
            for tag, b in (("h", bh), ("m", bm)):
                g = b.feature_importance("gain")
                order = np.argsort(g)[::-1][:15]
                print(tag, [(F.names[cols[i]], round(float(g[i] / g.sum()), 3)) for i in order])


if __name__ == "__main__":
    main()
