"""Freeze the 5-fold, date-grouped split of the 707 training observations.

Every out-of-fold artefact (dense maps, detections, the candidate bank) is
keyed to this file, so it is written once and never regenerated with other
settings. Observations of one calendar day stay in one fold (frames of a day
are near-duplicates). Days are dealt greedily, largest first, to the fold that
keeps the per-fold counts of (number of annotator views x GONG site) closest
to their targets: 3-view observations carry ~40% of the pooled-PQ weight, so
each fold needs its share of them.

    FILAMENT_DATA_ROOT=... python pipeline/make_cv_splits.py
writes pipeline/cv5_folds.json: {"fold_of_stem": {stem: k}, ...}
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from filcore.coco import Coco  # noqa: E402

K = 5
SEED = 20261028


def main() -> None:
    coco = Coco()
    views = defaultdict(list)
    for vid, view in coco.views.items():
        views[view.stem].append(vid)
    stratum = {}
    date_of = {}
    for stem, ids in views.items():
        view = coco.views[ids[0]]
        stratum[stem] = (len(ids), view.site)
        date_of[stem] = view.date
    days = defaultdict(list)
    for stem, d in date_of.items():
        days[d].append(stem)
    total = Counter(stratum.values())
    rng = random.Random(SEED)
    order = sorted(days, key=lambda d: (-len(days[d]), rng.random()))
    counts = [Counter() for _ in range(K)]
    sizes = [0] * K
    fold_of = {}
    for d in order:
        add = Counter(stratum[s] for s in days[d])

        def cost(k):
            c = sum((counts[k][key] + add[key] - total[key] / K) ** 2 - (counts[k][key] - total[key] / K) ** 2
                    for key in add)
            return c + 0.5 * ((sizes[k] + len(days[d]) - len(views) / K) ** 2 - (sizes[k] - len(views) / K) ** 2)

        k = min(range(K), key=lambda k: (cost(k), rng.random()))
        counts[k] += add
        sizes[k] += len(days[d])
        for s in days[d]:
            fold_of[s] = k
    out = {
        "k": K, "seed": SEED, "group_by": "date",
        "n_obs": len(fold_of), "n_views": sum(len(v) for v in views.values()),
        "fold_sizes": sizes,
        "fold_views": [sum(len(views[s]) for s in fold_of if fold_of[s] == k) for k in range(K)],
        "fold_by_nviews": [{str(r): sum(1 for s in fold_of if fold_of[s] == k and stratum[s][0] == r)
                            for r in (1, 2, 3)} for k in range(K)],
        "fold_of_stem": dict(sorted(fold_of.items())),
    }
    path = HERE / "cv5_folds.json"
    path.write_text(json.dumps(out, indent=1))
    digest = hashlib.sha256(json.dumps(out["fold_of_stem"], sort_keys=True).encode()).hexdigest()
    print(json.dumps({k: v for k, v in out.items() if k != "fold_of_stem"}))
    print(f"wrote {path}; fold_of_stem sha256 {digest}")


if __name__ == "__main__":
    main()
