"""Preprocessing cache of every training and test frame: limb circle, flat field, label masks.

Writes into --cache-dir:
  flat/<stem>.png        the flat field (filcore.limb.flat_field) as uint8, clip(flat, 0, 2) * 127.5, which the
                         networks read back as png / 255 in [0, 1]
  disk.json              {stem: {"cx", "cy", "radius"}} of the fitted limb circle, training frames then test
                         frames in sorted order (entries of an existing file are kept and updated)
  mask/<view_id>.png     for each annotated view, the union of its filament polygons (0 / 255)
  consensus/<stem>.png   for each annotated frame, the share of its views marking each pixel, round(share * 255)
Files that exist are kept unless --force (a kept flat field keeps its disk.json entry).

    FILAMENT_DATA_ROOT=<MAGFiLO_1.0_Kaggle_2026> python pipeline/preprocess.py --cache-dir data/cache
    python pipeline/preprocess.py --train-images EMPTY_DIR --cache-dir work/cache    # test frames only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filcore import paths  # noqa: E402
from filcore.coco import Coco  # noqa: E402
from filcore.limb import fit_limb, flat_field, flat_to_u8, load_gray  # noqa: E402
from filcore.rle import decode  # noqa: E402

_JOB: dict = {}


def _setup(coco: Coco | None, cache: Path, force: bool) -> None:
    cv2.setNumThreads(1)
    _JOB.update(coco=coco, cache=cache, force=force)


def _missing(path: Path) -> bool:
    return _JOB["force"] or not path.exists()


def _view_mask(vid: str) -> np.ndarray:
    view = _JOB["coco"].views[vid]
    union = np.zeros((view.height, view.width), dtype=np.uint8)
    for rle in _JOB["coco"].filament_rles(vid):
        union |= decode(rle).astype(np.uint8)
    return union


def process_frame(task: tuple[str, str, list[str]]) -> tuple[str, dict | None]:
    """Flat field and limb of one frame (unless cached), and its views' label masks. Returns the frame's disk.json
    entry, or None when the cached flat field was kept."""
    stem, image, views = task
    cache = _JOB["cache"]
    geometry = None
    flat_png = cache / "flat" / f"{stem}.png"
    if _missing(flat_png):
        gray = load_gray(image)
        disk = fit_limb(gray)
        cv2.imwrite(str(flat_png), flat_to_u8(flat_field(gray, disk)))
        geometry = {"cx": disk.cx, "cy": disk.cy, "radius": disk.radius}
    mask_pngs = [cache / "mask" / f"{vid}.png" for vid in views]
    consensus_png = cache / "consensus" / f"{stem}.png"
    if views and (_missing(consensus_png) or any(_missing(p) for p in mask_pngs)):
        masks = [_view_mask(vid) for vid in views]
        for png, m in zip(mask_pngs, masks):
            if _missing(png):
                cv2.imwrite(str(png), m * 255)
        if _missing(consensus_png):
            votes = masks[0].astype(np.float64)
            for m in masks[1:]:
                votes = votes + m
            cv2.imwrite(str(consensus_png), np.round(votes / len(masks) * 255.0).astype(np.uint8))
    return stem, geometry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-images", default=str(paths.TRAIN_IMAGES))
    ap.add_argument("--test-images", default=str(paths.TEST_IMAGES))
    ap.add_argument("--annotations", default=str(paths.TRAIN_COCO))
    ap.add_argument("--cache-dir", default=str(paths.CACHE))
    ap.add_argument("--workers", type=int, default=0, help="0 = all CPUs")
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (smoke test)")
    ap.add_argument("--force", action="store_true", help="recompute cached files")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    for sub in ("flat", "mask", "consensus"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    train, test = paths.image_files(args.train_images), paths.image_files(args.test_images)
    coco = Coco(args.annotations) if train else None
    views = coco.by_stem() if coco else {}
    tasks = [(p.stem, str(p), views.get(p.stem, [])) for p in train] + [(p.stem, str(p), []) for p in test]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise SystemExit("no frames found: check FILAMENT_DATA_ROOT / --train-images / --test-images")
    print(f"{len(tasks)} frames ({len(train)} training, {len(test)} test) -> {cache}", flush=True)

    disk_json = cache / "disk.json"
    disks = json.loads(disk_json.read_text(encoding="utf-8")) if disk_json.exists() else {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers or None, initializer=_setup,
                             initargs=(coco, cache, args.force)) as pool:
        for n, (stem, geometry) in enumerate(pool.map(process_frame, tasks, chunksize=4), start=1):
            if geometry is not None:
                disks[stem] = geometry
            if n % 25 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    disk_json.write_text(json.dumps(disks, indent=2), encoding="utf-8")
    print(f"wrote {disk_json} ({len(disks)} frames)", flush=True)


if __name__ == "__main__":
    main()
