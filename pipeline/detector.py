"""The YOLO11-seg instance detectors: training set export, training, and prediction.

    python pipeline/detector.py export  --split split_f0.json --out yolo_f0
    python pipeline/detector.py train   --data yolo_f0/data.yaml --out det_s1024_f0 --model yolo11s-seg.pt \
        --imgsz 1024 --epochs 60 --batch 8 --patience 0 --no-val --seed 0
    python pipeline/detector.py predict --weights det_s1024_f0/detector_last.pt --subset val --split split_f0.json \
        --imgsz 1024 --out dets_s1024_f0_oof.json            # or --subset test

export: every annotated view of the split is one sample (the organisers score each view, so the detector learns the
spread between annotators), its image the cached flat field (hard-linked), one polygon label per filament (a
multi-part annotation keeps its largest part) in normalised coordinates. train: Ultralytics with mosaic, mixup and
copy-paste off (a tiled mosaic of four Suns breaks the link between a filament's look and its place on the disk),
flips and mild scale / translation / brightness jitter, cosine LR, single class. predict: detections down to
confidence 0.05 (the selection decides later), masks at full resolution as RLE counts,
{stem: [{"score": s, "counts": "..."}, ...]}; --flips merge also predicts on the three mirrored copies and merges
the four lists (filcore.fusion.merge_lists), --flips separate keeps them apart, {stem: [[copy 0], ..., [copy 3]]}.

The split file holds view ids: {"train": [...], "val": [...]}.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filcore import paths  # noqa: E402
from filcore.coco import Coco  # noqa: E402
from filcore.fusion import merge_lists  # noqa: E402
from filcore.rle import mask_to_text  # noqa: E402

#: (mirror left-right, mirror top-bottom) of the four copies of --flips
COPIES = [(False, False), (True, False), (False, True), (True, True)]


# --- export ------------------------------------------------------------------------------------------------------

def shoelace(poly: list[float]) -> float:
    """Area of a flat x, y polygon."""
    x, y = np.asarray(poly[0::2]), np.asarray(poly[1::2])
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def yolo_labels(coco: Coco, vid: str) -> list[str]:
    """`0 x1 y1 x2 y2 ...` per filament of the view, coordinates divided by the frame size and clipped to [0, 1]."""
    view = coco.views[vid]
    lines = []
    for ann in coco.anns.get(vid, []):
        parts = [poly for poly in ann.get("segmentation") or [] if len(poly) >= 6]
        if not parts:
            continue
        poly = max(parts, key=shoelace)
        x = np.clip(np.asarray(poly[0::2], dtype=np.float64) / view.width, 0.0, 1.0)
        y = np.clip(np.asarray(poly[1::2], dtype=np.float64) / view.height, 0.0, 1.0)
        lines.append("0 " + " ".join(f"{a:.6f} {b:.6f}" for a, b in zip(x, y)))
    return lines


def cmd_export(args) -> None:
    coco = Coco(args.annotations)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    out, flat = Path(args.out).resolve(), Path(args.cache_dir) / "flat"
    sets = {"train": split["train"], "val": split["val"]}
    if args.all_data:  # Ultralytics insists on a val set; a few views stand in, nothing reads their scores
        sets = {"train": split["train"] + split["val"], "val": split["val"][:16]}
    for name, vids in sets.items():
        images, labels = out / "images" / name, out / "labels" / name
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        vids = vids[:args.limit] if args.limit else vids
        n_obj = 0
        for vid in vids:
            src, dst = flat / f"{coco.views[vid].stem}.png", images / f"{vid}.png"
            if not src.exists():
                raise SystemExit(f"{src} missing: run preprocess.py first")
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:  # another filesystem
                shutil.copy2(src, dst)
            lines = yolo_labels(coco, vid)
            (labels / f"{vid}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_obj += len(lines)
        print(f"{name}: {len(vids)} views, {n_obj} filaments", flush=True)
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n",
                                   encoding="utf-8")
    print(f"wrote {out / 'data.yaml'}", flush=True)


# --- train -------------------------------------------------------------------------------------------------------

def train_settings(args) -> dict:
    """Keyword arguments of ultralytics' YOLO.train."""
    return dict(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
        workers=args.workers, patience=args.patience, project=str(Path(args.out).resolve()), name="train",
        exist_ok=True, seed=args.seed, deterministic=False, single_cls=True, cos_lr=True, plots=False,
        fraction=args.fraction, val=not args.no_val,
        # augmentation: exact flips and mild jitter only; no mosaic / mixup / copy-paste, no rotation or shear
        mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0, fliplr=0.5, flipud=0.5, degrees=0.0, shear=0.0,
        perspective=0.0, translate=0.05, scale=0.2, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        overlap_mask=True, mask_ratio=4,
    )


def cmd_train(args) -> None:
    from ultralytics import YOLO
    out = Path(args.out).resolve()
    YOLO(args.model).train(**train_settings(args))
    run = out / "train"
    # last.pt (end of the cosine schedule) is what the pipeline uses; best.pt would be chosen on validation views
    for name in ("best", "last"):
        src = run / "weights" / f"{name}.pt"
        if src.exists():
            shutil.copy2(src, out / f"detector_{name}.pt")
            print(f"wrote {out / f'detector_{name}.pt'}", flush=True)
    if not (out / "detector_last.pt").exists():
        raise SystemExit(f"training ended without {run / 'weights' / 'last.pt'}")
    if (run / "results.csv").exists():
        shutil.copy2(run / "results.csv", out / "detector_results.csv")


# --- predict -----------------------------------------------------------------------------------------------------

def detect_copy(model, bgr: np.ndarray, lr: bool, ud: bool, imgsz: int, conf: float = 0.05, iou: float = 0.6,
                max_det: int = 150, device=None) -> list[dict]:
    """Detections on one mirrored copy of a frame, masks mirrored back: [{"score", "counts"}] (score to 4 places,
    empty masks dropped)."""
    view = bgr[:, ::-1] if lr else bgr
    view = np.ascontiguousarray(view[::-1] if ud else view)
    res = model.predict(view, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, retina_masks=True, device=device,
                        verbose=False)[0]
    if res.masks is None or not len(res.boxes):
        return []
    masks = res.masks.data.cpu().numpy() > 0.5
    if ud:
        masks = masks[:, ::-1]
    if lr:
        masks = masks[:, :, ::-1]
    return [{"score": round(float(s), 4), "counts": mask_to_text(m)}
            for m, s in zip(masks, res.boxes.conf.cpu().numpy()) if m.any()]


def cmd_predict(args) -> None:
    from ultralytics import YOLO
    if args.subset == "test":
        stems = paths.image_stems(args.test_images)
    else:
        stems = Coco(args.annotations).stems_of(json.loads(Path(args.split).read_text(encoding="utf-8"))["val"])
    if args.limit:
        stems = stems[:args.limit]
    model, flat, t0 = YOLO(args.weights), Path(args.cache_dir) / "flat", time.time()
    copies = COPIES if args.flips != "none" else COPIES[:1]
    found = {}
    for n, stem in enumerate(stems, start=1):
        bgr = cv2.imread(str(flat / f"{stem}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"{flat / f'{stem}.png'} missing: run preprocess.py first")
        lists = [detect_copy(model, bgr, lr, ud, args.imgsz, args.conf, args.iou, args.max_det, args.device)
                 for lr, ud in copies]
        if args.flips == "separate":
            found[stem] = lists
        elif args.flips == "merge":
            found[stem] = [{"score": round(d["score"], 4), "counts": d["counts"]} for d in merge_lists(lists)]
        else:
            found[stem] = lists[0]
        if n % 25 == 0 or n == len(stems):
            print(f"  {n}/{len(stems)} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(found), encoding="utf-8")
    print(f"wrote {out}: {len(stems)} frames", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--annotations", default=str(paths.TRAIN_COCO))
        p.add_argument("--cache-dir", default=str(paths.CACHE))
        p.add_argument("--limit", type=int, default=0, help="first N views / frames only (smoke test)")
    ex = sub.choices["export"]
    ex.add_argument("--split", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--all-data", action="store_true", help="train on the val views too (final detectors)")
    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True, help="data.yaml written by export")
    tr.add_argument("--out", required=True)
    tr.add_argument("--model", default="yolo11s-seg.pt")
    tr.add_argument("--imgsz", type=int, default=1024)
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--batch", type=int, default=8)
    tr.add_argument("--device", default="0")
    tr.add_argument("--workers", type=int, default=4)
    tr.add_argument("--patience", type=int, default=20)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--fraction", type=float, default=1.0, help="train on this share of the views (smoke test)")
    tr.add_argument("--no-val", action="store_true", help="no per-epoch validation")
    pr = sub.choices["predict"]
    pr.add_argument("--weights", required=True)
    pr.add_argument("--subset", choices=["val", "test"], required=True)
    pr.add_argument("--split", default="", help="split file (for --subset val)")
    pr.add_argument("--test-images", default=str(paths.TEST_IMAGES))
    pr.add_argument("--out", required=True)
    pr.add_argument("--imgsz", type=int, default=1024)
    pr.add_argument("--conf", type=float, default=0.05)
    pr.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    pr.add_argument("--max-det", type=int, default=150)
    pr.add_argument("--device", default=None)
    pr.add_argument("--flips", choices=["none", "merge", "separate"], default="none")
    args = ap.parse_args()
    {"export": cmd_export, "train": cmd_train, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
