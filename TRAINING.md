# How the weights were trained

Every model was trained on Kaggle GPUs (NVIDIA T4) with the code in this repository, five times: once per fold of
`pipeline/cv5_folds.json` (707 training observations in 5 date-grouped folds, frozen before any experiment). The model
of fold *k* never sees fold *k*, so its predictions on fold *k* are out of fold (OOF); the scorer and the refiner are
trained on those OOF predictions, and every validation number in the report is OOF. Commands below are run from the
repository root with `FILAMENT_DATA_ROOT` set to the competition data; the preprocessing cache goes to `data/cache`
(`FILAMENT_CACHE_DIR` or `--cache-dir` to move it). The folds were drawn once by `pipeline/make_cv_splits.py`.

## 0. Preprocessing (train and test frames)

```bash
python pipeline/preprocess.py --cache-dir data/cache
```

## 1. Dense probability maps (10 models)

`r34soft` = U-Net, `ppsoft` = UNet++, both ResNet34; soft consensus target, BCE + 0.5 soft Dice, EMA 0.998, 30 epochs.
Each run writes its checkpoint and its fold's OOF maps and the test maps (`maps/{oof,test}_<tag>_f<k>/`).

```bash
for k in 0 1 2 3 4; do
  python pipeline/train_cv.py --folds-file pipeline/cv5_folds.json --fold $k --tag r34soft --arch unet \
      --target consensus --dice-weight 0.5 --ema 0.998 --seed 0 --out maps
  python pipeline/train_cv.py --folds-file pipeline/cv5_folds.json --fold $k --tag ppsoft --arch unetpp \
      --target consensus --dice-weight 0.5 --ema 0.998 --batch-size 8 --accum 2 --seed 0 --out maps
done
```

## 2. Instance detectors (10 models)

For fold *k*, the split file has `train` = every annotator view of the other folds and `val` = the views of fold *k*
(`{"group_by": "date", "fold": k, "train": [...image ids], "val": [...]}`). YOLO11s-seg at 1024 px for 60 epochs and
YOLO11m-seg at 1280 px for 80 epochs, last epoch, no early stopping:

```bash
# split_f0.json ... split_f4.json from the frozen folds
python - <<'PY'
import json, sys
sys.path.insert(0, "pipeline")
from filcore.coco import Coco
fold_of, ann = json.load(open("pipeline/cv5_folds.json"))["fold_of_stem"], Coco()
for k in range(5):
    json.dump({"group_by": "date", "fold": k,
               "train": sorted(i for i, r in ann.views.items() if fold_of[r.stem] != k),
               "val": sorted(i for i, r in ann.views.items() if fold_of[r.stem] == k)}, open(f"split_f{k}.json", "w"))
PY
for k in 0 1 2 3 4; do
  python pipeline/detector.py export --out yolo_f$k --split split_f$k.json
  python pipeline/detector.py train --data yolo_f$k/data.yaml --out det_s1024_f$k \
      --model yolo11s-seg.pt --imgsz 1024 --epochs 60 --batch 8 --patience 0 --no-val --seed 0
  python pipeline/detector.py predict --weights det_s1024_f$k/detector_last.pt --subset val \
      --split split_f$k.json --imgsz 1024 --out dets/dets_s1024_f${k}_oof.json   # and --subset test -> _test.json
  # the same with yolo11m-seg.pt, --imgsz 1280, --epochs 80 -> det_m1280_f$k, dets_m1280_f$k_*.json
done
```

## 3. Candidate bank and scorer

```bash
python pipeline/bank.py --mode oof --maps maps --dets dets --disks data/cache/disk.json \
    --tags r34soft,ppsoft --detectors s1024,m1280 --out bank_oof
# cross-fitted scores and the nested-tau selection of every OOF observation (used by the refiner)
python pipeline/scorer.py --bank bank_oof --exp dump --params small --sets nofr_old --out oof_scores.npz
# final boosters on all 707 observations + tau from the OOF curve -> weights/scorer/
python pipeline/bank.py --mode test --maps maps --dets dets --disks data/cache/disk.json \
    --tags r34soft,ppsoft --detectors s1024,m1280 --out bank_test
python pipeline/submit.py --oof-bank bank_oof --test-bank bank_test --maps maps --dets dets \
    --disks data/cache/disk.json --sets nofr_old --params small --save-model weights/scorer --out stage1.csv
```

## 4. Refiner (5 models)

```bash
COMMON="--maps maps --dets dets --disks data/cache/disk.json --tags r34soft,ppsoft --detectors s1024,m1280"
python pipeline/refine.py crops --bank bank_oof --scores oof_scores.npz $COMMON --tau-delta 0.06 --out rcrops
python pipeline/refine.py train --crops rcrops --out refiner --save-models --arch unetpp
python pipeline/refine.py eval --bank bank_oof --scores oof_scores.npz $COMMON --crops rcrops --preds refiner \
    --families refj,refjx --thr 0.4,0.45,0.5,0.55,0.6 --org-out org.json --totals-out totals.json
```

## 5. fp16 export (exact under autocast)

```bash
for k in 0 1 2 3 4; do
  for tag in r34soft ppsoft; do
    python pipeline/export_fp16.py dense --in model_${tag}_f$k.pt --out weights/dense/model_${tag}_f$k.fp16.pt
    python pipeline/export_fp16.py check --in model_${tag}_f$k.pt --fp16 weights/dense/model_${tag}_f$k.fp16.pt --kind dense
  done
  python pipeline/export_fp16.py refiner --in refiner/refiner_f$k.pt --out weights/refiner/refiner_f$k.pt \
      --arch unetpp --encoder resnet34 --in-channels 5
  python pipeline/export_fp16.py check --in refiner/refiner_f$k.pt --fp16 weights/refiner/refiner_f$k.pt \
      --kind refiner --arch unetpp --encoder resnet34 --in-channels 5
done
```

The YOLO checkpoints (`weights/det/`) are Ultralytics' own fp16 `last.pt` files, used as saved.
