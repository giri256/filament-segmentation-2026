# Learned instance selection for solar filament segmentation

Our solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026):
instance masks of solar filaments in GONG H-alpha full-disk images (MAGFiLO), scored by pooled Panoptic Quality (PQ).
The repository contains every trained weight and reproduces our leaderboard submission byte for byte.

## Method in one paragraph

Two dense networks (U-Net and UNet++, ResNet34) give a filament probability map, and two YOLO11 detectors give
instance masks (each model trained on five date-grouped folds). Instead of fixed thresholds, each frame gets a **bank of
candidate masks**: the nested connected components of the probability map at 15 levels (a component tree), plus pieces
carved by the detections. A **hurdle LightGBM model** predicts each candidate's expected IoU per annotator view, and an
**exact tree search** keeps the disjoint set that maximises pooled PQ (keep a filament when its expected IoU per view
exceeds τ = 0.24, half the achievable PQ). A **refiner** (UNet++ on rotated, native-resolution crops) then redraws every
kept filament as one connected region.

| out of fold, 707 training observations | PQ |
|---|---|
| fixed fusion of the same maps and detections (filament-seg's method) | 0.4232 |
| + learned selection | 0.4417 |
| + refiner (submitted) | 0.4475 |
| public leaderboard | 0.39 |

## Run it

GPU required (developed on Kaggle's 2× NVIDIA T4, Python 3.13, CUDA 12.8). The 180 test frames take 151 minutes
on 2× T4: preprocessing 1, dense maps 111, detectors 18, bank 7, selection 6, refiner 8.

```bash
pip install -r requirements.txt      # see the torch/CUDA note inside
```

Then open `notebook.ipynb`, set `DATA_ROOT` to the competition's `MAGFiLO_1.0_Kaggle_2026` folder and run all cells:
it executes the six stages one by one (with an explanation of each) and writes `submission.csv`. The same from a
shell:

```bash
FILAMENT_DATA_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026 python pipeline/infer.py --weights weights --work work --out submission.csv
```

## Layout

| path | content |
|---|---|
| `notebook.ipynb` | the whole inference pipeline, stage by stage |
| `pipeline/infer.py` | the inference driver (`Run`: preprocess, dense, detect, bank, select, refine) |
| `pipeline/preprocess.py` | limb circle, limb-darkening flat field and label masks of every frame (the cache) |
| `pipeline/dense_infer.py` | tiled, 8-way TTA dense inference; loads fp32 or fp16 checkpoints |
| `pipeline/detector.py` | YOLO11-seg detectors: training-set export, training, prediction |
| `pipeline/bank.py` | candidate bank: component tree, detector-carved pieces, descriptors |
| `pipeline/scorer.py` | hurdle scorer, exact AND-OR tree selection, cross-validation experiments |
| `pipeline/submit.py` | final selection on the test bank → CSV (and the score dump for the refiner) |
| `pipeline/refine.py` | refiner: crops, training, evaluation, paste-back (`apply`) |
| `pipeline/train_cv.py`, `oof_fusion.py`, `orgstats.py`, `export_fp16.py` | training, the fixed-fusion baseline, the organisers' statistics, fp16 export |
| `pipeline/filcore/` | shared code: data paths, COCO annotations, limb fit and flat field, RLE and the submission file, morphology, detection merging, PQ, training crops |
| `pipeline/cv5_folds.json`, `make_cv_splits.py` | the five date-grouped folds used for every model and number, and the script that drew them |
| `weights/` | 10 dense (fp16), 10 detector, 5 refiner checkpoints, the scorer boosters, `config.json` (1.10 GB; largest file 52 MB) |
| `TRAINING.md` | the exact commands that trained every weight |
| `LICENSE` | MIT |

## Validation

Every number is out of fold: each training observation is predicted by models that never saw its day. Changes were
kept only with a paired gain of at least +0.0025 PQ (probability ≥ 0.95, bootstrap over days). The report gives the
full progression, the organisers' IoU/Dice/1:n/n:1 statistics and the ideas that did not pass.

## Credits and licence

Team "Jahnavi krishnaveni" on Kaggle: Konduru Giridhar Sai Pavan Kumar and Jahnavi krishnaveni (contact:
giridharpavan593@gmail.com).
All code in this repository is ours. Three methods follow [filament-seg](https://github.com/ameyypawar/filament-seg)
by Amey Pawar (MIT License), reimplemented here: the preprocessing (limb fit and flat field, written to reproduce
its output exactly because our networks were trained on it), the dense-model and detector training recipes, and the
fixed detector-guided fusion used as our baseline. Its published weights are not used; all weights here were trained
by us.
Licence: **MIT** (`LICENSE`) for the code and for the dense, refiner and scorer weights. The detector weights in
`weights/det/` were trained with [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO11, which is licensed
under AGPL-3.0; Ultralytics' terms apply to those weights and to the `ultralytics` package. The dense and refiner
encoders started from the ImageNet ResNet34 weights distributed with segmentation-models-pytorch.
