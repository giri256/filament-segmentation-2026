"""End-to-end inference: test H-alpha images -> submission CSV.

Runs the same steps, with the same file layout, as the validated Kaggle route (cv5 dense/detector kernels ->
bank -> sub2's scorer -> sub4's refiner), so its CSV must equal that route's byte for byte:

  1. preprocess   preprocess.py: limb circle, limb-darkening flat field, disk geometry (test images only)
  2. dense        dense_infer.py: every dense checkpoint, tiled D4 TTA -> maps/test_<tag>_f<k>/
  3. detect       detector.py predict: every detector checkpoint -> dets/dets_<det>_f<k>_test.json
  4. bank         bank.py --mode test: component tree of the fold-averaged map + carved detector pieces + descriptors
  5. select       submit.py --model: hurdle LightGBM scores, AND-OR DP selection at the OOF tau
                  (writes the stage-1 CSV and the selection as a score dump)
  6. refine       refine.py crops / train --fit-all / apply: rotated crops of the selection, mean of the refiner
                  fold models (4-flip TTA), paste-back with the configured variant (e.g. refjx@0.5)

Command line (all steps):

    FILAMENT_DATA_ROOT=<MAGFiLO_1.0_Kaggle_2026> python pipeline/infer.py --weights W --work WORK --out submission.csv

From a notebook, one stage per cell (each returns quickly if its output exists):

    run = Run(weights="solution/weights", work="work")
    run.preprocess(); run.dense(); run.detect(); run.bank(); run.select(); run.refine("submission.csv")

W (the repository's weights folder):
  config.json                 {"tags": ["r34soft", "ppsoft"], "detectors": {"s1024": 1024, "m1280": 1280},
                               "folds": [0, 1, 2, 3, 4], "variant": "refjx@0.5",
                               "refiner": [["refiner", "unetpp", "resnet34"]]}
  dense/model_<tag>_f<k>[.fp16].pt   train_cv.py checkpoints or their export_fp16.py exports
  det/det_<det>_f<k>_last.pt         YOLO11-seg weights
  scorer/                            submit.py --save-model output (config.json, h_0.txt, m_0.txt)
  refiner/refiner_f<k>.pt            refine.py train --save-models output (one dir per member)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Run:
    """One inference run: the weights folder, a work folder for every intermediate product, and the six stages."""

    def __init__(self, weights, work, workers=None, force=False):
        if "FILAMENT_DATA_ROOT" not in os.environ:
            raise SystemExit("set FILAMENT_DATA_ROOT to the MAGFiLO_1.0_Kaggle_2026 folder")
        self.weights, self.work = Path(weights).resolve(), Path(work).resolve()
        self.cfg = json.loads((self.weights / "config.json").read_text())
        self.tags, self.detectors, self.folds = self.cfg["tags"], self.cfg["detectors"], self.cfg["folds"]
        if self.folds != list(range(5)):
            raise SystemExit("the bank, scorer and refiner average the five cv5 folds; a fold subset is not wired yet")
        self.workers, self.force = workers or os.cpu_count() or 4, force
        self.work.mkdir(parents=True, exist_ok=True)
        self.env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.cache, self.maps, self.dets = self.work / "cache", self.work / "maps", self.work / "dets"
        self.scores = self.work / "test_scores.npz"
        self.crops = self.work / "rcrops_test"
        self.common = ["--maps", self.maps, "--dets", self.dets, "--disks", self.cache / "disk.json",
                       "--tags", ",".join(self.tags), "--detectors", ",".join(self.detectors)]
        self.minutes = {}

    def _run(self, stage, cmd, done=None):
        """Run one command unless its output `done` already exists (--force redoes it)."""
        if done is not None and Path(done).exists() and not self.force:
            log(f"{stage}: skip (exists) {done}")
            return
        log(f"{stage}: " + " ".join(str(c) for c in cmd))
        t0 = time.time()
        subprocess.run([sys.executable, *[str(c) for c in cmd]], env=self.env, check=True)
        self.minutes[stage] = self.minutes.get(stage, 0) + (time.time() - t0) / 60
        log(f"{stage}: done in {(time.time() - t0) / 60:.1f} min")

    def preprocess(self):
        """Limb fit, flat field and radius map for the test images (an empty train folder skips the training set)."""
        (self.work / "no_train").mkdir(exist_ok=True)
        self._run("preprocess", [HERE / "preprocess.py", "--train-images", self.work / "no_train",
                                 "--cache-dir", self.cache, "--workers", self.workers], self.cache / "disk.json")

    @staticmethod
    def n_gpus():
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        except FileNotFoundError:
            return 1
        return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))

    def dense(self):
        """Probability maps of every dense checkpoint (tiled, D4 TTA, fp16 autocast); with several GPUs, one
        dense_infer process per GPU takes every n-th checkpoint (each map is computed exactly as on one GPU)."""
        ckpts = [c for c in sorted((self.weights / "dense").glob("model_*_f*.pt"))
                 if any(c.name.startswith(f"model_{t}_f") for t in self.tags)]
        if len(ckpts) != len(self.tags) * len(self.folds):
            raise SystemExit(f"expected {len(self.tags) * len(self.folds)} dense checkpoints for {self.tags}, "
                             f"found {[c.name for c in ckpts]}")
        done = {c: self.maps / f"test_{c.name.split('.')[0][len('model_'):]}" / ".done" for c in ckpts}
        todo = [c for c in ckpts if self.force or not done[c].exists()]
        if not todo:
            log("dense: skip (all maps exist)")
            return
        n = min(self.n_gpus(), len(todo))
        log(f"dense: {len(todo)} checkpoints on {n} GPU(s)")
        t0 = time.time()
        procs = []
        for g in range(n):
            share = todo[g::n]
            cmd = [sys.executable, str(HERE / "dense_infer.py"), "--ckpts", ",".join(str(c) for c in share),
                   "--out", str(self.maps), "--cache-dir", str(self.cache)]
            procs.append((share, subprocess.Popen(cmd, env={**self.env, "CUDA_VISIBLE_DEVICES": str(g)})))
        failed = [s for s, pr in procs if pr.wait() != 0]
        if failed:
            raise SystemExit(f"dense_infer failed for {[c.name for s in failed for c in s]}")
        for c in todo:
            done[c].touch()
        self.minutes["dense"] = self.minutes.get("dense", 0) + (time.time() - t0) / 60
        log(f"dense: done in {(time.time() - t0) / 60:.1f} min")

    def detect(self):
        """YOLO11-seg instances of every detector checkpoint, kept down to confidence 0.05."""
        self.dets.mkdir(exist_ok=True)
        for det, imgsz in self.detectors.items():
            for k in self.folds:
                out = self.dets / f"dets_{det}_f{k}_test.json"
                self._run("detect", [HERE / "detector.py", "predict", "--weights",
                                     self.weights / "det" / f"det_{det}_f{k}_last.pt", "--subset", "test",
                                     "--imgsz", imgsz, "--device", "0", "--cache-dir", self.cache, "--out", out], out)

    def bank(self):
        """Candidate bank: component tree of the averaged map, detector-carved pieces, per-fold-averaged descriptors."""
        self._run("bank", [HERE / "bank.py", "--mode", "test", *self.common, "--workers", self.workers,
                           "--out", self.work / "bank_test"], self.work / "bank_test" / "stems.json")

    def select(self):
        """Stage-1 selection: saved hurdle boosters, exact tree DP at the OOF tau; stage-1 CSV + score dump."""
        self._run("select", [HERE / "submit.py", "--model", self.weights / "scorer", "--test-bank",
                             self.work / "bank_test", *self.common, "--scores-out", self.scores,
                             "--out", self.work / "submission_stage1.csv"], self.scores)

    def refine(self, out):
        """Refiner: rotated crops of the selection, mean of each member's fold models, paste-back -> final CSV."""
        test = ["--mode", "test", "--bank", self.work / "bank_test", "--scores", self.scores, *self.common,
                "--workers", self.workers]
        self._run("refine", [HERE / "refine.py", "crops", *test, "--out", self.crops], self.crops / ".done")
        (self.crops / ".done").touch()
        preds = []
        for member, arch, encoder in self.cfg["refiner"]:
            pred = self.work / f"preds_{member}"
            self._run("refine", [HERE / "refine.py", "train", "--crops", self.crops, "--out", pred, "--fit-all",
                                 "--models", self.weights / member, "--arch", arch, "--encoder", encoder],
                      pred / ".done")
            (pred / ".done").touch()
            preds.append(str(pred))
        self._run("refine", [HERE / "refine.py", "apply", *test, "--crops", self.crops, "--preds", ",".join(preds),
                             "--variant", self.cfg["variant"], "--out", Path(out).resolve()])

    STAGES = ("preprocess", "dense", "detect", "bank", "select", "refine")

    def all(self, out, until="refine"):
        """Every stage in order, stopping after `until` (select = the stage-1 CSV only)."""
        for name in self.STAGES[:self.STAGES.index(until) + 1]:
            if name == "refine":
                self.refine(out)
            else:
                getattr(self, name)()
        log("minutes per stage: " + ", ".join(f"{k} {v:.1f}" for k, v in self.minutes.items()))
        log(f"wrote {out if until == 'refine' else self.work / 'submission_stage1.csv' if until == 'select' else self.work}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="weights folder (see the layout above)")
    ap.add_argument("--work", required=True, help="scratch folder for the cache, maps, detections, bank and crops")
    ap.add_argument("--out", required=True, help="submission CSV")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="redo every step")
    ap.add_argument("--until", choices=Run.STAGES, default="refine",
                    help="stop after this stage (select: the stage-1 CSV is WORK/submission_stage1.csv)")
    args = ap.parse_args()
    Run(args.weights, args.work, args.workers, args.force).all(args.out, args.until)


if __name__ == "__main__":
    main()
