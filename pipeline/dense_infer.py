"""Dense-map inference shared by training (train_cv.py) and the inference package.

Tiles of 512 with 128 overlap (Hann blend), the eight dihedral orientations,
under CUDA fp16 autocast; per observation it saves p = sigmoid(mean logit) and
s = std over orientations of the per-orientation probability, both uint8
(p*255, min(s,0.25)*1020), as <out>/<stem>.npz; with a spine head also
q = sigmoid(mean spine logit), uint8 q*255. Non-finite logits abort instead of
being written as zeros.

    python pipeline/dense_infer.py --ckpts model_r34soft_f0.pt,model_ppsoft_f0.pt --out MAPS [--subset test]

writes MAPS/<subset>_<tag>_f<k>/<stem>.npz, the layout bank.py, submit.py and refine.py read. Checkpoints are
train_cv.py's (fp32) or export_fp16.py's (fp16 Conv/Linear weights, upcast on load: identical maps under autocast).
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filcore import paths  # noqa: E402
from filcore.limb import SolarDisk, load_disks  # noqa: E402

#: the eight orientations of the dihedral group D4 as (mirror left-right, mirror top-bottom, transpose), transpose
#: slowest and top-bottom fastest; the logits of all eight are averaged in this order
ORIENTATIONS = [(lr, ud, tr) for tr, lr, ud in itertools.product((False, True), repeat=3)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_model(arch: str, encoder: str, weights: str | None, classes: int = 1):
    """U-Net or UNet++ (segmentation_models_pytorch) over `encoder`, 2 input channels (flat field, radius),
    `classes` logit outputs, no activation."""
    import segmentation_models_pytorch as smp
    cls = smp.Unet if arch == "unet" else smp.UnetPlusPlus
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=2, classes=classes, activation=None)


def network_input(flat_dir, stem: str, disk: SolarDisk) -> np.ndarray | None:
    """(2, H, W) float32: the cached flat field as png / 255, and the distance from the disk centre in radii
    (None when the flat field is not cached)."""
    flat = cv2.imread(str(Path(flat_dir) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if flat is None:
        return None
    return np.stack([flat.astype(np.float32) / 255.0, disk.rho_map(flat.shape).astype(np.float32)], axis=0)


def orient(a: np.ndarray, lr: bool, ud: bool, tr: bool) -> np.ndarray:
    """One D4 orientation of the last two axes: transpose, then mirror left-right, then top-bottom."""
    if tr:
        a = np.swapaxes(a, -1, -2)
    if lr:
        a = a[..., ::-1]
    if ud:
        a = a[..., ::-1, :]
    return np.ascontiguousarray(a)


def unorient(a: np.ndarray, lr: bool, ud: bool, tr: bool) -> np.ndarray:
    """The inverse of orient: the same steps in reverse order."""
    if ud:
        a = a[..., ::-1, :]
    if lr:
        a = a[..., ::-1]
    if tr:
        a = np.swapaxes(a, -1, -2)
    return a


def hann2d(n: int) -> np.ndarray:
    """Separable Hann window of n x n (all ones for n = 1, where numpy's window is zero), float32."""
    w = np.hanning(n)
    if w.sum() == 0:
        w = np.ones(n)
    return np.outer(w, w).astype(np.float32)


def tile_origins(extent: int, tile: int, stride: int) -> list[int]:
    """Tile starts along one axis: every `stride` px from 0, plus one tile flush with the far edge."""
    last = max(extent - tile, 0)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != extent - tile:
        starts.append(last)
    return starts


@torch.no_grad()
def tiled_logits(model, x: np.ndarray, tile: int = 512, overlap: int = 128, device=None,
                 batch_size: int = 4) -> np.ndarray:
    """(C, H, W) logits of a full frame from overlapping tiles, each weighted by a 2-D Hann window so that tile
    borders, where the network lacks context, count least. Tiles run row by row, `batch_size` per forward pass."""
    model.eval()
    _, height, width = x.shape
    stride = max(tile - overlap, 1)
    window = hann2d(tile)
    spots = [(y, x0) for y in tile_origins(height, tile, stride) for x0 in tile_origins(width, tile, stride)]
    acc, weight = None, np.zeros((height, width), np.float32)
    for b in range(0, len(spots), batch_size):
        chunk = spots[b:b + batch_size]
        batch = np.stack([x[:, y:y + tile, x0:x0 + tile] for y, x0 in chunk], axis=0)
        out = model(torch.from_numpy(batch).to(device)).float().cpu().numpy()
        if acc is None:
            acc = np.zeros((out.shape[1], height, width), np.float32)
        for (y, x0), o in zip(chunk, out):
            acc[:, y:y + tile, x0:x0 + tile] += o * window
            weight[y:y + tile, x0:x0 + tile] += window
    weight[weight == 0] = 1.0
    return acc / weight


@torch.no_grad()
def predict_maps(model, stems, flat_dir, disks, out_dir: Path, device, batch_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    t0 = time.time()
    for n, stem in enumerate(stems, start=1):
        path = out_dir / f"{stem}.npz"
        if path.exists():
            continue
        x = network_input(flat_dir, stem, disks[stem])
        if x is None:
            raise FileNotFoundError(f"no cached flat image for {stem}")
        probs = []
        logit_sum = None
        for orientation in ORIENTATIONS:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lg = tiled_logits(model, orient(x, *orientation), tile=512, overlap=128,
                                  device=device, batch_size=batch_size)
            lg = unorient(lg, *orientation).astype(np.float32)
            logit_sum = lg.copy() if logit_sum is None else logit_sum + lg
            probs.append((1.0 / (1.0 + np.exp(-np.clip(lg[0], -40.0, 40.0)))).astype(np.float16))
        mean = logit_sum / len(probs)
        if not np.isfinite(mean).all():
            raise SystemExit(f"non-finite logits for {stem} ({out_dir.name}): refusing to write maps")
        # clipping at +-40 only silences exp overflow: every stored value rounds to the same uint8 / float16
        sig = lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -40.0, 40.0)))
        p = sig(mean[0])
        s = np.stack(probs).astype(np.float32).std(axis=0)
        u8 = lambda a: np.clip(np.round(a * 255), 0, 255).astype(np.uint8)
        extra = {"q": u8(sig(mean[1]))} if mean.shape[0] > 1 else {}
        np.savez_compressed(path, p=u8(p), s=np.clip(np.round(np.minimum(s, 0.25) * 1020), 0, 255).astype(np.uint8),
                            **extra)
        if n % 20 == 0 or n == len(stems):
            log(f"predict {out_dir.name}: {n}/{len(stems)} ({(time.time() - t0) / n:.1f}s/obs)")


def load_checkpoint(path, device):
    """A train_cv.py checkpoint (fp32) or its export_fp16.py export, as an fp32 model in eval mode."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(ck["arch"], ck["encoder"], None, classes=ck.get("classes", 1))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated model_<tag>_f<k>[.fp16].pt files")
    ap.add_argument("--out", required=True, help="maps root; writes <out>/<subset>_<tag>_f<k>/")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--cache-dir", default=str(paths.CACHE), help="preprocess.py cache (flat/ + disk.json)")
    ap.add_argument("--images", default=str(paths.TEST_IMAGES), help="the frames to predict (their stems)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda")
    cache = Path(args.cache_dir)
    disks = load_disks(cache / "disk.json")
    stems = paths.image_stems(args.images)
    missing = [s for s in stems if s not in disks]
    if missing:
        raise SystemExit(f"{len(missing)} observations missing from {cache / 'disk.json'}, e.g. {missing[:3]}")
    for path in args.ckpts.split(","):
        m = re.fullmatch(r"model_(.+)_f(\d+)(\.fp16)?\.pt", Path(path).name)
        if not m:
            raise SystemExit(f"cannot read tag/fold from {path} (expected model_<tag>_f<k>[.fp16].pt)")
        out = Path(args.out) / f"{args.subset}_{m.group(1)}_f{m.group(2)}"
        log(f"{Path(path).name} -> {out} ({len(stems)} observations)")
        predict_maps(load_checkpoint(path, device), stems, cache / "flat", disks, out, device, args.batch)


if __name__ == "__main__":
    main()
