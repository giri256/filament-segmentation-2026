"""Training crops for the dense networks, read from preprocess.py's cache.

Filaments cover about 0.2% of the disk, so uniform crops would be almost all empty: with probability `on_filament`
a crop is centred on the bounding-box centre of one of the view's filaments, jittered by up to a quarter crop each
way, otherwise on a point drawn uniformly over the disk area. Only exact transforms are used for augmentation
(flips, 90-degree rotations, and brightness/contrast on the image channel alone): resampling would blur structures a
few pixels wide. The input has two channels, the flat field (png / 255) and the normalised radius.

Targets: "annotator" (the view's own filament mask), "consensus" (the share of the frame's annotators marking each
pixel) or "union" (any annotator). Consensus and union belong to the frame, so those targets keep one view per frame.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset

from .coco import Coco, one_view_per_stem
from .limb import load_disks

#: "radius" is declared a mask so it is flipped and rotated with the image but never brightness-jittered
AUGMENT = A.Compose(
    [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=1.0),
     A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5)],
    additional_targets={"radius": "mask"},
)
TARGETS = ("annotator", "consensus", "union")


class CropSampler(Dataset):
    """`per_view` random crops of `crop` px per training view and epoch; item i belongs to view i // per_view."""

    def __init__(self, view_ids: Sequence[str], coco: Coco, cache_dir: str | Path, crop: int = 512,
                 per_view: int = 8, on_filament: float = 0.7, augment: bool = True, seed: int = 0,
                 target: str = "annotator", preload: bool = False):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, not {target!r}")
        self.coco, self.cache = coco, Path(cache_dir)
        self.crop, self.per_view, self.on_filament = crop, per_view, on_filament
        self.augment, self.seed, self.target = augment, seed, target
        view_ids = list(view_ids)
        self.view_ids = one_view_per_stem(coco, view_ids, seed=seed) if target != "annotator" else view_ids
        self.disks = load_disks(self.cache / "disk.json")
        self.centres = {vid: [self._bbox_centre(rle) for rle in coco.filament_rles(vid)] for vid in self.view_ids}
        self._recent: OrderedDict[str, np.ndarray] = OrderedDict()  # small LRU of decoded flat fields
        self._all = {stem: self._read_flat(stem) for stem in coco.stems_of(self.view_ids)} if preload else {}
        self._rngs: dict[int, random.Random] = {}

    @staticmethod
    def _bbox_centre(rle: dict) -> tuple[float, float]:
        x, y, w, h = coco_mask.toBbox(rle)
        return float(y + h / 2.0), float(x + w / 2.0)

    def __len__(self) -> int:
        return len(self.view_ids) * self.per_view

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        vid = self.view_ids[index // self.per_view]
        view = self.coco.views[vid]
        rng = self.rng()
        flat = self.flat(view.stem)
        y0, x0 = self.origin(rng, vid, view.stem, (view.height, view.width))
        n = self.crop
        image = flat[y0:y0 + n, x0:x0 + n].astype(np.float32) / 255.0
        rho = self.disks[view.stem].rho(y0, x0, *image.shape).astype(np.float32)
        target = self.target_map(vid, view.stem)[y0:y0 + n, x0:x0 + n].astype(np.float32)
        if self.augment:
            out = AUGMENT(image=image, mask=target, radius=rho)
            image, target, rho = out["image"], out["mask"], out["radius"]
        x = np.stack([image, rho], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(target[None, :, :].astype(np.float32))

    def rng(self) -> random.Random:
        """One generator per DataLoader worker (seed + worker id + 1; seed in the main process), created lazily, so
        workers forked from one dataset do not replay the same crops."""
        info = torch.utils.data.get_worker_info()
        key = 0 if info is None else info.id + 1
        if key not in self._rngs:
            self._rngs[key] = random.Random(self.seed + key)
        return self._rngs[key]

    def origin(self, rng: random.Random, vid: str, stem: str, shape: tuple[int, int]) -> tuple[int, int]:
        """Top-left corner of the next crop."""
        height, width = shape
        n = self.crop
        centres = self.centres[vid]
        if centres and rng.random() < self.on_filament:
            cy, cx = rng.choice(centres)
            cy += rng.uniform(-n / 4.0, n / 4.0)
            cx += rng.uniform(-n / 4.0, n / 4.0)
        else:  # uniform over the disk area: sqrt of a uniform radius fraction
            disk = self.disks[stem]
            angle = rng.uniform(0.0, 2.0 * np.pi)
            r = disk.radius * (rng.random() ** 0.5)
            cy = disk.cy + r * np.sin(angle)
            cx = disk.cx + r * np.cos(angle)
        return (int(np.clip(cy - n / 2.0, 0, max(height - n, 0))),
                int(np.clip(cx - n / 2.0, 0, max(width - n, 0))))

    def flat(self, stem: str) -> np.ndarray:
        if stem in self._all:
            return self._all[stem]
        if stem in self._recent:
            self._recent.move_to_end(stem)
            return self._recent[stem]
        flat = self._recent[stem] = self._read_flat(stem)
        while len(self._recent) > 8:
            self._recent.popitem(last=False)
        return flat

    def _read_flat(self, stem: str) -> np.ndarray:
        path = self.cache / "flat" / f"{stem}.png"
        flat = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if flat is None:
            raise FileNotFoundError(f"{path} missing: run preprocess.py first")
        return flat

    def target_map(self, vid: str, stem: str) -> np.ndarray:
        path = self.cache / ("mask" if self.target == "annotator" else "consensus") / \
            f"{vid if self.target == 'annotator' else stem}.png"
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:  # every training view has filaments: a missing file is a cache fault, never "no filament"
            raise FileNotFoundError(f"{path} missing: run preprocess.py first")
        if self.target == "consensus":
            return m.astype(np.float32) / 255.0
        return (m > 0).astype(np.uint8)
