"""Train one fold's dense model; write its out-of-fold and test probability maps.

Reads preprocess.py's cache (--cache-dir: flat/, mask/ or consensus/, disk.json) and runs on one GPU. Recipe: a
U-Net (or --arch unetpp: UNet++) over an ImageNet-initialised ResNet34 with 2 input channels (flat field, radius),
512 px filament-biased crops (filcore.crops), AdamW with a cosine schedule, mixed precision, the last epoch kept
(no per-epoch validation). Options:

* --target consensus: the per-pixel annotator share of the frame instead of one annotator's view;
* --dice-weight: weight of the soft-Dice term next to BCE;
* --ema: exponential moving average of the weights, used for inference;
* --spine-weight w > 0: a second output channel trained on the annotators'
  spine polylines (each view's spines drawn --spine-radius px wide, averaged
  over the observation's views like the consensus mask), loss = loss(mask) +
  w * loss(spine). Any timm encoder works through smp's "tu-" prefix
  (e.g. tu-convnext_tiny).

Non-finite guard (cnxsoft lost all five folds to an fp16 forward overflow):
a micro-batch whose loss is not finite is dropped and the BatchNorm running
stats it touched are restored, so neither the weights, the EMA nor the
GradScaler's scale see it; more than --max-bad such micro-batches in one epoch
aborts the run (exit 3). --clip-grad clips the unscaled gradient norm.
Inference refuses to write maps from non-finite logits.

Inference (dense_infer.py): tiles of 512 with 128 overlap (Hann blend), the
eight dihedral orientations; maps p, s (and q with a spine head) as
<out>/<subset>_<tag>_f<k>/<stem>.npz.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dense_infer import log, make_model, predict_maps  # inference (tiling, D4 TTA, map files) lives there
from filcore import paths
from filcore.coco import Coco, one_view_per_stem
from filcore.crops import CropSampler
from filcore.limb import load_disks

# filcore.crops.AUGMENT (flips, 90-degree rotations, brightness/contrast on the image only) with the spine target
_AUGMENT_SPINE = A.Compose(
    [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=1.0),
     A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5)],
    additional_targets={"radius": "mask", "spine": "mask"},
)


class SoftDiceBCE(torch.nn.Module):
    """BCE with logits + dice_weight * (1 - soft Dice); Dice per sample, smoothed by +1, averaged over the batch."""

    def __init__(self, dice_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight, self.smooth = dice_weight, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        prob, truth = torch.sigmoid(logits).flatten(1), target.flatten(1)
        inter = (prob * truth).sum(dim=1)
        extra, missed = prob.sum(dim=1) - inter, truth.sum(dim=1) - inter
        dice = (2.0 * inter + self.smooth) / (2.0 * inter + extra + missed + self.smooth)
        return self.dice_weight * (1.0 - dice.mean()) + bce


class SpineCrops(CropSampler):
    """CropSampler (consensus target) that also returns the consensus spine target of the same crop."""

    def __init__(self, *args, spine_radius: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.spine_radius = spine_radius
        views = self.coco.by_stem()
        self._spines = {}  # stem -> list over views of [polyline (N, 2) float]
        for stem in self.coco.stems_of(self.view_ids):
            self._spines[stem] = [[np.asarray(a["spine"], np.float64).reshape(-1, 2)
                                   for a in self.coco.anns.get(v, []) if a.get("spine")]
                                  for v in views[stem]]

    def _spine_crop(self, stem, y0, x0, h, w):
        out = np.zeros((h, w), np.float32)
        for lines in self._spines[stem]:
            canvas = np.zeros((h, w), np.uint8)
            pts = [np.round((ln - (x0, y0)) * 16).astype(np.int32) for ln in lines]
            if pts:
                cv2.polylines(canvas, pts, False, 1, thickness=2 * self.spine_radius + 1, lineType=cv2.LINE_8, shift=4)
            out += canvas
        return out / max(len(self._spines[stem]), 1)

    def __getitem__(self, index):
        vid = self.view_ids[index // self.per_view]
        view = self.coco.views[vid]
        stem = view.stem
        rng = self.rng()
        flat_u8 = self.flat(stem)
        y0, x0 = self.origin(rng, vid, stem, (view.height, view.width))
        size = self.crop
        flat_crop = flat_u8[y0:y0 + size, x0:x0 + size].astype(np.float32) / 255.0
        h, w = flat_crop.shape
        radius_crop = self.disks[stem].rho(y0, x0, h, w).astype(np.float32)
        mask_crop = self.target_map(vid, stem)[y0:y0 + size, x0:x0 + size].astype(np.float32)
        spine_crop = self._spine_crop(stem, y0, x0, h, w)
        if self.augment:
            o = _AUGMENT_SPINE(image=flat_crop, mask=mask_crop, radius=radius_crop, spine=spine_crop)
            flat_crop, mask_crop, radius_crop, spine_crop = o["image"], o["mask"], o["radius"], o["spine"]
        x = np.stack([flat_crop, radius_crop], axis=0).astype(np.float32)
        y = np.stack([mask_crop, spine_crop], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-file", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", choices=["unet", "unetpp"], default="unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--target", choices=["annotator", "consensus"], default="annotator")
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay; 0 = off")
    ap.add_argument("--spine-weight", type=float, default=0.0, help="weight of the spine head's loss; 0 = no head")
    ap.add_argument("--spine-radius", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--decoder-lr", type=float, default=None,
                    help="separate LR for everything outside model.encoder (default: --lr for all)")
    ap.add_argument("--clip-grad", type=float, default=0.0, help="clip the unscaled gradient norm; 0 = off")
    ap.add_argument("--max-bad", type=int, default=50, help="abort after this many non-finite micro-batches in an epoch")
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--crops-per-image", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predict-batch", type=int, default=8)
    ap.add_argument("--cache-dir", default=str(paths.CACHE), help="preprocess.py cache")
    ap.add_argument("--annotations", default=str(paths.TRAIN_COCO))
    ap.add_argument("--test-images", default=str(paths.TEST_IMAGES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-train", action="store_true", help="predict with an existing checkpoint")
    args = ap.parse_args()
    spine = args.spine_weight > 0
    if spine and args.target != "consensus":
        raise SystemExit("--spine-weight needs --target consensus")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / f"model_{args.tag}_f{args.fold}.pt"

    coco = Coco(args.annotations)
    fold_of = json.loads(Path(args.folds_file).read_text())["fold_of_stem"]
    pool = sorted(v for v, view in coco.views.items() if fold_of[view.stem] != args.fold)
    held = sorted(s for s, k in fold_of.items() if k == args.fold)
    test_stems = paths.image_stems(args.test_images)
    cache_dir = Path(args.cache_dir)
    disks = load_disks(cache_dir / "disk.json")
    missing = [s for s in held + test_stems if s not in disks]
    if missing:
        raise SystemExit(f"{len(missing)} observations missing from the preprocessing cache, e.g. {missing[:3]}")

    model = make_model(args.arch, args.encoder, "imagenet", classes=2 if spine else 1).to(device)
    # smoke-test the inference path on one observation before hours of training
    smoke = out / f"_smoke_{args.tag}_f{args.fold}"
    predict_maps(model, held[:1], cache_dir / "flat", disks, smoke, device, args.predict_batch)
    log(f"inference smoke test ok: {[p.name for p in smoke.glob('*.npz')]}")
    import shutil
    shutil.rmtree(smoke)
    if not args.skip_train:
        # one annotator view per observation for the annotator target; CropSampler keeps one view per
        # observation by itself for the consensus target
        train_ids = one_view_per_stem(coco, pool, seed=args.seed) if args.target == "annotator" else pool
        kw = dict(cache_dir=cache_dir, crop=args.crop_size, per_view=args.crops_per_image, augment=True,
                  seed=args.seed, target=args.target, preload=True)
        dataset = (SpineCrops(train_ids, coco, spine_radius=args.spine_radius, **kw) if spine
                   else CropSampler(train_ids, coco, **kw))
        if spine:
            yy = dataset[0][1]
            log(f"spine target check: mask mean {yy[0].mean():.4f}, spine mean {yy[1].mean():.4f}, "
                f"spine pixels inside mask {(yy[0] > 0)[yy[1] > 0].float().mean() if (yy[1] > 0).any() else float('nan'):.3f}")
        log(f"fold {args.fold} [{args.tag}]: {len(dataset.view_ids)} training views from "
            f"{len(coco.stems_of(pool))} observations; held out {len(held)}; test {len(test_stems)}")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)
        criterion = SoftDiceBCE(dice_weight=args.dice_weight)
        if args.decoder_lr is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        else:
            enc = {id(q) for q in model.encoder.parameters()}
            optimizer = torch.optim.AdamW([
                {"params": [q for q in model.parameters() if id(q) in enc], "lr": args.lr},
                {"params": [q for q in model.parameters() if id(q) not in enc], "lr": args.decoder_lr}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler(device="cuda")
        ema = copy.deepcopy(model).eval() if args.ema > 0 else None
        bufs = list(model.buffers())
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, run_loss, seen, bad, good = time.time(), 0.0, 0, 0, 0
            optimizer.zero_grad(set_to_none=True)
            for step, (x, y) in enumerate(loader, start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                saved = [b.clone() for b in bufs]
                with torch.autocast(device_type="cuda"):
                    out_ = model(x)
                    loss = criterion(out_[:, :1], y[:, :1])
                    if spine:
                        loss = loss + args.spine_weight * criterion(out_[:, 1:2], y[:, 1:2])
                if not torch.isfinite(loss):
                    with torch.no_grad():
                        for b, s_ in zip(bufs, saved):
                            b.copy_(s_)
                    bad += 1
                    if bad > args.max_bad:
                        raise SystemExit(f"epoch {epoch}: more than {args.max_bad} non-finite micro-batches; aborting")
                    continue
                scaler.scale(loss / args.accum).backward()
                good += 1  # finite micro-batches: a skipped one must not shift the accumulation window
                if good % args.accum == 0:
                    if args.clip_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        with torch.no_grad():
                            for pe, pm in zip(ema.parameters(), model.parameters()):
                                pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
                            for be, bm in zip(ema.buffers(), model.buffers()):
                                be.copy_(bm)
                run_loss += loss.item() * x.size(0)
                seen += x.size(0)
            scheduler.step()
            final = ema if ema is not None else model
            torch.save({"model": final.state_dict(), "arch": args.arch, "encoder": args.encoder,
                        "in_channels": 2, "classes": 2 if spine else 1, "spine_weight": args.spine_weight,
                        "target": args.target, "dice_weight": args.dice_weight,
                        "ema": args.ema, "epoch": epoch, "fold": args.fold}, ckpt_path)
            log(f"epoch {epoch}/{args.epochs} fold {args.fold} [{args.tag}] loss={run_loss / max(seen, 1):.4f} "
                f"({time.time() - t0:.0f}s, {seen / (time.time() - t0):.1f} crops/s)"
                + (f" skipped {bad} non-finite micro-batches" if bad else ""))
        model = ema if ema is not None else model
    else:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    predict_maps(model, held, cache_dir / "flat", disks, out / f"oof_{args.tag}_f{args.fold}", device,
                 args.predict_batch)
    predict_maps(model, test_stems, cache_dir / "flat", disks, out / f"test_{args.tag}_f{args.fold}", device,
                 args.predict_batch)
    log(f"fold {args.fold} [{args.tag}] done")


if __name__ == "__main__":
    main()
