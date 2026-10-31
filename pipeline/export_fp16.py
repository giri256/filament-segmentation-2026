"""Halve checkpoint size without changing a single output pixel under CUDA autocast.

Inference runs under torch.autocast(cuda, float16): every Conv/Linear casts its weight and bias to fp16 on the fly
(round to nearest), while normalisation layers and element-wise parameters (BatchNorm, LayerNorm, ConvNeXt layer
scale) stay fp32. So storing exactly the Conv/Linear weights and biases as fp16 and everything else as fp32 gives the
same autocast forward pass bit for bit; load_fp16 upcasts them into the fp32 model.

    python pipeline/export_fp16.py dense   --in model_r34soft_f0.pt --out model_r34soft_f0.fp16.pt
    python pipeline/export_fp16.py refiner --in refiner_f0.pt --out refiner_f0.fp16.pt --arch unetpp --encoder resnet34
    python pipeline/export_fp16.py check   --in X.pt --fp16 X.fp16.pt --kind dense|refiner [...]   # GPU parity check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HALF = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)


def build(kind, meta):
    import segmentation_models_pytorch as smp
    cls = smp.UnetPlusPlus if meta["arch"] == "unetpp" else smp.Unet
    return cls(encoder_name=meta["encoder"], encoder_weights=None, in_channels=meta["in_channels"],
               classes=meta["classes"], activation=None)


def meta_of(kind, ck, args):
    if kind == "dense":
        return {"arch": ck["arch"], "encoder": ck["encoder"], "in_channels": ck.get("in_channels", 2),
                "classes": ck.get("classes", 1)}
    return {"arch": args.arch, "encoder": args.encoder, "in_channels": args.in_channels, "classes": 1}


def half_keys(model):
    keys = set()
    for name, mod in model.named_modules():
        if isinstance(mod, HALF):
            for p in ("weight", "bias"):
                if getattr(mod, p, None) is not None:
                    keys.add(f"{name}.{p}" if name else p)
    return keys


def export(kind, src, dst, args):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    state = ck["model"] if kind == "dense" else ck
    meta = meta_of(kind, ck, args)
    model = build(kind, meta)
    model.load_state_dict(state)  # the checkpoint must match the architecture exactly
    hk = half_keys(model)
    out = {k: (v.half() if k in hk and v.is_floating_point() else v) for k, v in state.items()}
    payload = {**{k: v for k, v in ck.items() if k != "model"}, "model": out} if kind == "dense" else \
        {"model": out, **meta}
    payload["fp16_keys"] = sorted(k for k in out if k in hk)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)
    print(f"{src} -> {dst}: {len(hk)} fp16 tensors, {Path(src).stat().st_size / 1e6:.1f} MB -> "
          f"{Path(dst).stat().st_size / 1e6:.1f} MB")


def load_fp16(path, kind=None):
    """An fp32 model holding the exported weights (fp16 values upcast); use it under CUDA autocast."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = {k: ck[k] for k in ("arch", "encoder", "in_channels", "classes") if k in ck}
    meta.setdefault("in_channels", 2)
    meta.setdefault("classes", 1)
    model = build(kind, meta)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    return model


@torch.no_grad()
def check(args):
    dev = torch.device("cuda")
    ck = torch.load(args.input, map_location="cpu", weights_only=False)
    meta = meta_of(args.kind, ck, args)
    ref = build(args.kind, meta)
    ref.load_state_dict(ck["model"] if args.kind == "dense" else ck)
    new = load_fp16(args.fp16, args.kind)
    ref, new = ref.to(dev).eval(), new.to(dev).eval()
    torch.manual_seed(0)
    shape = (4, meta["in_channels"], *((512, 512) if args.kind == "dense" else (128, 384)))
    worst, noise = 0.0, 0.0
    for _ in range(args.batches):
        x = torch.rand(shape, device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            a, a2, b = ref(x).float(), ref(x).float(), new(x).float()
        noise = max(noise, float((a - a2).abs().max()))  # run-to-run noise of the fp32 model itself
        worst = max(worst, float((a - b).abs().max()))
    ok = worst <= noise
    print(f"parity {args.input}: max |logit diff| under autocast = {worst:.3e} (fp32 run-to-run {noise:.3e}) over "
          f"{args.batches} batches ({'EXACT' if worst == 0 else 'within run-to-run noise' if ok else 'NOT exact'})")
    if not ok:
        raise SystemExit(1)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["dense", "refiner", "check"])
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--out")
    ap.add_argument("--fp16")
    ap.add_argument("--kind", dest="ckind", choices=["dense", "refiner"], default="dense")
    ap.add_argument("--arch", default="unetpp")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--in-channels", type=int, default=5)
    ap.add_argument("--batches", type=int, default=3)
    args = ap.parse_args()
    if args.kind == "check":
        args.kind = args.ckind
        check(args)
    else:
        export(args.kind, args.input, args.out, args)


if __name__ == "__main__":
    main()
