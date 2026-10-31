"""The solar disk of a frame: its limb circle, and the limb-darkening flat field.

GONG H-alpha frames show the chromosphere as a disk on dark sky, darker toward the limb (limb darkening) and often
inside a smooth bright halo (scattered light, and the area the frame covered before it was registered) whose outer
edge is a sharp circle too. Per frame:

* `fit_limb`: three first guesses, an Otsu threshold of the blurred frame, an Otsu threshold of local texture (the
  chromosphere is busy, the halo smooth) and GONG's nominal registered geometry, are each tried as given and 50, 100
  and 150 px (at 2048) further in. Every start is pulled onto the strongest radial edge near it (`_snap`: 720 rays,
  a wide search band, then a narrow one) and scored by the share of rays whose edge lies within 3 px of the fitted
  circle. The halo is always outside the limb, so the smallest circle scoring at least 80% of the best score is the
  limb; when no circle reaches 50% the Otsu circle is kept.
* `flat_field`: the frame divided by its median intensity in 128 rings of normalised radius (rings with a zero
  median interpolated, then a 9-ring box smoothing), 1.0 off the disk. Filaments then sit below 1.0 at every radius.

The method is the preprocessing of filament-seg (A. Pawar, MIT licence). This is our implementation of it. It is
written to reproduce that preprocessing to the last floating-point bit, because the networks were trained on its
output; the equivalence kernel (kernels/cv5-templates/equiv.py) checks every training and test frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .morph import disc, fill_holes

#: GONG registers the disk to about 0.439 frame widths in radius (899 px at 2048), centred
NOMINAL_RADIUS = 0.439
RAYS = 720
#: start radii tried inside each first guess, in px at 2048: the halo can reach 150 px beyond the limb
STEPS_IN = (0.0, 50.0, 100.0, 150.0)
#: a ray agrees with a circle when its edge lies within this many px (at 2048) of it
AGREE_PX = 3.0
#: circles scoring at least this share of the best score count as supported
SUPPORTED = 0.8
#: the flat field is stored as uint8: clip(flat, 0, FLAT_MAX) * 255 / FLAT_MAX (quiet Sun at 1.0, filaments
#: around 0.5-0.8, headroom for bright plage)
FLAT_MAX = 2.0


@dataclass(frozen=True)
class SolarDisk:
    cx: float
    cy: float
    radius: float

    def _dist2(self, y0: int, x0: int, height: int, width: int) -> np.ndarray:
        ys, xs = np.ogrid[y0:y0 + height, x0:x0 + width]
        return (xs - self.cx) ** 2 + (ys - self.cy) ** 2

    def mask(self, shape: tuple[int, int], shrink: float = 0.98) -> np.ndarray:
        """Pixels within shrink x radius of the centre: the outermost 2% of the disk is dropped as noisy limb."""
        return self._dist2(0, 0, *shape) <= (self.radius * shrink) ** 2

    def rho(self, y0: int, x0: int, height: int, width: int) -> np.ndarray:
        """Distance from the centre in disk radii (1.0 on the limb) over rows y0.. and columns x0.. of the frame."""
        return np.sqrt(self._dist2(y0, x0, height, width)) / max(self.radius, 1.0)

    def rho_map(self, shape: tuple[int, int]) -> np.ndarray:
        return self.rho(0, 0, *shape)


def load_disks(path: str | Path) -> dict[str, SolarDisk]:
    """preprocess.py's disk.json as {stem: SolarDisk}."""
    return {stem: SolarDisk(**g) for stem, g in json.loads(Path(path).read_text(encoding="utf-8")).items()}


def load_gray(path: str | Path) -> np.ndarray:
    """A frame as a single-channel uint8 array (the competition JPEGs are grayscale)."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read {path}")
    return gray


# --- limb circle -------------------------------------------------------------------------------------------------

def fit_limb(gray: np.ndarray) -> SolarDisk:
    """The limb circle of a uint8 frame (see the module notes)."""
    s = min(gray.shape) / 2048.0
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(1.5 * s, 0.8))
    guesses = [_otsu_circle(gray), _texture_circle(gray), _nominal_circle(gray)]
    scored = []
    for guess in guesses:
        if guess is None:
            continue
        for step in STEPS_IN:
            start = SolarDisk(guess.cx, guess.cy, guess.radius - step * s)
            if start.radius <= 0:
                continue
            wide, _ = _snap(smooth, start, inner=60.0, outer=20.0)
            scored.append(_snap(smooth, wide, inner=10.0, outer=10.0))  # the centre settles once r is close
    top = max((score for _, score in scored), default=0.0)
    if top < 0.5:
        return guesses[0]
    return min((circle for circle, score in scored if score >= SUPPORTED * top), key=lambda c: c.radius)


def _snap(smooth: np.ndarray, start: SolarDisk, inner: float, outer: float) -> tuple[SolarDisk, float]:
    """Move `start` onto the strongest radial edge between `inner` px inside and `outer` px outside it (px at 2048).

    Each ray's intensity profile is sampled at 1 px steps; its edge is the largest absolute gradient, with a
    parabola through the peak and its neighbours for the sub-pixel position. A peak at either end of the band is
    the limb-darkening slope rising toward it, not an edge, and is ignored, as are samples within 1 px of the frame
    border. The edge points get a robust circle fit; returns it and the share of all rays agreeing with it.
    """
    height, width = smooth.shape
    s = min(height, width) / 2048.0
    angle = np.linspace(0.0, 2.0 * np.pi, RAYS, endpoint=False)
    radii = np.arange(start.radius - inner * s, start.radius + outer * s + 1.0, 1.0)
    px = (start.cx + radii[None, :] * np.cos(angle)[:, None]).astype(np.float32)
    py = (start.cy + radii[None, :] * np.sin(angle)[:, None]).astype(np.float32)
    profile = cv2.remap(smooth, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    grad = np.abs(np.gradient(profile, axis=1))
    grad[~((px >= 1) & (px <= width - 2) & (py >= 1) & (py <= height - 2))] = -1.0
    peak = np.argmax(grad, axis=1)
    usable = (grad[np.arange(RAYS), peak] > 0) & (peak > 1) & (peak < radii.size - 2)
    if usable.sum() < 3:
        return start, 0.0
    rows, peak = np.flatnonzero(usable), peak[usable]
    left, mid, right = grad[rows, peak - 1], grad[rows, peak], grad[rows, peak + 1]
    bend = left - 2.0 * mid + right
    offset = np.divide(0.5 * (left - right), bend, out=np.zeros_like(mid), where=bend < 0)
    r = radii[peak] + np.clip(offset, -0.5, 0.5)
    xs = start.cx + r * np.cos(angle[usable])
    ys = start.cy + r * np.sin(angle[usable])
    circle, _ = _robust_circle(xs, ys)
    off = np.hypot(xs - circle.cx, ys - circle.cy) - circle.radius
    return circle, float((np.abs(off) <= max(AGREE_PX * s, 1.0)).sum()) / RAYS


def _robust_circle(xs: np.ndarray, ys: np.ndarray, rounds: int = 4) -> tuple[SolarDisk, np.ndarray]:
    """Algebraic (Kasa) least-squares circle, refitted `rounds` times on the points within max(3 MAD, 2 px) of the
    previous fit. Returns the circle and the final inlier mask."""
    inlier = np.ones(xs.size, dtype=bool)
    circle = SolarDisk(0.0, 0.0, 1.0)
    for _ in range(rounds):
        n = int(inlier.sum())
        if n < 3:
            break
        x, y = xs[inlier], ys[inlier]
        # x^2 + y^2 + D x + E y + F = 0, centre (-D/2, -E/2)
        d, e, f = np.linalg.lstsq(np.column_stack([x, y, np.ones(n)]), -(x ** 2 + y ** 2), rcond=None)[0]
        cx, cy = -d / 2.0, -e / 2.0
        circle = SolarDisk(float(cx), float(cy), float(np.sqrt(max(cx * cx + cy * cy - f, 1.0))))
        off = np.hypot(xs - circle.cx, ys - circle.cy) - circle.radius
        mad = 1.4826 * np.median(np.abs(off[inlier] - np.median(off[inlier])))
        inlier = np.abs(off) <= max(3.0 * mad, 2.0)
    return circle, inlier


def _largest_region(binary: np.ndarray) -> np.ndarray | None:
    """The largest 8-connected foreground component as a 0/1 uint8 image (None without foreground)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def _nominal_circle(gray: np.ndarray) -> SolarDisk:
    height, width = gray.shape
    return SolarDisk((width - 1) / 2.0, height / 2.0, min(height, width) * NOMINAL_RADIUS)


def _texture_circle(gray: np.ndarray) -> SolarDisk | None:
    """Circle through the outline of the textured region (fine structure smoothed over ~15 px, Otsu threshold,
    largest component with its holes filled)."""
    s = min(gray.shape) / 2048.0
    f = gray.astype(np.float32)
    busy = cv2.GaussianBlur(np.abs(f - cv2.GaussianBlur(f, (0, 0), max(3.0 * s, 1.0))), (0, 0), max(15.0 * s, 2.0))
    gain = 255.0 / max(float(busy.max()), 1e-6)
    _, on = cv2.threshold(np.clip(busy * gain, 0, 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    region = _largest_region(on)
    if region is None:
        return None
    outlines, _ = cv2.findContours(fill_holes(region), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(outlines, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(pts) < 3:
        return None
    return _robust_circle(pts[:, 0], pts[:, 1])[0]


def _otsu_circle(gray: np.ndarray) -> SolarDisk:
    """Smallest circle enclosing the largest bright region (Otsu threshold after a sigma-5 blur, 31 px closing).
    The halo usually passes the threshold too, so this radius is an upper bound."""
    _, on = cv2.threshold(cv2.GaussianBlur(gray, (0, 0), sigmaX=5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    region = _largest_region(cv2.morphologyEx(on, cv2.MORPH_CLOSE, disc(15)))
    if region is None:
        height, width = gray.shape
        return SolarDisk(width / 2, height / 2, min(height, width) * 0.45)
    outlines, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx, cy), r = cv2.minEnclosingCircle(max(outlines, key=cv2.contourArea))
    return SolarDisk(float(cx), float(cy), float(r))


# --- flat field --------------------------------------------------------------------------------------------------

def ring_profile(gray: np.ndarray, disk: SolarDisk, rings: int = 128, box: int = 9,
                 rho: np.ndarray | None = None) -> np.ndarray:
    """Median intensity per ring of normalised radius in [0, 1]; rings with a zero median are interpolated from
    their neighbours (empty rings keep 1.0), then a `box`-ring moving average."""
    rho = disk.rho_map(gray.shape) if rho is None else rho
    on_disk = rho <= 1.0
    ring = np.clip((rho[on_disk] * rings).astype(np.int32), 0, rings - 1)
    value = gray[on_disk].astype(np.float32)
    profile = np.ones(rings, dtype=np.float32)
    if ring.size:
        order = np.argsort(ring, kind="stable")
        ring, value = ring[order], value[order]
        cuts = np.flatnonzero(np.diff(ring)) + 1
        for k, part in zip(ring[np.r_[0, cuts]], np.split(value, cuts)):
            profile[k] = np.median(part)
    positive = profile > 0
    if positive.any():
        profile = np.interp(np.arange(rings), np.flatnonzero(positive), profile[positive])
    if box > 1:
        profile = np.convolve(profile, np.ones(box, dtype=np.float32) / box, mode="same")
    profile[profile <= 0] = 1.0
    return profile


def flat_field(gray: np.ndarray, disk: SolarDisk, rings: int = 128) -> np.ndarray:
    """The frame divided by its ring profile (1.0 = typical quiet Sun at that radius), 1.0 off the disk."""
    rho = disk.rho_map(gray.shape)
    profile = ring_profile(gray, disk, rings, rho=rho)
    expected = profile[np.clip((rho * rings).astype(np.int32), 0, rings - 1)]
    flat = gray.astype(np.float32) / np.maximum(expected, 1e-3)
    flat[rho > 1.0] = 1.0
    return flat


def flat_to_u8(flat: np.ndarray) -> np.ndarray:
    """The stored flat field: [0, FLAT_MAX] packed into a byte, read back by the networks as png / 255."""
    return (np.clip(flat, 0.0, FLAT_MAX) * (255.0 / FLAT_MAX)).astype(np.uint8)
