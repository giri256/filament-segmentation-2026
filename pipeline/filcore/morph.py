"""Binary morphology used on probability maps and masks."""

from __future__ import annotations

import cv2
import numpy as np


def disc(radius: int) -> np.ndarray:
    """Elliptical structuring element of diameter 2 * radius + 1 (at least 1)."""
    side = max(1, 2 * radius + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))


def close(binary: np.ndarray, radius: int) -> np.ndarray:
    """Morphological closing with disc(radius) of a binary image, as 0/1 uint8 (radius 0: no closing)."""
    out = (binary > 0).astype(np.uint8)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, disc(radius)) if radius > 0 else out


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Background not connected to the top-left pixel becomes foreground (0/1 uint8 in and out). The top-left pixel
    must be background, which holds for everything off the solar disk."""
    fg = (binary > 0).astype(np.uint8)
    outside = fg.copy()
    cv2.floodFill(outside, np.zeros((fg.shape[0] + 2, fg.shape[1] + 2), np.uint8), (0, 0), 1)
    return (fg | (1 - outside)).astype(np.uint8)


def logit(p: float) -> float:
    """The logit equal to probability p, so `logits > logit(p)` is `sigmoid(logits) > p` (+-1e9 at 0 and 1)."""
    if p <= 0.0:
        return -1e9
    if p >= 1.0:
        return 1e9
    return float(np.log(p / (1.0 - p)))


def map_foreground(logits: np.ndarray, disk_mask: np.ndarray, threshold: float, close_radius: int) -> np.ndarray:
    """The map's filament pixels on the disk: probability above `threshold`, closed, holes filled (boolean)."""
    on = ((logits > logit(threshold)) & disk_mask).astype(np.uint8)
    return fill_holes(close(on, close_radius)).astype(bool) & disk_mask
