"""MAGFiLO's COCO annotation file: annotated views, their filament polygons, and the frame each view belongs to.

A frame (an observation, named by its file stem such as 20160920230134Lh) may be annotated by several annotators.
Each annotation set is its own COCO image entry, a "view", whose id carries the annotator batch as a prefix
(010401-20160920230134Lh). The organisers score a frame's predictions against each of its views and pool the counts,
so every evaluation here keeps all views; training on single-annotator targets keeps one view per frame.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import FRAME, TRAIN_COCO
from .rle import polygons_to_rle


@dataclass(frozen=True)
class View:
    view_id: str
    file_name: str
    height: int
    width: int

    @property
    def stem(self) -> str:
        return Path(self.file_name).stem

    @property
    def date(self) -> str:
        """YYYYMMDD of the observation."""
        return self.stem[:8]

    @property
    def site(self) -> str:
        """Two-letter GONG site code at the end of the stem (Bh, Ch, Lh, Mh, Th, Uh)."""
        return self.stem[14:16]


class Coco:
    """The training annotations: `views` {view id: View} and `anns` {view id: [COCO annotation dict]}."""

    def __init__(self, path: str | Path = TRAIN_COCO):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.views: dict[str, View] = {}
        for entry in raw.get("images", []):
            vid = str(entry["id"])
            self.views[vid] = View(vid, entry["file_name"], int(entry.get("height") or FRAME),
                                   int(entry.get("width") or FRAME))
        self.anns: dict[str, list[dict]] = {}
        for ann in raw.get("annotations", []):
            self.anns.setdefault(str(ann["image_id"]), []).append(ann)

    def by_stem(self) -> dict[str, list[str]]:
        """{frame stem: its view ids, sorted}, frames in sorted order."""
        groups: dict[str, list[str]] = {}
        for vid, view in self.views.items():
            groups.setdefault(view.stem, []).append(vid)
        return {stem: sorted(groups[stem]) for stem in sorted(groups)}

    def filament_rles(self, view_id: str) -> list[dict]:
        """The view's annotated filaments, one RLE each, in file order."""
        view = self.views[view_id]
        return [polygons_to_rle(ann["segmentation"], view.height, view.width)
                for ann in self.anns.get(view_id, []) if ann.get("segmentation")]

    def rles_of(self, view_ids: Iterable[str]) -> dict[str, list[dict]]:
        return {vid: self.filament_rles(vid) for vid in view_ids}

    def stems_of(self, view_ids: Iterable[str]) -> list[str]:
        return sorted({self.views[v].stem for v in view_ids})


def one_view_per_stem(coco: Coco, view_ids: Iterable[str], seed: int = 0) -> list[str]:
    """One view of each frame among `view_ids`, for single-annotator training targets. The draw is fixed by `seed`:
    frames in sorted order each pick one of their sorted views with random.Random(seed)."""
    groups: dict[str, list[str]] = {}
    for vid in view_ids:
        groups.setdefault(coco.views[vid].stem, []).append(vid)
    rng = random.Random(seed)
    picked = [rng.choice(sorted(groups[stem])) for stem in sorted(groups)]
    return sorted(picked)
