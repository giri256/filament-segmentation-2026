"""Shared building blocks of the pipeline: data locations, the COCO annotations, the solar-disk geometry and flat
field, mask encoding and the submission file, binary morphology, detection handling, and the PQ bookkeeping.

    paths      where the competition data and the preprocessing cache live
    coco       annotated views, their filament polygons, one-view-per-observation sampling
    limb       limb circle fit and limb-darkening flat field (SolarDisk, fit_limb, flat_field)
    rle        COCO run-length encoding, pairwise IoU, submission CSV
    morph      disc structuring element, closing, hole filling, probability -> logit thresholds
    fusion     detections: grown claim regions, claiming map pixels, merging several detectors' lists
    pq         IoU > 0.5 matching and pooled Panoptic Quality totals
    crops      the filament-biased training crop sampler of the dense networks
"""
