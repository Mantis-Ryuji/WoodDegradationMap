"""Three-class Multi-Otsu masks from integrated 200 Hz raw intensity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.filters import threshold_multiotsu

from .masking import MaskResult, build_score_mask


@dataclass(frozen=True)
class ThreeClassMaskResult:
    """Multi-Otsu classes and the combined wood-mask stages."""

    thresholds: tuple[float, float]
    class_map: np.ndarray
    mask_result: MaskResult


def build_three_class_multiotsu_mask(
    intensity_integral: np.ndarray,
    *,
    min_object_size: int,
    erosion_radius: int = 1,
    connectivity: int = 2,
) -> ThreeClassMaskResult:
    """Treat the middle and high Multi-Otsu classes as wood."""

    intensity_integral = np.asarray(intensity_integral, dtype=np.float64)
    if intensity_integral.ndim != 2:
        raise ValueError("intensity_integral must be two-dimensional")
    if not np.isfinite(intensity_integral).all():
        raise ValueError("intensity_integral contains non-finite values")
    thresholds_array = threshold_multiotsu(intensity_integral, classes=3)
    thresholds = (float(thresholds_array[0]), float(thresholds_array[1]))
    class_map = np.digitize(
        intensity_integral,
        bins=thresholds_array,
        right=True,
    ).astype(np.uint8)
    mask_result = build_score_mask(
        intensity_integral,
        threshold=thresholds[0],
        min_object_size=min_object_size,
        erosion_radius=erosion_radius,
        connectivity=connectivity,
    )
    return ThreeClassMaskResult(
        thresholds=thresholds,
        class_map=class_map,
        mask_result=mask_result,
    )
