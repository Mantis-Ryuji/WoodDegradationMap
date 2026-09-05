"""Wood-region masking from integrated 200 Hz raw intensity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, disk, remove_small_objects


@dataclass(frozen=True)
class MaskResult:
    """Intermediate and final masks plus quality metrics for one specimen."""

    score_map: np.ndarray
    otsu_threshold: float
    otsu_mask: np.ndarray
    after_erosion: np.ndarray
    final_mask: np.ndarray
    initial_component_count: int
    final_component_count: int

    def metrics(self) -> dict[str, int | float]:
        """Return scalar mask diagnostics suitable for a manifest table."""

        total_pixels = int(self.final_mask.size)
        otsu_pixels = int(self.otsu_mask.sum())
        after_erosion_pixels = int(self.after_erosion.sum())
        final_pixels = int(self.final_mask.sum())
        return {
            "height": int(self.final_mask.shape[0]),
            "width": int(self.final_mask.shape[1]),
            "total_pixels": total_pixels,
            "otsu_threshold": self.otsu_threshold,
            "otsu_pixels": otsu_pixels,
            "after_erosion_pixels": after_erosion_pixels,
            "final_mask_pixels": final_pixels,
            "erosion_removed_pixels": otsu_pixels - after_erosion_pixels,
            "small_object_removed_pixels": after_erosion_pixels - final_pixels,
            "final_mask_fraction": final_pixels / total_pixels,
            "initial_component_count": self.initial_component_count,
            "final_component_count": self.final_component_count,
        }


def integrate_intensity(cube_200: np.ndarray) -> np.ndarray:
    """Sum the raw 200 Hz intensity over bands without reflectance conversion."""

    cube_200 = np.asarray(cube_200)
    if cube_200.ndim != 3:
        raise ValueError(f"Expected a 3D intensity cube, got {cube_200.ndim}D")
    if cube_200.shape[2] == 0:
        raise ValueError("Intensity cube has no spectral bands")
    return np.sum(cube_200, axis=2, dtype=np.float64)


def build_score_mask(
    score_map: np.ndarray,
    *,
    threshold: float,
    min_object_size: int,
    erosion_radius: int = 1,
    connectivity: int = 2,
) -> MaskResult:
    """Threshold a scalar map, erode it, and then remove small objects."""

    score_map = np.asarray(score_map, dtype=np.float64)
    if score_map.ndim != 2:
        raise ValueError("score_map must be two-dimensional")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if min_object_size <= 0:
        raise ValueError("min_object_size must be a positive pixel count")
    if erosion_radius < 0:
        raise ValueError("erosion_radius must be non-negative")
    if connectivity not in {1, 2}:
        raise ValueError("connectivity must be 1 or 2 for a two-dimensional mask")

    threshold_mask = np.isfinite(score_map) & (score_map > threshold)
    initial_component_count = int(label(threshold_mask, connectivity=connectivity).max())
    if erosion_radius == 0:
        after_erosion = threshold_mask.copy()
    else:
        after_erosion = binary_erosion(threshold_mask, footprint=disk(erosion_radius))
    final_mask = remove_small_objects(
        after_erosion,
        max_size=min_object_size - 1,
        connectivity=connectivity,
    )
    final_component_count = int(label(final_mask, connectivity=connectivity).max())
    return MaskResult(
        score_map=score_map,
        otsu_threshold=float(threshold),
        otsu_mask=threshold_mask,
        after_erosion=after_erosion,
        final_mask=final_mask,
        initial_component_count=initial_component_count,
        final_component_count=final_component_count,
    )


def connected_component_records(
    mask: np.ndarray,
    *,
    connectivity: int = 2,
) -> list[dict[str, int | bool]]:
    """Describe every connected component without changing the mask."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if connectivity not in {1, 2}:
        raise ValueError("connectivity must be 1 or 2 for a two-dimensional mask")

    height, width = mask.shape
    labeled = label(mask, connectivity=connectivity)
    records: list[dict[str, int | bool]] = []
    for region in regionprops(labeled):
        min_row, min_col, max_row, max_col = region.bbox
        records.append(
            {
                "component_label": int(region.label),
                "area_pixels": int(region.area),
                "bbox_min_row": int(min_row),
                "bbox_min_col": int(min_col),
                "bbox_max_row_exclusive": int(max_row),
                "bbox_max_col_exclusive": int(max_col),
                "touches_image_boundary": bool(
                    min_row == 0 or min_col == 0 or max_row == height or max_col == width
                ),
            }
        )
    return records
