"""Label-invariant spatial reproducibility on the unchanged measurement grid.

Only internal four-neighbor edges enter the boundary set. Their midpoints are
represented on a doubled lattice so distances remain exact, including half
pixels. The three observation widths are never interpreted as distance radii.
This module uses NumPy only and performs no fitting, RNG, or I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NotRequired, TypedDict

import numpy as np


WINDOWS = (3, 5, 9)
K = 8


class DirectedCoverage(TypedDict):
    matched_count: int
    boundary_count: int
    value: float | None
    undefined_reason: str | None


class BoundaryComparison(TypedDict):
    window: int
    radius_pixels: int
    a_to_b: DirectedCoverage
    b_to_a: DirectedCoverage
    lower: float | None
    upper: float | None


class ARIResult(TypedDict):
    value: float | None
    undefined_reason: str | None
    used_clusters: list[int]
    degeneracy_flags: list[str]


class OverlapResult(TypedDict):
    counts: list[list[int]]
    row_fractions: list[list[float | None]]
    column_fractions: list[list[float | None]]
    h_b_given_a_bits: float
    h_a_given_b_bits: float


class MapComparison(TypedDict):
    ari: ARIResult
    boundaries: list[BoundaryComparison]
    overlap: NotRequired[OverlapResult]


class MapDiagnostics(TypedDict):
    valid_pixels: int
    valid_edges: int
    boundary_count: int
    boundary_density: float | None
    boundary_density_undefined_reason: str | None
    cluster_counts: list[int]
    occupancy: list[float]
    used_clusters: int
    maximum_occupancy: float


def validate_map(labels: np.ndarray, valid_mask: np.ndarray) -> None:
    """Require all and only source-valid pixels, with integer labels in 1..8.

    Parameters
    ----------
    labels : numpy.ndarray
        Two-dimensional integer map; invalid/background pixels must be zero.
    valid_mask : numpy.ndarray
        Source preprocessing mask, supplied independently of predictions.

    Raises
    ------
    ValueError
        On an empty mask, invalid values, or any missing/additional prediction.
        Input defects are never converted to score missingness.
    """
    if (not isinstance(labels, np.ndarray) or labels.ndim != 2
            or labels.dtype.kind not in "iu"):
        raise ValueError("labels must be a two-dimensional integer array")
    if (not isinstance(valid_mask, np.ndarray) or valid_mask.ndim != 2
            or valid_mask.shape != labels.shape or valid_mask.dtype.kind not in "biu"
            or not np.isin(valid_mask, (0, 1)).all()):
        raise ValueError("valid_mask must be a matching two-dimensional binary array")
    valid = valid_mask.astype(bool, copy=False)
    if not np.any(valid):
        raise ValueError("Empty valid-pixel set is an input failure")
    if np.any(labels < 0) or np.any(labels > K):
        raise ValueError("Labels must be in 0..8")
    if np.any(labels[~valid] != 0):
        raise ValueError("Background/invalid pixels must have label 0")
    if np.any(labels[valid] == 0):
        raise ValueError("Missing predictions in valid_mask; do not discard pixels")


def _valid_edges(valid: np.ndarray) -> int:
    return (int(np.count_nonzero(valid[:, :-1] & valid[:, 1:]))
            + int(np.count_nonzero(valid[:-1, :] & valid[1:, :])))


def boundary_lattice(labels: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return each label-changing internal edge once at its doubled midpoint.

    The returned boolean array has shape ``(2 * H - 1, 2 * W - 1)``.
    Position ``(i, j)`` represents image coordinates ``(i / 2, j / 2)``.
    Outside edges and edges to holes/background are absent, without mask erosion.
    """
    validate_map(labels, valid_mask)
    valid = valid_mask.astype(bool, copy=False)
    height, width = labels.shape
    boundary = np.zeros((2 * height - 1, 2 * width - 1), dtype=bool)
    boundary[::2, 1::2] = (valid[:, :-1] & valid[:, 1:]
                           & (labels[:, :-1] != labels[:, 1:]))
    boundary[1::2, ::2] = (valid[:-1, :] & valid[1:, :]
                           & (labels[:-1, :] != labels[1:, :]))
    return boundary


def _validate_boundary(boundary: np.ndarray) -> None:
    if (not isinstance(boundary, np.ndarray) or boundary.ndim != 2
            or boundary.dtype != np.bool_ or any(size < 1 or size % 2 == 0
                                                for size in boundary.shape)):
        raise ValueError("boundary must be a nonempty odd-shaped boolean doubled lattice")
    if np.any(boundary[::2, ::2]) or np.any(boundary[1::2, 1::2]):
        raise ValueError("Boundary positions must be four-neighbor edge midpoints")


def boundary_neighborhood(boundary: np.ndarray, window: int) -> np.ndarray:
    """Dilate midpoint positions by the specified observation width.

    A width of 3/5/9 corresponds to a radius of 1/2/4 image pixels, hence
    2/4/8 doubled-lattice units. Separable boolean shifts need O(H W) memory;
    no boundary-by-boundary distance matrix or wraparound is used.
    """
    _validate_boundary(boundary)
    if type(window) is not int or window not in WINDOWS:
        raise ValueError("window must be one of the fixed widths 3, 5, 9")
    radius = window - 1
    horizontal = boundary.copy()
    for offset in range(1, min(radius, boundary.shape[1] - 1) + 1):
        horizontal[:, offset:] |= boundary[:, :-offset]
        horizontal[:, :-offset] |= boundary[:, offset:]
    neighborhood = horizontal.copy()
    for offset in range(1, min(radius, boundary.shape[0] - 1) + 1):
        neighborhood[offset:, :] |= horizontal[:-offset, :]
        neighborhood[:-offset, :] |= horizontal[offset:, :]
    return neighborhood


def map_diagnostics(labels: np.ndarray, valid_mask: np.ndarray) -> MapDiagnostics:
    """Return JSON-safe pixel, occupancy, and internal-edge diagnostics."""
    boundary = boundary_lattice(labels, valid_mask)
    valid = valid_mask.astype(bool, copy=False)
    pixels = int(np.count_nonzero(valid))
    edges = _valid_edges(valid)
    boundary_count = int(np.count_nonzero(boundary))
    counts = np.bincount(labels[valid].astype(np.int64), minlength=K + 1)[1:]
    occupancy = [int(count) / pixels for count in counts]
    return {
        "valid_pixels": pixels,
        "valid_edges": edges,
        "boundary_count": boundary_count,
        "boundary_density": boundary_count / edges if edges else None,
        "boundary_density_undefined_reason": None if edges else "no_valid_adjacent_edges",
        "cluster_counts": [int(count) for count in counts],
        "occupancy": occupancy,
        "used_clusters": int(np.count_nonzero(counts)),
        "maximum_occupancy": max(occupancy),
    }


def _contingency(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.bincount((left.astype(np.int64) - 1) * K + right.astype(np.int64) - 1,
                       minlength=K * K).reshape(K, K)


def _ari(table: np.ndarray) -> ARIResult:
    """Follow diagnostic_metrics._ari_pair's integer and FP32 conventions."""
    n = int(table.sum())
    rows, columns = table.sum(axis=1), table.sum(axis=0)
    used = [int(np.count_nonzero(rows)), int(np.count_nonzero(columns))]
    flags = [f"{side}_{kind}"
             for side, count in zip(("a", "b"), used, strict=True)
             for kind, applies in (("single_cluster", count == 1),
                                   ("all_singletons", count == n)) if applies]
    result: ARIResult = {
        "value": None,
        "undefined_reason": "fewer_than_two_valid_pixels" if n < 2 else None,
        "used_clusters": used,
        "degeneracy_flags": flags,
    }
    if n < 2:
        return result

    def pairs(count: int) -> int:
        value = int(count)
        return value * (value - 1) // 2

    total = pairs(n)
    a = sum(pairs(count) for count in rows)
    b = sum(pairs(count) for count in columns)
    c = sum(pairs(count) for count in table.flat)
    numerator = 2 * (c * total - a * b)
    denominator = (a + b) * total - 2 * a * b
    result["value"] = (1.0 if denominator == 0
                       else float(np.float32(numerator) / np.float32(denominator)))
    return result


def _coverage(source: np.ndarray, target_neighborhood: np.ndarray,
              valid_edges: int) -> DirectedCoverage:
    denominator = int(np.count_nonzero(source))
    numerator = int(np.count_nonzero(source & target_neighborhood))
    reason = ("no_valid_adjacent_edges" if valid_edges == 0
              else "empty_source_boundary" if denominator == 0 else None)
    return {
        "matched_count": numerator,
        "boundary_count": denominator,
        "value": numerator / denominator if denominator else None,
        "undefined_reason": reason,
    }


def _overlap(table: np.ndarray) -> OverlapResult:
    rows, columns = table.sum(axis=1), table.sum(axis=0)
    n = int(table.sum())
    row_fractions = [[int(value) / int(rows[i]) if rows[i] else None
                      for value in row] for i, row in enumerate(table)]
    column_fractions = [[int(value) / int(columns[j]) if columns[j] else None
                         for j, value in enumerate(row)] for row in table]
    # Only positive entries enter log2; absent clusters retain null fractions.
    i, j = np.nonzero(table)
    mass = table[i, j].astype(np.float64)
    h_b_given_a = float(np.sum(mass / n * np.log2(rows[i] / mass)))
    h_a_given_b = float(np.sum(mass / n * np.log2(columns[j] / mass)))
    return {
        "counts": [[int(value) for value in row] for row in table],
        "row_fractions": row_fractions,
        "column_fractions": column_fractions,
        "h_b_given_a_bits": h_b_given_a,
        "h_a_given_b_bits": h_a_given_b,
    }


def compare_maps(
    labels_a: np.ndarray, labels_b: np.ndarray, valid_mask: np.ndarray, *,
    include_overlap: bool = False,
) -> MapComparison:
    """Compare partitions and boundary locations on exactly the source mask.

    Returns
    -------
    MapComparison
        ARI plus both directed boundary coverages for each fixed width. Empty
        source boundaries yield ``None``; nonempty sources with an empty target
        yield zero. ``lower`` and ``upper`` require both directions. Optional
        contingency and conditional entropy diagnostics do not align labels.

    Notes
    -----
    Coverage permits many-to-one proximity and is not a boundary F-score.
    Neither label numbers nor boundary orientation constrain the matching.
    Ratios use integer counts; ARI preserves the existing FP32 final ratio.
    """
    boundary_a = boundary_lattice(labels_a, valid_mask)
    boundary_b = boundary_lattice(labels_b, valid_mask)
    valid = valid_mask.astype(bool, copy=False)
    edges = _valid_edges(valid)
    table = _contingency(labels_a[valid], labels_b[valid])
    comparisons: list[BoundaryComparison] = []
    for window in WINDOWS:
        a_to_b = _coverage(boundary_a, boundary_neighborhood(boundary_b, window), edges)
        b_to_a = _coverage(boundary_b, boundary_neighborhood(boundary_a, window), edges)
        forward, reverse = a_to_b["value"], b_to_a["value"]
        comparisons.append({
            "window": window,
            "radius_pixels": (window - 1) // 2,
            "a_to_b": a_to_b,
            "b_to_a": b_to_a,
            "lower": (min(forward, reverse)
                      if forward is not None and reverse is not None else None),
            "upper": (max(forward, reverse)
                      if forward is not None and reverse is not None else None),
        })
    result: MapComparison = {"ari": _ari(table), "boundaries": comparisons}
    if include_overlap:
        result["overlap"] = _overlap(table)
    return result


def boundary_frequency(
    maps: Sequence[np.ndarray], valid_mask: np.ndarray, window: int,
) -> np.ndarray:
    """Return the descriptive nine-map boundary frequency at pixel centers.

    All nine maps must be present and valid. An empty boundary contributes
    zero, whereas a missing map is an input error. Invalid image positions are
    NaN in the returned FP32 array; valid positions have values in 0..1. This
    frequency is not a confidence or an independent-trial probability estimate.
    """
    if len(maps) != 9:
        raise ValueError("Exactly nine completed maps are required for boundary frequency")
    frequency: np.ndarray | None = None
    for labels in maps:
        near = boundary_neighborhood(boundary_lattice(labels, valid_mask), window)
        centers = near[::2, ::2]
        if frequency is None:
            frequency = centers.astype(np.uint8)
        else:
            frequency += centers
    if frequency is None:
        raise ValueError("Exactly nine completed maps are required for boundary frequency")
    result = frequency.astype(np.float32) / np.float32(9)
    result[~valid_mask.astype(bool, copy=False)] = np.nan
    return result
