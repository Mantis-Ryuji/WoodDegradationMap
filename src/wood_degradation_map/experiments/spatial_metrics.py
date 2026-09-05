"""Per-sample LLA and occupancy-adjusted LLA from the fixed valid-pixel mask."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CLUSTER_COUNTS, experiment_config


@dataclass(frozen=True)
class LLAWindowResult:
    window: int
    matching_pairs: int
    valid_pairs: int
    pixels_with_neighbors: int
    neighbor_pixel_fraction: float
    lla: float | None
    lla_undefined_reason: str | None
    adjusted_lla: float | None
    adjusted_undefined_reasons: tuple[str, ...]


@dataclass(frozen=True)
class LLAResult:
    """One sample's diagnostics and three separate scores; no sample aggregation.

    Pair counts are directed, as in the protocol's sum over centers and their
    neighbors. Occupancy includes every valid pixel, including isolated pixels.
    neighbor_pixel_fraction is the fraction of valid centers with at least one
    valid neighbor; it is a coverage diagnostic, not the LLA denominator.
    Undefined scores use None so dataclasses.asdict can be saved as JSON null.
    """

    k: int
    valid_pixels: int
    cluster_counts: tuple[int, ...]
    occupancy: tuple[float, ...]
    used_clusters: int
    maximum_occupancy: float
    expected_agreement: float | None
    windows: tuple[LLAWindowResult, ...]


def _ratio(numerator: int, denominator: int) -> float:
    """Perform a final ratio in FP32; all preceding counts remain integers."""
    return float(np.float32(numerator) / np.float32(denominator))


def _pair_counts(labels: np.ndarray, valid: np.ndarray, window: int) -> tuple[int, int, int]:
    height, width = labels.shape
    radius = window // 2
    matching, pairs = 0, 0
    has_neighbor = np.zeros(labels.shape, dtype=bool)
    # Only overlapping image slices are compared. No padding, wraparound, or
    # H x W x window^2 neighborhood tensor is needed, even at image boundaries.
    for dy in range(-min(radius, height - 1), min(radius, height - 1) + 1):
        top, bottom = max(0, -dy), min(height, height - dy)
        for dx in range(-min(radius, width - 1), min(radius, width - 1) + 1):
            if dy == 0 and dx == 0:
                continue
            left, right = max(0, -dx), min(width, width - dx)
            center = (slice(top, bottom), slice(left, right))
            neighbor = (slice(top + dy, bottom + dy), slice(left + dx, right + dx))
            usable = valid[center] & valid[neighbor]
            pairs += int(np.count_nonzero(usable))
            matching += int(np.count_nonzero(usable & (labels[center] == labels[neighbor])))
            has_neighbor[center] |= usable
    return matching, pairs, int(np.count_nonzero(has_neighbor))


def _scores(
    matching: int, pairs: int, counts: tuple[int, ...],
) -> tuple[float | None, float | None, tuple[str, ...]]:
    n = sum(counts)
    reasons = []
    if n < 2:
        reasons.append("fewer_than_two_valid_pixels")
    if pairs == 0:
        reasons.append("no_valid_neighbor_pairs")
    if sum(count > 0 for count in counts) == 1:
        reasons.append("single_cluster")
    lla = _ratio(matching, pairs) if pairs else None
    if reasons:
        return lla, None, tuple(reasons)
    total = n * (n - 1)
    equal = sum(count * (count - 1) for count in counts)
    # Algebraically (LLA - P)/(1 - P). Integer differences avoid declaring a
    # multi-cluster sample undefined when its FP32 P rounds to one. Do not add
    # an epsilon, clip negatives, or subtract two rounded near-one probabilities.
    adjusted = _ratio(matching * total - pairs * equal, pairs * (total - equal))
    return lla, adjusted, ()


def local_label_agreement(labels: np.ndarray, valid_mask: np.ndarray, *, k: int) -> LLAResult:
    """Compute LLA-3/5/9 and adjusted LLA per evaluation_metrics.md, section 4.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer H x W map, background 0 and valid labels 1..K.
    valid_mask : numpy.ndarray
        The matching preprocessing mask, with only 0/1 or boolean values.
        It must be supplied explicitly; missing predictions cannot define it.
    k : int
        A cluster count in the fixed experiment plan.

    Returns
    -------
    LLAResult
        Integer counts, FP32-derived ratios and explicit undefined-value reasons.

    Raises
    ------
    ValueError
        If the mask is empty, inputs have invalid types/shapes/values, or labels
        do not cover exactly the valid mask. Invalid inputs are not score missingness.

    Notes
    -----
    The input arrays are not modified. Work is O(H W sum(r^2 - 1)) for r=3,5,9,
    with O(H W) temporary memory. No spectra, GPU, RNG, fitting, or I/O is used.
    """
    if type(k) is not int or k not in CLUSTER_COUNTS:
        raise ValueError("K must be an integer in the fixed experiment plan")
    if (not isinstance(labels, np.ndarray) or labels.ndim != 2
            or labels.dtype.kind not in "iu"):
        raise ValueError("labels must be a two-dimensional integer array")
    if (not isinstance(valid_mask, np.ndarray) or valid_mask.ndim != 2
            or valid_mask.shape != labels.shape or valid_mask.dtype.kind not in "biu"
            or not np.isin(valid_mask, (0, 1)).all()):
        raise ValueError("valid_mask must be a matching two-dimensional binary array")
    valid = valid_mask.astype(bool, copy=True)
    n = int(np.count_nonzero(valid))
    if n == 0:
        raise ValueError("Empty valid-pixel set is an input failure")
    if np.any(labels < 0) or np.any(labels > k):
        raise ValueError("Labels must be in 0..K")
    if np.any(labels[~valid] != 0):
        raise ValueError("Background/invalid pixels must have label 0")
    if np.any(labels[valid] == 0):
        raise ValueError("Missing predictions in valid_mask; do not discard pixels")
    counts = tuple(int(count) for count in np.bincount(
        labels[valid].astype(np.int64, copy=False), minlength=k + 1,
    )[1:])
    occupancy = tuple(_ratio(count, n) for count in counts)
    expected = (_ratio(sum(count * (count - 1) for count in counts), n * (n - 1))
                if n >= 2 else None)
    windows = []
    for window in experiment_config()["evaluation"]["lla_windows"]:
        matching, pairs, covered = _pair_counts(labels, valid, window)
        lla, adjusted, reasons = _scores(matching, pairs, counts)
        windows.append(LLAWindowResult(
            window, matching, pairs, covered, _ratio(covered, n), lla,
            "no_valid_neighbor_pairs" if pairs == 0 else None, adjusted, reasons,
        ))
    return LLAResult(k, n, counts, occupancy, sum(count > 0 for count in counts),
                     max(occupancy), expected, tuple(windows))
