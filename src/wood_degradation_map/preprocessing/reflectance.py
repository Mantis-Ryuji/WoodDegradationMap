"""Reflectance conversion used by production preprocessing."""

from __future__ import annotations

import numpy as np


def convert_to_reflectance(
    intensity: np.ndarray,
    white: np.ndarray,
    dark: np.ndarray,
) -> np.ndarray:
    """Convert intensity using broadcast-compatible white and dark references.

    Non-positive or non-finite denominators are represented as ``NaN``. No
    clipping, epsilon addition, interpolation, or spectral filtering is applied.
    """

    intensity = np.asarray(intensity, dtype=np.float64)
    white = np.asarray(white, dtype=np.float64)
    dark = np.asarray(dark, dtype=np.float64)
    if white.shape != dark.shape:
        raise ValueError(f"white.shape {white.shape} != dark.shape {dark.shape}")
    try:
        output_shape = np.broadcast_shapes(intensity.shape, white.shape)
    except ValueError as exc:
        raise ValueError(
            f"Intensity {intensity.shape} and reference {white.shape} do not broadcast"
        ) from exc

    denominator = white - dark
    valid_denominator = np.isfinite(denominator) & (denominator > 0.0)
    numerator = intensity - dark
    reflectance = np.full(output_shape, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=reflectance,
        where=np.broadcast_to(valid_denominator, output_shape),
    )
    return reflectance


def reflectance_l2_norm(
    reflectance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the final axis to an L2 norm only when every band is finite."""

    reflectance = np.asarray(reflectance, dtype=np.float64)
    if reflectance.ndim < 1 or reflectance.shape[-1] == 0:
        raise ValueError("reflectance must contain a non-empty spectral axis")
    finite = np.isfinite(reflectance)
    finite_band_count = finite.sum(axis=-1)
    squared_sum = np.square(np.where(finite, reflectance, 0.0)).sum(axis=-1)
    all_bands_finite = finite_band_count == reflectance.shape[-1]
    norm = np.full(finite_band_count.shape, np.nan, dtype=np.float64)
    norm[all_bands_finite] = np.sqrt(squared_sum[all_bands_finite])
    return norm, finite_band_count
