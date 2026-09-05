"""Reference-derived denominator quality diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reference_band_quality(
    white: np.ndarray,
    dark: np.ndarray,
    wavelengths_nm: np.ndarray,
    *,
    mode: str,
    snr_threshold: float = 10.0,
    denominator_floor: float = 0.0,
    eps: float = 1e-12,
    dtype_max: float | None = None,
) -> pd.DataFrame:
    """Compute the spatial reference SNR proxy and related band diagnostics.

    The SNR proxy is ``mean(W - D) / std(W - D)`` over reference pixels. It is
    a measure of spatial denominator stability, not temporal SNR from repeated
    acquisitions.
    """

    white = np.asarray(white)
    dark = np.asarray(dark)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
    if white.shape != dark.shape:
        raise ValueError(f"white.shape {white.shape} != dark.shape {dark.shape}")
    if white.ndim != 3:
        raise ValueError(f"Expected 3D references, got {white.ndim}D")
    if wavelengths_nm.ndim != 1 or wavelengths_nm.size != white.shape[2]:
        raise ValueError("Wavelength vector does not match reference bands")
    if snr_threshold <= 0:
        raise ValueError("snr_threshold must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    denominator = white.astype(np.float64, copy=False) - dark.astype(np.float64, copy=False)
    spatial_axes = (0, 1)
    denominator_mean = denominator.mean(axis=spatial_axes)
    denominator_std = denominator.std(axis=spatial_axes, ddof=0)
    snr_proxy = denominator_mean / (denominator_std + eps)
    low_snr = ~np.isfinite(snr_proxy) | (snr_proxy <= snr_threshold)

    if dtype_max is None:
        white_at_dtype_max = np.full(white.shape[2], np.nan)
        dark_at_dtype_max = np.full(dark.shape[2], np.nan)
    else:
        white_at_dtype_max = (white == dtype_max).mean(axis=spatial_axes)
        dark_at_dtype_max = (dark == dtype_max).mean(axis=spatial_axes)

    return pd.DataFrame(
        {
            "mode": mode,
            "band_index": np.arange(white.shape[2], dtype=np.int32),
            "wavelength_nm": wavelengths_nm,
            "denominator_mean": denominator_mean,
            "denominator_std": denominator_std,
            "denominator_min": denominator.min(axis=spatial_axes),
            "denominator_q01": np.quantile(denominator, 0.01, axis=spatial_axes),
            "denominator_q05": np.quantile(denominator, 0.05, axis=spatial_axes),
            "denominator_nonpositive_fraction": (denominator <= 0).mean(axis=spatial_axes),
            "denominator_below_floor_fraction": (
                denominator <= denominator_floor
            ).mean(axis=spatial_axes),
            "snr_proxy": snr_proxy,
            "low_snr": low_snr,
            "white_max": white.max(axis=spatial_axes),
            "dark_max": dark.max(axis=spatial_axes),
            "white_at_dtype_max_fraction": white_at_dtype_max,
            "dark_at_dtype_max_fraction": dark_at_dtype_max,
        }
    )
