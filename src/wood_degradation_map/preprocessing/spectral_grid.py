"""Reference-SNR cutoff and spectral-grid interpolation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


_CUTOFF_RULE = (
    "exclude the terminal contiguous run of bands whose 200 Hz reference "
    "spatial SNR proxy is non-finite or <= the fixed threshold"
)


@dataclass(frozen=True)
class TerminalSnrCutoffDecision:
    """Result of the reference-only terminal low-SNR cutoff rule."""

    total_source_bands: int
    retained_source_bands: int
    excluded_source_bands: int
    last_retained_band: int
    last_retained_wavelength_nm: float
    first_excluded_band: int | None
    first_excluded_wavelength_nm: float | None
    cutoff_boundary_nm: float | None
    snr_threshold: float
    rule: str = _CUTOFF_RULE

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-compatible decision record."""

        return asdict(self)


@dataclass(frozen=True)
class LinearInterpolationPlan:
    """Precomputed indices and weights for a shared linear wavelength grid."""

    source_wavelength_nm: np.ndarray
    target_wavelength_nm: np.ndarray
    left_source_index: np.ndarray
    right_source_index: np.ndarray
    right_weight: np.ndarray


def derive_terminal_snr_cutoff(
    reference_quality: pd.DataFrame,
    *,
    snr_threshold: float,
) -> TerminalSnrCutoffDecision:
    """Derive one upper cutoff from a terminal contiguous low-SNR run."""

    required = {"band_index", "wavelength_nm", "snr_proxy"}
    missing = required - set(reference_quality.columns)
    if missing:
        raise ValueError(f"Reference quality is missing columns: {sorted(missing)}")
    if snr_threshold <= 0.0:
        raise ValueError("snr_threshold must be positive")
    ordered = reference_quality.sort_values("band_index")
    band_index = ordered["band_index"].to_numpy(dtype=np.int64)
    wavelength = ordered["wavelength_nm"].to_numpy(dtype=np.float64)
    snr = ordered["snr_proxy"].to_numpy(dtype=np.float64)
    if not np.array_equal(band_index, np.arange(len(ordered))):
        raise ValueError("band_index must be contiguous and start at zero")
    if len(wavelength) < 3 or not np.isfinite(wavelength).all():
        raise ValueError("At least three finite wavelengths are required")
    if not (np.diff(wavelength) > 0.0).all():
        raise ValueError("Wavelengths must be strictly increasing")

    low_snr = ~np.isfinite(snr) | (snr <= snr_threshold)
    if not low_snr[-1]:
        if low_snr.any():
            raise ValueError(
                "Low-SNR bands are not a terminal suffix; one upper cutoff is invalid"
            )
        retained = len(wavelength)
    else:
        retained = len(wavelength) - 1
        while retained > 0 and low_snr[retained - 1]:
            retained -= 1
        if low_snr[:retained].any():
            raise ValueError(
                "Low-SNR bands also occur before the terminal suffix; one cutoff is invalid"
            )
    if retained < 2:
        raise ValueError("The automatic cutoff would retain fewer than two bands")

    first_excluded = retained if retained < len(wavelength) else None
    boundary = (
        float((wavelength[retained - 1] + wavelength[retained]) / 2.0)
        if first_excluded is not None
        else None
    )
    return TerminalSnrCutoffDecision(
        total_source_bands=len(wavelength),
        retained_source_bands=retained,
        excluded_source_bands=len(wavelength) - retained,
        last_retained_band=retained - 1,
        last_retained_wavelength_nm=float(wavelength[retained - 1]),
        first_excluded_band=first_excluded,
        first_excluded_wavelength_nm=(
            float(wavelength[first_excluded]) if first_excluded is not None else None
        ),
        cutoff_boundary_nm=boundary,
        snr_threshold=snr_threshold,
    )


def build_linear_interpolation_plan(
    source_wavelength_nm: np.ndarray,
    *,
    target_bands: int,
) -> LinearInterpolationPlan:
    """Build a no-extrapolation linear plan over the retained wavelength range."""

    source = np.asarray(source_wavelength_nm, dtype=np.float64)
    if source.ndim != 1 or source.size < 2:
        raise ValueError("source_wavelength_nm must contain at least two values")
    if not np.isfinite(source).all() or not (np.diff(source) > 0.0).all():
        raise ValueError("Source wavelengths must be finite and strictly increasing")
    if target_bands < 2:
        raise ValueError("target_bands must be at least two")

    target = np.linspace(source[0], source[-1], target_bands, dtype=np.float64)
    right = np.searchsorted(source, target, side="right")
    right = np.clip(right, 1, source.size - 1)
    left = right - 1
    denominator = source[right] - source[left]
    right_weight = (target - source[left]) / denominator
    return LinearInterpolationPlan(
        source_wavelength_nm=source,
        target_wavelength_nm=target,
        left_source_index=left.astype(np.int32),
        right_source_index=right.astype(np.int32),
        right_weight=right_weight,
    )


def interpolate_spectra_linear(
    spectra: np.ndarray,
    plan: LinearInterpolationPlan,
    *,
    spectrum_chunk_size: int,
) -> np.ndarray:
    """Apply a shared linear interpolation plan without extrapolation."""

    spectra = np.asarray(spectra)
    if spectra.ndim != 2 or spectra.shape[1] != plan.source_wavelength_nm.size:
        raise ValueError("Spectra do not match the interpolation source grid")
    if spectrum_chunk_size <= 0:
        raise ValueError("spectrum_chunk_size must be positive")
    output = np.empty(
        (spectra.shape[0], plan.target_wavelength_nm.size),
        dtype=np.float32,
    )
    right_weight = plan.right_weight[None, :]
    for start in range(0, spectra.shape[0], spectrum_chunk_size):
        stop = min(start + spectrum_chunk_size, spectra.shape[0])
        block = spectra[start:stop].astype(np.float64)
        interpolated = (
            block[:, plan.left_source_index] * (1.0 - right_weight)
            + block[:, plan.right_source_index] * right_weight
        )
        output[start:stop] = interpolated.astype(np.float32)
    return output


def wavelength_grid_table(plan: LinearInterpolationPlan) -> pd.DataFrame:
    """Return the target grid and its source interpolation weights."""

    return pd.DataFrame(
        {
            "target_band_index": np.arange(plan.target_wavelength_nm.size),
            "target_wavelength_nm": plan.target_wavelength_nm,
            "left_source_band": plan.left_source_index,
            "right_source_band": plan.right_source_index,
            "right_source_weight": plan.right_weight,
        }
    )
