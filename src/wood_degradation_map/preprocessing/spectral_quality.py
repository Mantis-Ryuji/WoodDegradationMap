"""Spectral extraction, SNV, and production quality diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .envi_io import CubeDescriptor, ReferenceSet
from .reflectance import convert_to_reflectance


ANOMALY_COLUMNS = [
    "candidate_id",
    "metric_name",
    "rank",
    "metric_value",
    "sample_id",
    "row",
    "column",
    "band_index",
    "wavelength_nm",
    "reflectance",
    "snv",
    "snv_input_mean",
    "snv_input_std",
]


def extract_masked_reflectance(
    cube: np.ndarray,
    descriptor: CubeDescriptor,
    reference: ReferenceSet,
    mask: np.ndarray,
    *,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert and extract masked pixels in bounded row chunks."""

    pixel_count = int(mask.sum())
    spectra = np.empty((pixel_count, descriptor.shape[2]), dtype=np.float32)
    coordinates = np.empty((pixel_count, 2), dtype=np.int32)
    cursor = 0
    for row_start in range(0, descriptor.shape[0], row_chunk_size):
        row_stop = min(row_start + row_chunk_size, descriptor.shape[0])
        block_mask = mask[row_start:row_stop]
        if not block_mask.any():
            continue
        rows, columns = np.nonzero(block_mask)
        intensity_block = np.asarray(cube[row_start:row_stop], dtype=np.float64)
        reflectance_block = convert_to_reflectance(
            intensity_block,
            reference.white,
            reference.dark,
        )
        selected = reflectance_block[block_mask]
        next_cursor = cursor + selected.shape[0]
        spectra[cursor:next_cursor] = selected.astype(np.float32)
        coordinates[cursor:next_cursor, 0] = rows + row_start
        coordinates[cursor:next_cursor, 1] = columns
        cursor = next_cursor
    if cursor != pixel_count:
        raise RuntimeError(f"Extracted {cursor} masked pixels, expected {pixel_count}")
    return spectra, coordinates


def band_statistics(
    spectra: np.ndarray,
    *,
    sample_id: str,
    stage: str,
    wavelengths_nm: np.ndarray,
    low_snr: np.ndarray,
) -> pd.DataFrame:
    """Summarize finite values and out-of-range fractions for every band."""

    records: list[dict[str, object]] = []
    total = spectra.shape[0]
    for band_index in range(spectra.shape[1]):
        values = spectra[:, band_index].astype(np.float64, copy=False)
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            quantiles = np.quantile(
                finite_values,
                (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
            )
            value_mean = float(finite_values.mean())
            value_std = float(finite_values.std(ddof=0))
            value_min = float(finite_values.min())
            value_max = float(finite_values.max())
            negative_fraction = float((finite_values < 0.0).mean())
            above_one_fraction = float((finite_values > 1.0).mean())
        else:
            quantiles = np.full(7, np.nan)
            value_mean = value_std = value_min = value_max = np.nan
            negative_fraction = above_one_fraction = np.nan
        records.append(
            {
                "sample_id": sample_id,
                "stage": stage,
                "band_index": band_index,
                "wavelength_nm": float(wavelengths_nm[band_index]),
                "low_snr": bool(low_snr[band_index]),
                "masked_pixel_count": total,
                "finite_count": int(finite_values.size),
                "nonfinite_fraction": 1.0 - finite_values.size / total,
                "mean": value_mean,
                "std": value_std,
                "min": value_min,
                "q01": float(quantiles[0]),
                "q05": float(quantiles[1]),
                "q25": float(quantiles[2]),
                "median": float(quantiles[3]),
                "q75": float(quantiles[4]),
                "q95": float(quantiles[5]),
                "q99": float(quantiles[6]),
                "max": value_max,
                "negative_fraction_of_finite": negative_fraction,
                "above_one_fraction_of_finite": above_one_fraction,
            }
        )
    return pd.DataFrame(records)


def apply_snv_in_place(
    spectra: np.ndarray,
    *,
    spectrum_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply pixel-wise SNV with sample standard deviation in place."""

    nonfinite_band_count = (~np.isfinite(spectra)).sum(axis=1).astype(np.uint16)
    input_mean = np.full(spectra.shape[0], np.nan, dtype=np.float64)
    input_std = np.full(spectra.shape[0], np.nan, dtype=np.float64)
    valid_snv = np.zeros(spectra.shape[0], dtype=bool)
    for start in range(0, spectra.shape[0], spectrum_chunk_size):
        stop = min(start + spectrum_chunk_size, spectra.shape[0])
        block = spectra[start:stop].astype(np.float64)
        finite = np.isfinite(block).all(axis=1)
        if finite.any():
            finite_block = block[finite]
            means = finite_block.mean(axis=1)
            stds = finite_block.std(axis=1, ddof=1)
            usable = np.isfinite(stds) & (stds > 0.0)
            block_indices = np.flatnonzero(finite)
            input_mean[start + block_indices] = means
            input_std[start + block_indices] = stds
            usable_indices = block_indices[usable]
            valid_snv[start + usable_indices] = True
            normalized = np.full(block.shape, np.nan, dtype=np.float64)
            normalized[usable_indices] = (
                finite_block[usable] - means[usable, None]
            ) / stds[usable, None]
            spectra[start:stop] = normalized.astype(np.float32)
        else:
            spectra[start:stop] = np.nan
    return valid_snv, input_mean, input_std, nonfinite_band_count


def snv_pixel_metrics(
    snv: np.ndarray,
    valid_snv: np.ndarray,
    low_snr: np.ndarray,
    *,
    spectrum_chunk_size: int,
) -> dict[str, np.ndarray]:
    """Compute normalization and spectral-roughness metrics per pixel."""

    metric_names = (
        "snv_mean",
        "snv_sample_std",
        "snv_l2_norm",
        "first_difference_rms",
        "max_abs_second_difference",
        "low_snr_difference_rms",
        "stable_difference_rms",
        "low_to_stable_roughness_ratio",
    )
    metrics = {
        name: np.full(snv.shape[0], np.nan, dtype=np.float64) for name in metric_names
    }
    low_pairs = low_snr[:-1] & low_snr[1:]
    stable_pairs = ~low_snr[:-1] & ~low_snr[1:]
    for start in range(0, snv.shape[0], spectrum_chunk_size):
        stop = min(start + spectrum_chunk_size, snv.shape[0])
        usable = valid_snv[start:stop]
        if not usable.any():
            continue
        block = snv[start:stop][usable].astype(np.float64)
        indices = start + np.flatnonzero(usable)
        first_difference = np.diff(block, axis=1)
        second_difference = np.diff(block, n=2, axis=1)
        metrics["snv_mean"][indices] = block.mean(axis=1)
        metrics["snv_sample_std"][indices] = block.std(axis=1, ddof=1)
        metrics["snv_l2_norm"][indices] = np.linalg.norm(block, axis=1)
        metrics["first_difference_rms"][indices] = np.sqrt(
            np.mean(np.square(first_difference), axis=1)
        )
        metrics["max_abs_second_difference"][indices] = np.max(
            np.abs(second_difference), axis=1
        )
        if low_pairs.any():
            low_rms = np.sqrt(
                np.mean(np.square(first_difference[:, low_pairs]), axis=1)
            )
            metrics["low_snr_difference_rms"][indices] = low_rms
        if stable_pairs.any():
            stable_rms = np.sqrt(
                np.mean(np.square(first_difference[:, stable_pairs]), axis=1)
            )
            metrics["stable_difference_rms"][indices] = stable_rms
        if low_pairs.any() and stable_pairs.any():
            ratio = np.full(block.shape[0], np.nan, dtype=np.float64)
            np.divide(low_rms, stable_rms, out=ratio, where=stable_rms > 0.0)
            metrics["low_to_stable_roughness_ratio"][indices] = ratio
    return metrics


def anomaly_candidates(
    *,
    sample_id: str,
    coordinates: np.ndarray,
    snv: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    metrics: dict[str, np.ndarray],
    per_sample_count: int,
) -> list[dict[str, object]]:
    """Collect the highest-ranked SNV anomaly candidates for one sample."""

    candidates: list[dict[str, object]] = []
    for metric_name in (
        "max_abs_second_difference",
        "low_to_stable_roughness_ratio",
    ):
        values = metrics[metric_name]
        finite_indices = np.flatnonzero(np.isfinite(values))
        if not finite_indices.size:
            continue
        count = min(per_sample_count, finite_indices.size)
        ordered = finite_indices[np.argsort(values[finite_indices])[-count:]]
        for pixel_index in ordered:
            snv_spectrum = snv[pixel_index].astype(np.float64)
            candidates.append(
                {
                    "metric_name": metric_name,
                    "metric_value": float(values[pixel_index]),
                    "sample_id": sample_id,
                    "row": int(coordinates[pixel_index, 0]),
                    "column": int(coordinates[pixel_index, 1]),
                    "snv_input_mean": float(input_mean[pixel_index]),
                    "snv_input_std": float(input_std[pixel_index]),
                    "reflectance": snv_spectrum * input_std[pixel_index]
                    + input_mean[pixel_index],
                    "snv": snv_spectrum,
                }
            )
    return candidates


def aggregate_band_statistics(statistics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-sample band statistics with equal sample weighting."""

    records: list[dict[str, object]] = []
    for (stage, band_index), group in statistics.groupby(
        ["stage", "band_index"], sort=True
    ):
        records.append(
            {
                "stage": stage,
                "band_index": int(band_index),
                "wavelength_nm": float(group["wavelength_nm"].iloc[0]),
                "low_snr": bool(group["low_snr"].iloc[0]),
                "sample_count": int(len(group)),
                "median_nonfinite_fraction": float(group["nonfinite_fraction"].median()),
                "q90_nonfinite_fraction": float(group["nonfinite_fraction"].quantile(0.9)),
                "max_nonfinite_fraction": float(group["nonfinite_fraction"].max()),
                "median_negative_fraction": float(
                    group["negative_fraction_of_finite"].median()
                ),
                "q90_negative_fraction": float(
                    group["negative_fraction_of_finite"].quantile(0.9)
                ),
                "max_negative_fraction": float(
                    group["negative_fraction_of_finite"].max()
                ),
                "median_above_one_fraction": float(
                    group["above_one_fraction_of_finite"].median()
                ),
                "q90_above_one_fraction": float(
                    group["above_one_fraction_of_finite"].quantile(0.9)
                ),
                "max_above_one_fraction": float(
                    group["above_one_fraction_of_finite"].max()
                ),
                "median_sample_q01": float(group["q01"].median()),
                "median_sample_median": float(group["median"].median()),
                "median_sample_q99": float(group["q99"].median()),
                "q10_sample_median": float(group["median"].quantile(0.1)),
                "q90_sample_median": float(group["median"].quantile(0.9)),
            }
        )
    return pd.DataFrame(records)


def select_global_anomalies(
    candidates: list[dict[str, object]],
    *,
    top_spectra: int,
    wavelengths_nm: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Select globally ranked candidates and expand them to long-form tables."""

    selected: dict[str, pd.DataFrame] = {}
    for metric_name in (
        "max_abs_second_difference",
        "low_to_stable_roughness_ratio",
    ):
        metric_candidates = [
            candidate
            for candidate in candidates
            if candidate["metric_name"] == metric_name
        ]
        metric_candidates.sort(key=lambda item: float(item["metric_value"]), reverse=True)
        records: list[dict[str, object]] = []
        for rank, candidate in enumerate(metric_candidates[:top_spectra], start=1):
            candidate_id = (
                f"{metric_name}:{rank}:{candidate['sample_id']}:"
                f"{candidate['row']}:{candidate['column']}"
            )
            spectrum = np.asarray(candidate["snv"], dtype=np.float64)
            raw_spectrum = np.asarray(candidate["reflectance"], dtype=np.float64)
            for band_index, value in enumerate(spectrum):
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "metric_name": metric_name,
                        "rank": rank,
                        "metric_value": candidate["metric_value"],
                        "sample_id": candidate["sample_id"],
                        "row": candidate["row"],
                        "column": candidate["column"],
                        "band_index": band_index,
                        "wavelength_nm": float(wavelengths_nm[band_index]),
                        "reflectance": float(raw_spectrum[band_index]),
                        "snv": float(value),
                        "snv_input_mean": candidate["snv_input_mean"],
                        "snv_input_std": candidate["snv_input_std"],
                    }
                )
        selected[metric_name] = pd.DataFrame(records, columns=ANOMALY_COLUMNS)
    return selected
