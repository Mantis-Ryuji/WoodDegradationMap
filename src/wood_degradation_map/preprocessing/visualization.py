"""Title-free figures emitted by production preprocessing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def shared_robust_display_limits(
    images: list[np.ndarray],
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[float, float]:
    """Return one robust display range shared by every supplied image."""

    if not images:
        raise ValueError("At least one image is required")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Display quantiles must satisfy 0 <= lower < upper <= 1")

    finite_parts = [image[np.isfinite(image)].ravel() for image in images]
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        return 0.0, 1.0
    finite = np.concatenate(finite_parts)
    lower, upper = np.quantile(finite, (lower_quantile, upper_quantile))
    if lower == upper:
        upper = lower + 1.0
    return float(lower), float(upper)


def plot_masked_scalar_map(
    scalar_map: np.ndarray,
    output_path: Path,
    *,
    display_limits: tuple[float, float],
    cmap: str,
) -> None:
    """Plot a title-free scalar map with non-finite background transparent."""

    if scalar_map.ndim != 2:
        raise ValueError("scalar_map must be two-dimensional")
    vmin, vmax = display_limits
    if not np.isfinite((vmin, vmax)).all() or vmin >= vmax:
        raise ValueError("display_limits must contain finite values with vmin < vmax")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = scalar_map.shape
    fig_width = 5.0
    fig_height = max(2.0, fig_width * height / width)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    transparent_cmap = plt.get_cmap(cmap).with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    axis.imshow(
        np.ma.masked_invalid(scalar_map),
        cmap=transparent_cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


def _shade_low_snr_bands(axis: plt.Axes, band_summary: pd.DataFrame) -> None:
    ordered = band_summary.sort_values("band_index")
    low_snr = ordered["low_snr"].to_numpy(dtype=bool)
    wavelengths = ordered["wavelength_nm"].to_numpy(dtype=np.float64)
    midpoints = (wavelengths[:-1] + wavelengths[1:]) / 2.0
    edges = np.concatenate(
        (
            [wavelengths[0] - (midpoints[0] - wavelengths[0])],
            midpoints,
            [wavelengths[-1] + (wavelengths[-1] - midpoints[-1])],
        )
    )
    padded = np.pad(low_snr.astype(np.int8), (1, 1))
    transitions = np.flatnonzero(np.diff(padded))
    for start, stop in transitions.reshape(-1, 2):
        axis.axvspan(edges[start], edges[stop], color="0.7", alpha=0.2, linewidth=0)


def plot_band_distribution(
    band_summary: pd.DataFrame,
    output_path: Path,
    *,
    y_label: str,
) -> None:
    """Plot an equal-sample median spectrum and its median 1--99% envelope."""

    ordered = band_summary.sort_values("band_index")
    wavelength = ordered["wavelength_nm"].to_numpy(dtype=np.float64)
    median = ordered["median_sample_median"].to_numpy(dtype=np.float64)
    lower = ordered["median_sample_q01"].to_numpy(dtype=np.float64)
    upper = ordered["median_sample_q99"].to_numpy(dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.0, 4.0), dpi=180)
    _shade_low_snr_bands(axis, ordered)
    axis.fill_between(
        wavelength,
        lower,
        upper,
        color="#0072B2",
        alpha=0.2,
        linewidth=0,
        label="Median sample 1–99% range",
    )
    axis.plot(
        wavelength,
        median,
        color="#0072B2",
        linewidth=1.4,
        label="Median sample median",
    )
    axis.set_xlabel("Wavelength [nm]")
    axis.set_ylabel(y_label)
    axis.grid(True, which="major", alpha=0.3)
    axis.tick_params(direction="in", top=True, right=True)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_snr_cutoff_decision(
    reference_quality: pd.DataFrame,
    band_summary: pd.DataFrame,
    output_path: Path,
    *,
    snr_threshold: float,
    cutoff_boundary_nm: float | None,
) -> None:
    """Visualize the automatic terminal low-SNR cutoff and its evidence."""

    reference = reference_quality.sort_values("band_index")
    reflectance = band_summary.loc[band_summary["stage"] == "reflectance"].sort_values(
        "band_index"
    )
    snv = band_summary.loc[band_summary["stage"] == "snv"].sort_values("band_index")
    wavelength = reference["wavelength_nm"].to_numpy(dtype=np.float64)
    if len(reflectance) != len(reference) or len(snv) != len(reference):
        raise ValueError("Band summaries do not match the reference wavelength grid")
    if not np.allclose(reflectance["wavelength_nm"], wavelength):
        raise ValueError("Reflectance summary wavelength grid differs from reference")
    if not np.allclose(snv["wavelength_nm"], wavelength):
        raise ValueError("SNV summary wavelength grid differs from reference")

    midpoints = (wavelength[:-1] + wavelength[1:]) / 2.0
    wavelength_edges = np.concatenate(
        (
            [wavelength[0] - (midpoints[0] - wavelength[0])],
            midpoints,
            [wavelength[-1] + (wavelength[-1] - midpoints[-1])],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 9.0), dpi=180, sharex=True)
    for axis_index, axis in enumerate(axes):
        if cutoff_boundary_nm is None:
            axis.axvspan(
                wavelength_edges[0],
                wavelength_edges[-1],
                color="#009E73",
                alpha=0.08,
                linewidth=0,
                label="Retained" if axis_index == 0 else None,
            )
        else:
            axis.axvspan(
                wavelength_edges[0],
                cutoff_boundary_nm,
                color="#009E73",
                alpha=0.08,
                linewidth=0,
                label="Retained" if axis_index == 0 else None,
            )
            axis.axvspan(
                cutoff_boundary_nm,
                wavelength_edges[-1],
                color="#D55E00",
                alpha=0.10,
                linewidth=0,
                label="Excluded" if axis_index == 0 else None,
            )
            axis.axvline(
                cutoff_boundary_nm,
                color="#D55E00",
                linewidth=1.1,
                linestyle=":",
                label="Cut boundary" if axis_index == 0 else None,
            )

    snr = reference["snr_proxy"].to_numpy(dtype=np.float64)
    axes[0].plot(
        wavelength,
        np.where(np.isfinite(snr) & (snr > 0.0), snr, np.nan),
        color="black",
        linewidth=1.2,
        label="Reference SNR proxy",
    )
    axes[0].axhline(
        snr_threshold,
        color="#0072B2",
        linewidth=1.0,
        linestyle="--",
        label=f"Threshold = {snr_threshold:g}",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("SNR proxy")
    axes[0].legend(frameon=False)

    reflectance_span = (
        reflectance["median_sample_q99"].to_numpy(dtype=np.float64)
        - reflectance["median_sample_q01"].to_numpy(dtype=np.float64)
    )
    axes[1].plot(wavelength, reflectance_span, color="#0072B2", linewidth=1.2)
    axes[1].set_ylabel("Reflectance\n1–99% span")

    fraction_series = (
        ("max_nonfinite_fraction", "Non-finite", "#CC79A7"),
        ("max_negative_fraction", "Reflectance < 0", "#D55E00"),
        ("max_above_one_fraction", "Reflectance > 1", "#0072B2"),
    )
    for column, label, color in fraction_series:
        axes[2].plot(
            wavelength,
            reflectance[column],
            color=color,
            linewidth=1.0,
            label=label,
        )
    axes[2].set_ylabel("Maximum sample\nfraction")
    axes[2].set_ylim(bottom=0.0)
    axes[2].legend(frameon=False)

    snv_span = (
        snv["median_sample_q99"].to_numpy(dtype=np.float64)
        - snv["median_sample_q01"].to_numpy(dtype=np.float64)
    )
    axes[3].plot(wavelength, snv_span, color="#0072B2", linewidth=1.2)
    axes[3].set_xlabel("Wavelength [nm]")
    axes[3].set_ylabel("SNV\n1–99% span")
    for axis in axes:
        axis.grid(True, which="major", alpha=0.3)
        axis.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ranked_snv_spectra(
    spectra: pd.DataFrame,
    reflectance_summary: pd.DataFrame,
    snv_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare reflectance and SNV spectra for ranked anomaly candidates."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=180, sharex=True)
    _shade_low_snr_bands(axes[0], reflectance_summary)
    _shade_low_snr_bands(axes[1], snv_summary)
    metric_names = spectra["metric_name"].dropna().unique()
    candidate_count = spectra["candidate_id"].nunique()
    candidate_label = (
        f"Top {candidate_count} spike candidates"
        if len(metric_names) == 1 and metric_names[0] == "max_abs_second_difference"
        else f"Top {candidate_count} ranked candidates"
    )
    for index, (_, group) in enumerate(spectra.groupby("candidate_id", sort=False)):
        ordered = group.sort_values("band_index")
        line_label = candidate_label if index == 0 else None
        axes[0].plot(
            ordered["wavelength_nm"],
            ordered["reflectance"],
            color="#D55E00",
            linewidth=0.7,
            alpha=0.25,
            label=line_label,
        )
        axes[1].plot(
            ordered["wavelength_nm"],
            ordered["snv"],
            color="#D55E00",
            linewidth=0.7,
            alpha=0.25,
            label=line_label,
        )
    ordered_reflectance = reflectance_summary.sort_values("band_index")
    ordered_summary = snv_summary.sort_values("band_index")
    axes[0].plot(
        ordered_reflectance["wavelength_nm"],
        ordered_reflectance["median_sample_median"],
        color="black",
        linewidth=1.5,
        label="Median of sample medians",
    )
    axes[1].plot(
        ordered_summary["wavelength_nm"],
        ordered_summary["median_sample_median"],
        color="black",
        linewidth=1.5,
        label="Median of sample medians",
    )
    axes[0].set_ylabel("Reflectance")
    axes[1].set_xlabel("Wavelength [nm]")
    axes[1].set_ylabel("SNV")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(True, which="major", alpha=0.3)
        axis.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
