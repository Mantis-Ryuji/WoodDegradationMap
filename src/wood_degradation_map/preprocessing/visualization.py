"""Title-free figures emitted by production preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


def configure_spectrum_axis(
    axis: plt.Axes,
    *,
    representation: Literal["reflectance", "snv"],
) -> None:
    """Apply the shared display scale without changing the plotted values."""

    if representation == "reflectance":
        limits, ticks, label = (0.0, 1.0), np.linspace(0.0, 1.0, 6), "Reflectance"
    elif representation == "snv":
        limits, ticks, label = (-2.0, 2.0), np.linspace(-2.0, 2.0, 9), "SNV"
    else:
        raise ValueError(f"Unknown spectral representation: {representation}")
    axis.set_yticks(ticks)
    axis.set_ylim(*limits)
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.set_ylabel(label)


def configure_wavelength_axis(axis: plt.Axes, wavelength: np.ndarray) -> None:
    """Show both grid endpoints and 200 nm ticks without horizontal padding."""

    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or not np.isfinite(wavelength).all()
        or not np.all(np.diff(wavelength) > 0.0)
    ):
        raise ValueError("Wavelengths must be a finite, strictly increasing vector")
    left, right = float(wavelength[0]), float(wavelength[-1])
    interior = np.arange(1000.0, right, 200.0)
    interior = interior[interior > left]
    ticks = np.concatenate(([left], interior, [right]))
    labels = [f"{left:.2f}", *(f"{value:.0f}" for value in interior), f"{right:.2f}"]
    axis.set_xticks(ticks, labels=labels)
    axis.margins(x=0)
    axis.set_xlim(left, right)
    axis.tick_params(
        axis="both", which="both", direction="out",
        bottom=True, left=True, top=False, right=False, pad=5,
    )
    axis.tick_params(axis="x", labelsize=9)
    # Keep the endpoint text clear of the nearby regular ticks in two-column figures.
    visible_labels = axis.get_xticklabels()
    if visible_labels:
        visible_labels[0].set_horizontalalignment("right")
        visible_labels[-1].set_horizontalalignment("left")


def _mark_outside_limits(
    axis: plt.Axes,
    wavelength: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
) -> None:
    """Mark hidden values at the display boundary rather than clipping the data."""

    lower, upper = axis.get_ylim()
    for mask, boundary, marker in (
        (values < lower, lower, "v"),
        (values > upper, upper, "^"),
    ):
        if np.any(mask):
            axis.scatter(
                wavelength[mask], np.full(np.count_nonzero(mask), boundary),
                marker=marker, s=13, color=color, linewidths=0, clip_on=False, zorder=4,
            )


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
    if y_label.lower() in ("reflectance", "snv"):
        configure_spectrum_axis(
            axis, representation="snv" if y_label.lower() == "snv" else "reflectance",
        )
        for values in (lower, median, upper):
            _mark_outside_limits(axis, wavelength, values, color="#0072B2")
    else:
        axis.set_ylabel(y_label)
    configure_wavelength_axis(axis, wavelength)
    axis.grid(True, which="major", alpha=0.3)
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
    fig, axis = plt.subplots(figsize=(8.0, 4.0), dpi=180)
    axis.axvspan(
        wavelength_edges[0],
        wavelength_edges[-1] if cutoff_boundary_nm is None else cutoff_boundary_nm,
        color="#009E73", alpha=0.08, linewidth=0, label="Retained",
    )
    if cutoff_boundary_nm is not None:
        axis.axvspan(
            cutoff_boundary_nm, wavelength_edges[-1],
            color="#D55E00", alpha=0.10, linewidth=0, label="Excluded",
        )
        axis.axvline(
            cutoff_boundary_nm, color="#D55E00", linewidth=1.1, linestyle=":",
            label=f"Cut boundary = {cutoff_boundary_nm:.2f} nm",
        )

    snr = reference["snr_proxy"].to_numpy(dtype=np.float64)
    axis.plot(
        wavelength,
        np.where(np.isfinite(snr) & (snr > 0.0), snr, np.nan),
        color="black",
        linewidth=1.2,
        label="Reference SNR proxy",
    )
    axis.axhline(
        snr_threshold,
        color="#0072B2",
        linewidth=1.0,
        linestyle="--",
        label=f"Threshold = {snr_threshold:g}",
    )
    axis.set_yscale("log")
    axis.set_ylabel("SNR proxy")
    axis.set_xlabel("Wavelength [nm]")
    configure_wavelength_axis(axis, wavelength)
    axis.legend(
        frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
    )
    axis.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ranked_snv_spectra(
    spectra: pd.DataFrame,
    reflectance_summary: pd.DataFrame,
    snv_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Show each ranked SNV candidate separately on the common display scale.

    The reflectance summary argument is retained for existing report callers;
    individual panels now focus on the SNV quantity used for ranking.
    """

    if spectra.empty:
        raise ValueError("At least one ranked spectrum is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = list(spectra.sort_values("rank", kind="stable").groupby("candidate_id", sort=False))
    columns = min(4, len(groups))
    rows = (len(groups) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 2.5 * rows + 0.6),
        dpi=180, sharex=True, sharey=True, squeeze=False,
    )
    summary = snv_summary.sort_values("band_index")
    wavelength = summary["wavelength_nm"].to_numpy(dtype=np.float64)
    for index, (_, group) in enumerate(groups):
        axis = axes.flat[index]
        ordered = group.sort_values("band_index")
        candidate_wavelength = ordered["wavelength_nm"].to_numpy(dtype=np.float64)
        values = ordered["snv"].to_numpy(dtype=np.float64)
        record = ordered.iloc[0]
        _shade_low_snr_bands(axis, summary)
        axis.fill_between(
            wavelength, summary["median_sample_q01"], summary["median_sample_q99"],
            color="#0072B2", alpha=0.12, linewidth=0,
        )
        axis.plot(
            wavelength, summary["median_sample_median"],
            color="0.25", linewidth=1.0, linestyle="--",
        )
        axis.plot(candidate_wavelength, values, color="#D55E00", linewidth=1.0)
        configure_spectrum_axis(axis, representation="snv")
        _mark_outside_limits(axis, candidate_wavelength, values, color="#D55E00")
        axis.text(
            0.98, 0.96,
            f"#{int(record['rank'])}  {record['sample_id']}\n"
            f"pixel ({int(record['row'])}, {int(record['column'])})",
            transform=axis.transAxes, ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )
        if np.any((values < -2.0) | (values > 2.0)):
            axis.text(
                0.03, 0.04, f"min {values.min():.2f} / max {values.max():.2f}",
                transform=axis.transAxes, fontsize=8, color="#A54400",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
            )
        if index % columns:
            axis.set_ylabel("")
        if index // columns == rows - 1:
            axis.set_xlabel("Wavelength [nm]")
        configure_wavelength_axis(axis, wavelength)
        axis.grid(True, which="major", alpha=0.3)
        axis.tick_params(labelsize=8)
        # Endpoint labels and the first regular tick are close in these small panels.
        axis.tick_params(axis="x", labelrotation=45)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    for axis in axes.flat[len(groups):]:
        axis.set_visible(False)
    fig.legend(
        handles=[
            Line2D([], [], color="#D55E00", label="Ranked SNV candidate"),
            Line2D([], [], color="0.25", linestyle="--", label="Median of sample medians"),
            Patch(facecolor="#0072B2", alpha=0.12, label="Median sample 1–99% range"),
        ],
        loc="upper center", ncol=3, frameon=False, fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
