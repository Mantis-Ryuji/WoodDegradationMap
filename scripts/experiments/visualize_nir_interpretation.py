"""Plot M11 NIR means and derivative zooms, optionally for individual samples.

No raw spectra are read. Pseudoabsorbance was calculated pixelwise before the
saved within-sample means. Only x is converted to wavenumber; the saved SG
second derivative remains with respect to wavelength (nm).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
from matplotlib.transforms import ScaledTranslation
from scipy.signal import savgol_filter

from visualize_ftir import (
    DERIVATIVE_LIMIT, SPECTRUM_WIDTH_INCHES, WAVENUMBER_START,
    SelectedCurve, load_selection, save_figure,
)


KINDS = ("snv", "pseudoabsorbance", "second_derivative")
REGIONS = (("(a)", 8700.0, 8000.0), ("(b)", 6600.0, 5400.0), ("(c)", 4500.0, None))
FIGURE_HEIGHT_INCHES = 10.4
SAMPLE_FIGURE_HEIGHT_INCHES = 7.4
BAND_NUMBER_SIZE = 14


@dataclass(frozen=True)
class NirSummary:
    """Saved display-cluster means and their kind-specific pixel counts."""

    wavelength: np.ndarray
    palette: tuple[str, ...]
    sample_ids: tuple[str, ...]
    values: np.ndarray  # kind x sample x cluster x band
    pixels: np.ndarray  # sample x cluster; valid pseudoabsorbance pixels
    excluded: np.ndarray
    snv_pixels: np.ndarray


@dataclass(frozen=True)
class PlotCurve:
    """One observed curve with its display color and explicit legend label."""

    cluster: int
    label: str
    values: np.ndarray


@dataclass(frozen=True)
class BandAnnotation:
    """Literature position, not a peak detected or a component identified in this data."""

    number: int
    upper: float
    lower: float
    short_text: str

    @property
    def position(self) -> float:
        return (self.upper + self.lower) / 2

    @property
    def caption(self) -> str:
        return str(self.number)


# Schwanninger et al. (2011), Table 2; full assignments and integrated remarks
# are recorded in outputs/nir_interpretation/band_assignments.md.
BANDS = (
    BandAnnotation(1, 8547, 8547, "2nd OT asym. C–H str.\nHC=CH"),
    BandAnnotation(2, 8420, 8370, "2nd OT C–H str."),
    BandAnnotation(3, 8250, 8160, "2nd OT C–H str."),
    BandAnnotation(4, 7321, 7315, "1st OT C–H str.\n+ C–H def."),
    BandAnnotation(5, 7003, 6993, "1st OT O–H str.\n+ H₂O"),
    BandAnnotation(6, 6974, 6954, "1st OT O–H str."),
    BandAnnotation(7, 6757, 6757, "1st OT O–H str."),
    BandAnnotation(8, 6450, 6450, "1st OT O–H str."),
    BandAnnotation(9, 6281, 6281, "1st OT O–H str."),
    BandAnnotation(10, 5980, 5963, "1st OT Cₐᵣ–H str."),
    BandAnnotation(11, 5890, 5890, "1st OT C–H str."),
    BandAnnotation(12, 5865, 5865, "1st OT C–H str."),
    BandAnnotation(13, 5795, 5795, "1st OT C–H str."),
    BandAnnotation(14, 5776, 5776, "1st OT C–H str."),
    BandAnnotation(15, 5577, 5577, "1st OT C–H str."),
    BandAnnotation(16, 5464, 5464, "O–H str.\n+ 2nd OT C–O str."),
    BandAnnotation(17, 5220, 5150, "O–H asym. str.\n+ O–H def. (H₂O)"),
    BandAnnotation(18, 4392, 4392, "O–H + C–C str.\nand/or C–H str. + def."),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path,
        default=root / "outputs/experiments/global_v1/results/figures/global_k8_v1",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/nir_interpretation")
    parser.add_argument("--scope", choices=("summary", "samples", "both"), default="summary",
                        help="Generate only the requested figure group; default: summary.")
    parser.add_argument("--sample-ids", nargs="+", help="Generate only these target samples.")
    parser.add_argument(
        "--selection", type=Path, default=Path(__file__).with_name("ftir_plot_selection.yaml"),
        help="Reuse the sample/cluster selection and cross-sample references from the FT-IR plots.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true", help="Replace this script's outputs.")
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("DPI must be positive")
    if args.sample_ids is not None and args.scope == "summary":
        parser.error("--sample-ids requires --scope samples or both")
    return args


def load_summary(source: Path) -> NirSummary:
    """Validate the small saved tables without loading any raw pixel data."""
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    contract, definitions = report["contract"], report["definitions"]
    if contract["K"] != 8 or contract["repeat"] != 1 or "M11" not in contract["conditions"]:
        raise ValueError("Expected the saved M11 K8 repeat-1 report")
    sample_ids = tuple(report["sample_ids"])
    palette = tuple(definitions["palette"])
    if not sample_ids or len(set(sample_ids)) != len(sample_ids) or len(palette) != 9:
        raise ValueError("Invalid sample IDs or display-cluster palette")
    if any(not isinstance(sid, str) or not sid.startswith("KYOw")
           or not sid[4:].isdigit() for sid in sample_ids):
        raise ValueError("Expected KYOw sample IDs")
    grid = pd.read_csv(source / "wavelengths.csv")
    wavelength = grid.wavelength_nm.to_numpy(dtype=float)
    bands = [f"band_{index:03d}" for index in range(256)]
    if grid.band.tolist() != bands or not np.isfinite(wavelength).all() \
            or not np.all(wavelength > 0) or not np.all(np.diff(wavelength) > 0) \
            or not np.allclose(wavelength, np.linspace(wavelength[0], wavelength[-1], 256),
                               atol=1e-5, rtol=0):
        raise ValueError("Expected the saved uniform 256-band wavelength grid")
    expected_sg = {"window_length": 7, "polyorder": 2, "deriv": 2, "mode": "interp", "axis": -1}
    if any(definitions["sg"].get(key) != value for key, value in expected_sg.items()) \
            or not np.isclose(definitions["sg"]["delta_nm"], wavelength[1] - wavelength[0]):
        raise ValueError("Saved SG settings differ from the reference figure")
    expected = pd.MultiIndex.from_product(
        [sample_ids, range(1, 9)], names=["sample_id", "cluster"],
    )
    counts = pd.read_csv(source / "M11/spectrum_counts.csv").set_index(["sample_id", "cluster"])
    if not counts.index.is_unique or set(counts.index) != set(expected) \
            or not counts.condition.eq("M11").all():
        raise ValueError("Saved counts do not cover the expected M11 sample/cluster pairs")
    counts = counts.loc[expected]
    columns = ["pixels", "pseudoabsorbance_pixels", "pseudoabsorbance_excluded_pixels"]
    numbers = counts[columns].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or np.any(numbers < 0) \
            or not np.equal(numbers, np.floor(numbers)).all() \
            or not np.array_equal(numbers[:, 0], numbers[:, 1] + numbers[:, 2]):
        raise ValueError("Invalid saved valid/excluded pixel counts")
    pixels = counts.pseudoabsorbance_pixels.to_numpy(dtype=np.int64)
    spectra = pd.read_csv(source / "M11/sample_spectra.csv")
    curves = []
    for kind in KINDS:
        table = spectra.loc[spectra.kind == kind].set_index(["sample_id", "cluster"])
        if not table.index.is_unique or set(table.index) != set(expected) \
                or not table.condition.eq("M11").all():
            raise ValueError(f"Invalid sample/cluster coverage for {kind}")
        table = table.loc[expected]
        values = table[bands].to_numpy(dtype=float)
        kind_pixels = counts.pixels.to_numpy(dtype=np.int64) if kind == "snv" else pixels
        if not np.array_equal(table.pixel_count, kind_pixels) \
                or not np.isfinite(values[kind_pixels > 0]).all() \
                or not np.isnan(values[kind_pixels == 0]).all():
            raise ValueError(f"Invalid {kind} values or contributor counts")
        curves.append(values.reshape(len(sample_ids), 8, 256))
    return NirSummary(
        wavelength, palette, sample_ids, np.stack(curves), pixels.reshape(-1, 8),
        counts.pseudoabsorbance_excluded_pixels.to_numpy(dtype=np.int64).reshape(-1, 8),
        counts.pixels.to_numpy(dtype=np.int64).reshape(-1, 8),
    )


def aggregate_means(summary: NirSummary, source: Path) -> np.ndarray:
    """Check equal-sample means against the saved reference and SG settings."""
    means = np.empty((len(KINDS), 8, 256), dtype=float)
    saved = pd.read_csv(source / "M11/representative_spectra.csv")
    for cluster in range(8):
        for ki, kind in enumerate(KINDS):
            counts = summary.snv_pixels if kind == "snv" else summary.pixels
            present = counts[:, cluster] > 0
            if not present.any():
                raise ValueError(f"Cluster {cluster + 1}: no valid {kind} contributors")
            values = summary.values[ki, present, cluster]
            means[ki, cluster] = values.mean(axis=0)
            reference = saved.loc[(saved.kind == kind) & (saved.cluster == cluster + 1)]
            reference = reference.sort_values("wavelength_nm")
            if len(reference) != 256 or not reference.condition.eq("M11").all() \
                    or not np.allclose(reference.wavelength_nm, summary.wavelength,
                                       atol=1e-8, rtol=0) \
                    or not np.allclose(reference["mean"], means[ki, cluster],
                                       atol=1e-12, rtol=1e-10) \
                    or not reference.sample_count.eq(int(present.sum())).all() \
                    or not reference.pixel_count.eq(int(counts[:, cluster].sum())).all():
                raise ValueError(f"Cluster {cluster + 1}: {kind} differs from saved reference")
    derived = savgol_filter(
        means[1], window_length=7, polyorder=2, deriv=2,
        delta=float(summary.wavelength[1] - summary.wavelength[0]), axis=-1, mode="interp",
    )
    if not np.allclose(derived, means[2], atol=1e-12, rtol=1e-8):
        raise ValueError("Saved derivatives do not match SG applied to the absorbance means")
    return means


def plot_curves(
    axis: Axes, wavenumber: np.ndarray, curves: tuple[PlotCurve, ...], palette: tuple[str, ...],
    upper: float, lower: float, major: float, minor: float,
) -> None:
    """Keep the existing line, wavenumber-axis and grid style on every panel."""
    for curve in curves:
        axis.plot(wavenumber, curve.values, color=palette[curve.cluster],
                  linewidth=1.4, label=curve.label)
    axis.set_xlim(upper, lower)
    axis.xaxis.set_major_locator(MultipleLocator(major))
    axis.xaxis.set_minor_locator(MultipleLocator(minor))
    axis.set_axisbelow(True)
    axis.grid(which="major", color="0.83", linewidth=0.7)
    axis.grid(which="minor", axis="x", color="0.91", linewidth=0.5, linestyle=":")
    axis.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=12)


def region_limits(
    curves: np.ndarray, wavenumber: np.ndarray, upper: float, lower: float,
) -> tuple[float, float]:
    """Fit all curves, including the linearly drawn values at the region edges."""
    selected = curves[:, (wavenumber >= lower) & (wavenumber <= upper)]
    measured_bounds = [max(lower, float(wavenumber.min())), min(upper, float(wavenumber.max()))]
    if measured_bounds[0] >= measured_bounds[1]:
        raise ValueError("Zoom region does not intersect measured wavelengths")
    edges = np.array([np.interp(measured_bounds, wavenumber[::-1], curve[::-1])
                      for curve in curves])
    values = np.concatenate((selected, edges), axis=1)
    low, high = float(values.min()), float(values.max())
    margin = 0.08 * max(high - low, abs(low), abs(high), 1e-12)
    return low - margin, high + margin


def zoom_regions(wavenumber: np.ndarray) -> tuple[tuple[str, float, float], ...]:
    """Resolve 'last' from the measured grid, without extending the spectrum."""
    last = float(wavenumber[-1])
    return tuple((label, upper, last if lower is None else lower)
                 for label, upper, lower in REGIONS)


def in_zoom(band: BandAnnotation, wavenumber: np.ndarray) -> bool:
    return any(lower <= band.lower <= band.upper <= upper
               for _, upper, lower in zoom_regions(wavenumber))


def annotate_bands(
    axis: Axes, wavenumber: np.ndarray, curves: tuple[PlotCurve, ...],
    bands: tuple[BandAnnotation, ...], *, captions: bool, allow_headroom: bool = False,
) -> None:
    """Draw axis-based arrows and place numbers inside the top, clear of all spectra."""
    if not curves or not bands:
        return
    for band in bands:
        if not float(wavenumber.min()) <= band.lower <= band.upper <= float(wavenumber.max()):
            raise ValueError(f"Band {band.number}: annotation outside measured wavelengths")
    size = axis.figure.get_size_inches()
    height_pt = axis.get_position().height * size[1] * 72
    font = FontProperties(family="DejaVu Sans", weight="bold", size=BAND_NUMBER_SIZE)
    renderer = axis.figure.canvas.get_renderer()
    padding = renderer.points_to_pixels(3)
    curve_labels = {curve.label for curve in curves}
    transform = axis.get_xaxis_transform()
    arrow_top = 14 / height_pt
    if captions:
        for attempt in range(2):
            occupied = [text.get_window_extent(renderer).padded(padding)
                        for text in axis.texts if text.get_text()]
            spectra = [line.get_path().transformed(line.get_transform())
                       for line in axis.lines if line.get_label() in curve_labels]
            numbers = []
            for band in bands:
                number_transform = transform
                if band.number == 6:
                    number_transform += ScaledTranslation(6 / 72, 0, axis.figure.dpi_scale_trans)
                number = axis.text(
                    band.position, 1, band.caption, transform=number_transform,
                    ha="center", va="top", fontproperties=font, color="0.1", clip_on=True,
                    bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none"},
                    zorder=6,
                )
                number.set_gid(f"band-number-{band.number}")
                numbers.append(number)
                # Keep each chosen x position and stagger within the upper half.
                for inset in np.arange(6, height_pt / 2, BAND_NUMBER_SIZE + 8):
                    number.set_y(1 - inset / height_pt)
                    bounds = number.get_window_extent(renderer).padded(padding)
                    if not all(axis.bbox.contains(x, y) for x, y in bounds.get_points()) \
                            or any(bounds.overlaps(other) for other in occupied) \
                            or any(path.intersects_bbox(bounds, filled=False) for path in spectra):
                        continue
                    occupied.append(bounds)
                    break
                else:
                    break
            else:
                break
            for number in numbers:
                number.remove()
            if not allow_headroom or attempt:
                raise ValueError(f"Band {band.number}: no clear number position in the upper half")
            # Some sample zooms need room for two rows; keep the fixed overview limits.
            low, high = axis.get_ylim()
            reserve = min((2 * (BAND_NUMBER_SIZE + 8) + 6) / height_pt, 0.45)
            axis.set_ylim(low, low + (high - low) / (1 - reserve))
    for band in bands:
        endpoints = (band.upper,) if band.upper == band.lower else (band.upper, band.lower)
        if len(endpoints) == 2:
            shade = axis.axvspan(band.lower, band.upper, facecolor="0.70", alpha=0.18,
                                edgecolor="none", zorder=0.6)
            shade.set_gid(f"band-range-{band.number}")
        for index, position in enumerate(endpoints):
            guide, = axis.plot([position, position], [arrow_top, 1], transform=transform,
                               color="black", linewidth=0.8, linestyle=":", zorder=1.4)
            guide.set_gid(f"band-guide-{band.number}-{index}")
            arrow = axis.annotate(
                "", xy=(position, arrow_top), xytext=(position, 0),
                xycoords=transform, textcoords=transform,
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 0.8,
                            "mutation_scale": 9, "shrinkA": 0, "shrinkB": 0},
                zorder=3,
            )
            arrow.set_gid(f"band-arrow-{band.number}-{index}")


def annotation_description(wavenumber: np.ndarray) -> str:
    """Keep the annotation rule and numbering in every PNG's provenance."""
    return (
        "Literature assignments: Schwanninger et al. (2011), Table 2; "
        "see band_assignments.md for remarks and transcription corrections. "
        "Arabic band numbers 1-18 retain the supplied order; former band 19 (4365 cm^-1) removed. "
        "No vibration text on figures. Black upward arrows start at the x axis and span 14 pt. "
        "Black dotted guides extend from arrow tips to the upper frame. "
        "Ranges have arrows/guides at both endpoints and pale gray fill (0.70) with alpha=0.18. "
        "Numbers sit inside the upper half of each plot, "
        "staggered vertically to avoid all spectra and other labels. "
        "Number 6 alone is shifted 6 pt right of its band center; all other numbers are centered. "
        "A small white text background separates numbers from dotted guides; "
        "zoom upper limits gain headroom only when needed for number placement. "
        "Arrow positions are fixed to the axis; cluster 2 remains plotted. "
        "Fixed literature x positions; no detected-peak or component-identification claim. "
        "Numbers for zoomed positions appear only on zoom panels; overview retains guides. "
        f"Zoom regions (cm^-1)={zoom_regions(wavenumber)}; widths follow their spans. "
        "Major ticks=100, minor ticks=20 cm^-1. "
        f"Band numbers/endpoints={[(band.caption, band.upper, band.lower) for band in BANDS]}. "
    )


def mark_region(
    axis: Axes, label: str, upper: float, lower: float, limits: tuple[float, float],
) -> None:
    """Draw the exact x/y extent displayed in the corresponding zoom panel."""
    low, high = limits
    axis.add_patch(Rectangle(
        (lower, low), upper - lower, high - low, fill=False, edgecolor="black",
        linewidth=1.2, linestyle=":", zorder=5,
    ))
    axis.annotate(label, ((upper + lower) / 2, min(high, axis.get_ylim()[1])), xytext=(0, 4),
                  textcoords="offset points", ha="center", va="bottom", fontsize=10,
                  color="0.25", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})


def draw_derivative_panels(
    derivative_axis: Axes, zoom_axes: tuple[Axes, ...], wavenumber: np.ndarray,
    curves: tuple[PlotCurve, ...], palette: tuple[str, ...],
) -> None:
    """Share the overview, overflow marks and three derivative zooms across figures."""
    regions = zoom_regions(wavenumber)
    visible = wavenumber <= WAVENUMBER_START
    plot_curves(derivative_axis, wavenumber, curves, palette,
                WAVENUMBER_START, float(wavenumber[-1]), 500, 100)
    derivative_axis.set_ylim(-DERIVATIVE_LIMIT, DERIVATIVE_LIMIT)
    derivative_axis.set_yticks(np.linspace(-DERIVATIVE_LIMIT, DERIVATIVE_LIMIT, 11))
    derivative_axis.set_ylabel(r"$d^2A/d\lambda^2$ (nm$^{-2}$)", fontsize=12)
    derivative_axis.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4), useMathText=True)
    derivative_axis.axhline(0, color="0.55", linewidth=0.6, zorder=0)
    overflow = False
    for curve in curves:
        values = curve.values
        for outside, boundary, marker in (
            ((values > DERIVATIVE_LIMIT) & visible, DERIVATIVE_LIMIT, "^"),
            ((values < -DERIVATIVE_LIMIT) & visible, -DERIVATIVE_LIMIT, "v"),
        ):
            overflow |= bool(outside.any())
            derivative_axis.plot(wavenumber[outside], np.full(outside.sum(), boundary),
                                  linestyle="none", marker=marker, markersize=4,
                                  color=palette[curve.cluster], clip_on=False, zorder=4)
    if overflow:
        derivative_axis.text(0.99, 0.97, "Triangles: outside y-range",
                              transform=derivative_axis.transAxes, ha="right", va="top",
                              fontsize=8, color="0.35")
    if not curves:
        derivative_axis.text(0.5, 0.5, "No valid pseudoabsorbance spectra", ha="center",
                              va="center", transform=derivative_axis.transAxes)
    for column, (axis, (label, upper, lower)) in enumerate(zip(zoom_axes, regions, strict=True)):
        plot_curves(axis, wavenumber, curves, palette, upper, lower, 100, 20)
        axis.set_xticks(np.arange(upper, lower - 1e-8, -100))
        if column == len(REGIONS) - 1:
            # The narrow final panel retains the same label size without clipping.
            axis.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=12, loc="right")
        if curves:
            limits = region_limits(np.stack([curve.values for curve in curves]),
                                   wavenumber, upper, lower)
            axis.set_ylim(*limits)
        else:
            axis.set_ylim(-DERIVATIVE_LIMIT, DERIVATIVE_LIMIT)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.tick_params(axis="both", labelsize=9)
        axis.text(0.97, 0.96, label, transform=axis.transAxes, ha="right", va="top",
                  fontsize=11, color="0.25",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4), useMathText=True)
        axis.axhline(0, color="0.55", linewidth=0.6, zorder=0)
        if column == 0:
            axis.set_ylabel(r"$d^2A/d\lambda^2$ (nm$^{-2}$)", fontsize=12)
        selected = tuple(band for band in BANDS if lower <= band.lower <= band.upper <= upper)
        annotate_bands(axis, wavenumber, curves, selected, captions=True, allow_headroom=True)
        if curves:
            mark_region(derivative_axis, label, upper, lower, axis.get_ylim())
    annotate_bands(derivative_axis, wavenumber, curves,
                   tuple(band for band in BANDS if in_zoom(band, wavenumber)), captions=False)
    annotate_bands(derivative_axis, wavenumber, curves,
                   tuple(band for band in BANDS if not in_zoom(band, wavenumber)), captions=True)


def add_legend(figure: plt.Figure, axis: Axes) -> None:
    """Keep one shared legend, including explicit labels for cross-sample curves."""
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return
    figure.legend(
        handles, labels, title="Cluster ID", ncol=len(handles), loc="lower center",
        bbox_to_anchor=(0.53, 0.008), frameon=True, facecolor="white",
        edgecolor="0.85", framealpha=0.90, handlelength=2, columnspacing=1.2,
        fontsize=14, title_fontsize=16, borderpad=0.4, labelspacing=0.25,
    )


def render_figure(
    summary: NirSummary, means: np.ndarray, path: Path,
    dpi: int, description: str, *, overwrite: bool = False,
) -> None:
    """Draw SNV/absorbance, the full derivative, and three derivative zooms."""
    wavenumber = 1e7 / summary.wavelength
    last = float(wavenumber[-1])
    figure = plt.figure(figsize=(SPECTRUM_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    figure.subplots_adjust(left=0.07, right=0.98, bottom=1.32 / FIGURE_HEIGHT_INCHES,
                           top=1 - 0.12 / FIGURE_HEIGHT_INCHES)
    # Preserve panel heights while removing the former rows for outside band numbers.
    rows = figure.add_gridspec(3, 1, height_ratios=(2.73, 4.0, 3.6), hspace=0.22)
    overview = rows[0].subgridspec(1, 2, wspace=0.20)
    snv_axis = figure.add_subplot(overview[0, 0])
    absorbance_axis = figure.add_subplot(overview[0, 1])
    for kind, axis in enumerate((snv_axis, absorbance_axis)):
        curves = tuple(PlotCurve(cluster, str(cluster), means[kind, cluster - 1])
                       for cluster in range(1, 9))
        plot_curves(axis, wavenumber, curves, summary.palette, WAVENUMBER_START, last, 1000, 500)
    snv_axis.set_ylabel("Reflectance (SNV)", fontsize=12)
    snv_axis.set_ylim(-2, 2)
    snv_axis.yaxis.set_major_locator(MultipleLocator(0.5))
    absorbance_axis.set_ylabel("Absorbance", fontsize=12)
    absorbance_axis.set_ylim(0, 1.1)
    absorbance_axis.yaxis.set_major_locator(MultipleLocator(0.2))
    derivative_axis = figure.add_subplot(rows[1])
    widths = tuple(upper - lower for _, upper, lower in zoom_regions(wavenumber))
    zoom_grid = rows[2].subgridspec(1, 3, width_ratios=widths, wspace=0.25)
    zoom_axes = tuple(figure.add_subplot(zoom_grid[0, column]) for column in range(3))
    curves = tuple(PlotCurve(cluster, str(cluster), means[2, cluster - 1]) for cluster in range(1, 9))
    draw_derivative_panels(derivative_axis, zoom_axes, wavenumber, curves, summary.palette)
    add_legend(figure, derivative_axis)
    save_figure(figure, path, dpi, description, overwrite=overwrite, crop_to_content=False)


def render_sample(
    summary: NirSummary, sample_id: str, selection: tuple[SelectedCurve, ...],
    path: Path, dpi: int, source: Path, *, overwrite: bool = False,
) -> None:
    """Use the same saved means, selected clusters and reference curves as FT-IR NIR plots."""
    curves, records = [], []
    ordered = sorted(selection, key=lambda curve: (
        curve.source_sample_id != sample_id, curve.source_sample_id, curve.cluster,
    ))
    for selected in ordered:
        sample_index = summary.sample_ids.index(selected.source_sample_id)
        cluster_index = selected.cluster - 1
        pixels = int(summary.pixels[sample_index, cluster_index])
        label = selected.legend_label(sample_id)
        records.append({
            "source_sample_id": selected.source_sample_id, "cluster": selected.cluster,
            "legend_label": label, "pixel_count": pixels,
            "excluded_pixels": int(summary.excluded[sample_index, cluster_index]),
            "plotted": pixels > 0,
        })
        if pixels:
            curves.append(PlotCurve(selected.cluster, label,
                                    summary.values[2, sample_index, cluster_index]))
        else:
            print(f"{sample_id}: omit {label}; no valid pseudoabsorbance pixels", flush=True)
    figure = plt.figure(figsize=(SPECTRUM_WIDTH_INCHES, SAMPLE_FIGURE_HEIGHT_INCHES))
    figure.subplots_adjust(left=0.07, right=0.98, bottom=1.32 / SAMPLE_FIGURE_HEIGHT_INCHES,
                           top=1 - 0.30 / SAMPLE_FIGURE_HEIGHT_INCHES)
    wavenumber = 1e7 / summary.wavelength
    rows = figure.add_gridspec(2, 1, height_ratios=(4.0, 3.6), hspace=0.22)
    derivative_axis = figure.add_subplot(rows[0])
    widths = tuple(upper - lower for _, upper, lower in zoom_regions(wavenumber))
    zoom_grid = rows[1].subgridspec(1, 3, width_ratios=widths, wspace=0.25)
    zoom_axes = tuple(figure.add_subplot(zoom_grid[0, column]) for column in range(3))
    draw_derivative_panels(derivative_axis, zoom_axes, wavenumber,
                           tuple(curves), summary.palette)
    add_legend(figure, derivative_axis)
    description = (
        f"{source}; {sample_id}; M11 K8 repeat 1; saved within-sample cluster means; "
        "same aggregation and selected curves as outputs/ftir/nir; no across-sample averaging. "
        "Pixelwise A=-log10(R); SG(window=7, polyorder=2, deriv=2, delta in nm, mode=interp); "
        "x=1e7/wavelength_nm; y remains d2A/dlambda2 in nm^-2. "
        "Derivative overview and zooms only; no title or IQR. "
        f"Canvas={SPECTRUM_WIDTH_INCHES:g} x {SAMPLE_FIGURE_HEIGHT_INCHES:g} inches; "
        "dotted rectangles identify zoom extents; triangles mark overview y overflow; "
        "zoom y-limits are fitted to the selected curves. "
        f"{annotation_description(wavenumber)}"
        f"Selected curves={json.dumps(records)}."
    )
    save_figure(figure, path, dpi, description, overwrite=overwrite, crop_to_content=False)


def main() -> int:
    args = parse_args()
    source, output = args.report_dir.resolve(), args.output_dir.resolve()
    summary = load_summary(source)
    make_summary = args.scope in ("summary", "both")
    make_samples = args.scope in ("samples", "both")
    selections = load_selection(args.selection, list(summary.sample_ids)) if make_samples else {}
    target_ids = args.sample_ids if args.sample_ids is not None else list(summary.sample_ids)
    if len(set(target_ids)) != len(target_ids) or not set(target_ids) <= set(summary.sample_ids):
        raise ValueError("--sample-ids must contain unique saved sample IDs")
    means = aggregate_means(summary, source) if make_summary else None
    png_name = "M11_equal_sample.png"
    paths = [output / png_name] if make_summary else []
    if make_samples:
        paths.extend(output / "each_nir_spectra" / f"{sid}.png" for sid in target_ids)
    for path in paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite")
    description = (
        f"Source: {source}; M11 K8 repeat 1; saved M00-aligned display clusters 1-8. "
        "Pixelwise A=-log10(R), retaining pixels positive and finite in all 256 bands. "
        "SG(window=7, polyorder=2, deriv=2, delta in nm, mode=interp). "
        "x=1e7/wavelength_nm; y remains d2A/dlambda2 in nm^-2. "
        "Observed pixelwise SNV is averaged within sample, then equally across present samples. "
        "Saved sample means only; no raw data scan, no refit, no IQR; equal-sample aggregation. "
        "Overview x starts at 10000 cm^-1; derivative overview y=+/-2.5e-4. "
        "dotted rectangles appear only on the derivative overview and identify derivative zooms. "
        f"Canvas={SPECTRUM_WIDTH_INCHES:g} x {FIGURE_HEIGHT_INCHES:g} inches; no figure title."
        f" {annotation_description(1e7 / summary.wavelength)}"
    )
    output.mkdir(parents=True, exist_ok=True)
    if means is not None:
        render_figure(summary, means, output / png_name, args.dpi, description,
                      overwrite=args.overwrite)
        print(f"Saved {output / png_name}; validated reference means and SG consistency", flush=True)
    if make_samples:
        sample_output = output / "each_nir_spectra"
        sample_output.mkdir(parents=True, exist_ok=True)
        for index, sample_id in enumerate(target_ids, start=1):
            render_sample(summary, sample_id, selections[sample_id],
                          sample_output / f"{sample_id}.png", args.dpi, source,
                          overwrite=args.overwrite)
            print(f"[{index}/{len(target_ids)}] Saved {sample_id}.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
