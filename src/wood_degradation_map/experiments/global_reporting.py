"""Observed spectral summaries, M00 matching, and static global-fit PNG/CSV reports."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import stat
from tempfile import TemporaryDirectory

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from .global_clustering import K, check_global_clustering, condition_paths, read_label_map
from .global_manifest import GLOBAL_CONDITIONS, GlobalData
from .input_validation import InputInventory
from .manifests import _digest, _read_json, _require, _write_json
from .oof_sanity import LABEL_COLORS, REPRESENTATIVES, occupancy_values

CONDITIONS = GLOBAL_CONDITIONS
REFERENCE = CONDITIONS.index("M00")
KINDS = ("reflectance", "snv", "pseudoabsorbance", "second_derivative")
PLOTTED_KINDS = (0, 1, 3)
Y_LABELS = ("Reflectance", "SNV", "Pseudoabsorbance", r"$d^2A/d\lambda^2$ (nm$^{-2}$)")
DEFAULT_DIRECTORY = "results/figures/global_k8_v1"
REPRESENTATIVE_FIGURE = "01_representative_samples_5x7.png"
REPRESENTATIVE_LABEL_FIGURE = "07_representative_samples_1x7.png"


@dataclass(frozen=True)
class SpectralSummaries:
    sample_ids: tuple[str, ...]
    wavelength: np.ndarray
    # sample x condition x original cluster x kind x wavelength; absent curves are NaN.
    values: np.ndarray
    pixels: np.ndarray
    pseudoabsorbance_pixels: np.ndarray
    # target condition x M00 cluster x target cluster; all common valid pixels.
    contingency: np.ndarray


def second_derivative(values: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    _require(wavelength.shape == (256,) and np.isfinite(wavelength).all()
             and np.all(np.diff(wavelength) > 0), "Invalid wavelength grid")
    delta = float(wavelength[1] - wavelength[0])
    _require(np.allclose(wavelength, np.linspace(wavelength[0], wavelength[-1], 256),
                        atol=1e-5, rtol=0), "SG requires the saved uniform wavelength grid")
    return savgol_filter(values, window_length=7, polyorder=2, deriv=2,
                         delta=delta, axis=-1, mode="interp")


def collect_spectra(
    experiment: Path, inventory: InputInventory, *, chunk_pixels: int = 2048,
) -> SpectralSummaries:
    """Read each spectral chunk once, keeping only sample/cluster sums in memory."""
    _require(type(chunk_pixels) is int and chunk_pixels > 0, "Invalid spectral chunk size")
    samples = tuple(sorted(inventory.samples, key=lambda s: s.sample_id))
    shape = (len(samples), len(CONDITIONS), K)
    values = np.full((*shape, len(KINDS), 256), np.nan, dtype=np.float64)
    counts, positive_counts = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    contingency = np.zeros((len(CONDITIONS), K, K), dtype=np.int64)
    wavelength: np.ndarray | None = None
    for si, sample in enumerate(samples):
        maps = np.stack([read_label_map(condition_paths(experiment, c)[0] / "maps"
                                        / f"{sample.sample_id}.npz", sample) for c in CONDITIONS])
        sums = np.zeros((len(CONDITIONS), K, 3, 256), dtype=np.float64)
        with h5py.File(sample.path, "r") as handle:
            _require(handle.attrs.get("sample_id") == sample.sample_id
                     and handle.attrs.get("saved_pixel_count") == sample.saved_pixel_count,
                     f"{sample.sample_id}: spectral identity differs")
            wave = handle["wavelength_nm"][:].astype(np.float64)
            second_derivative(np.zeros(256), wave)
            _require(np.allclose(wave, np.linspace(inventory.wavelength_start_nm,
                         inventory.wavelength_end_nm, 256), atol=1e-5, rtol=0),
                     "Wavelength grid differs from inventory")
            if wavelength is None:
                wavelength = wave
            _require(np.array_equal(wavelength, wave), "Samples have different wavelength grids")
            for key in ("reflectance", "snv"):
                _require(handle[key].shape == (sample.saved_pixel_count, 256)
                         and handle[key].dtype == np.dtype("float32"), f"Invalid {key} dataset")
            _require(handle["pixel_row_col"].shape == (sample.saved_pixel_count, 2)
                     and handle["pixel_row_col"].dtype.kind in "iu", "Invalid coordinates")
            seen = np.zeros((sample.height, sample.width), dtype=bool)
            for start in range(0, sample.saved_pixel_count, chunk_pixels):
                stop = min(start + chunk_pixels, sample.saved_pixel_count)
                coordinates = handle["pixel_row_col"][start:stop]
                _require(((coordinates >= 0) & (coordinates < [sample.height, sample.width])).all(),
                         "Spectral coordinates out of bounds")
                rows, columns = coordinates.T
                flat = rows.astype(np.int64) * sample.width + columns
                _require(len(np.unique(flat)) == len(flat) and not seen[rows, columns].any(),
                         "Duplicate spectral coordinate")
                seen[rows, columns] = True
                labels = maps[:, rows, columns].astype(np.int64) - 1
                _require(np.all((labels >= 0) & (labels < K)), "Spectrum on invalid map pixel")
                reflectance = handle["reflectance"][start:stop].astype(np.float64)
                snv = handle["snv"][start:stop].astype(np.float64)
                _require(np.isfinite(reflectance).all() and np.isfinite(snv).all(),
                         "Nonfinite observed reflectance/SNV")
                positive = np.all(reflectance > 0, axis=1)
                absorbance = -np.log10(reflectance[positive])
                for ci in range(len(CONDITIONS)):
                    contingency[ci] += np.bincount(
                        labels[REFERENCE] * K + labels[ci], minlength=K * K).reshape(K, K)
                    counts[si, ci] += np.bincount(labels[ci], minlength=K)
                    positive_counts[si, ci] += np.bincount(labels[ci, positive], minlength=K)
                    for cluster in range(K):
                        selected = labels[ci] == cluster
                        sums[ci, cluster, 0] += reflectance[selected].sum(axis=0)
                        sums[ci, cluster, 1] += snv[selected].sum(axis=0)
                        sums[ci, cluster, 2] += absorbance[labels[ci, positive] == cluster].sum(axis=0)
            _require(np.array_equal(seen, maps[REFERENCE] > 0), "Incomplete spectral coverage")
        for ci in range(len(CONDITIONS)):
            for cluster in range(K):
                n, valid = counts[si, ci, cluster], positive_counts[si, ci, cluster]
                if n:
                    values[si, ci, cluster, :2] = sums[ci, cluster, :2] / n
                if valid:
                    values[si, ci, cluster, 2] = sums[ci, cluster, 2] / valid
                    values[si, ci, cluster, 3] = second_derivative(
                        values[si, ci, cluster, 2], wavelength)
        print(f"Spectra {sample.sample_id}: {sample.saved_pixel_count} observed pixels", flush=True)
    _require(wavelength is not None, "No samples")
    return SpectralSummaries(tuple(s.sample_id for s in samples), wavelength, values,
                             counts, positive_counts, contingency)


def equal_sample_summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Exclude absent sample curves; never pool pixels or differentiate quantile curves."""
    valid = np.isfinite(values).all(axis=-1)
    _require(np.all(valid | np.isnan(values).all(axis=-1)), "Partially invalid sample curve")
    selected = values[valid]
    if not len(selected):
        empty = np.full(values.shape[-1], np.nan)
        return empty, empty.copy(), empty.copy(), 0
    return selected.mean(axis=0), *np.quantile(selected, (0.25, 0.75), axis=0), len(selected)


def match_snv(summary: SpectralSummaries) -> tuple[np.ndarray, np.ndarray]:
    """Match equal-sample observed SNV means directly to M00 by summed cosine."""
    shape = (len(summary.sample_ids), len(CONDITIONS), K)
    _require(summary.values.shape == (*shape, len(KINDS), 256)
             and summary.pixels.shape == shape, "Invalid spectral matching shape")
    empty = summary.pixels.sum(axis=0) == 0
    _require(not empty.any(), "Empty cluster for SNV matching: " + ", ".join(
        f"{CONDITIONS[ci]} cluster {k + 1}" for ci, k in np.argwhere(empty)))
    representatives = np.empty((len(CONDITIONS), K, 256), dtype=np.float64)
    for ci, condition in enumerate(CONDITIONS):
        for cluster in range(K):
            mean, _, _, _ = equal_sample_summary(summary.values[:, ci, cluster, KINDS.index("snv")])
            norm = np.linalg.norm(mean)
            _require(np.isfinite(mean).all() and np.isfinite(norm) and norm > 0,
                     f"Undefined or zero-norm SNV representative: {condition} cluster {cluster + 1}")
            representatives[ci, cluster] = mean / norm
    similarity = np.stack([representatives[REFERENCE] @ target.T for target in representatives])
    similarity = np.clip(similarity, -1, 1)
    mapping = np.zeros((len(CONDITIONS), K + 1), dtype=np.uint8)
    for ci in range(len(CONDITIONS)):
        if ci == REFERENCE:
            mapping[ci, 1:] = np.arange(1, K + 1)
        else:
            ref, target = linear_sum_assignment(similarity[ci], maximize=True)
            mapping[ci, target + 1] = ref + 1
    return mapping, similarity


def _pooled_iou(summary: SpectralSummaries) -> np.ndarray:
    """Report spatial agreement independently of the spectral assignment."""
    counts = summary.contingency
    _require(counts.shape == (len(CONDITIONS), K, K) and counts.dtype.kind in "iu"
             and np.all(counts >= 0), "Invalid matching contingency")
    reference_counts, target_counts = counts.sum(axis=2), counts.sum(axis=1)
    expected = summary.pixels.sum(axis=0)
    _require(np.array_equal(target_counts, expected)
             and np.all(reference_counts == expected[REFERENCE]), "Matching pixel counts differ")
    empty = expected == 0
    _require(not empty.any(), "Empty cluster for IoU diagnostic: " + ", ".join(
        f"{CONDITIONS[ci]} cluster {k + 1}" for ci, k in np.argwhere(empty)))
    union = reference_counts[:, :, None] + target_counts[:, None, :] - counts
    return counts / union


def _write_tables(
    output: Path, summary: SpectralSummaries, mapping: np.ndarray, similarity: np.ndarray,
) -> np.ndarray:
    sample_rows, count_rows, aggregate_rows, match_rows, matrix_rows = [], [], [], [], []
    occupancy_rows, difference_rows = [], []
    means = np.full((len(CONDITIONS), K, len(KINDS), 256), np.nan)
    intervals = np.full((len(CONDITIONS), K, len(KINDS), 2, 256), np.nan)
    band_names = [f"band_{i:03d}" for i in range(256)]
    iou = _pooled_iou(summary)
    for ci, condition in enumerate(CONDITIONS):
        common_pixels = int(summary.contingency[ci].sum())
        matched_pixels = sum(int(summary.contingency[ci, mapping[ci, original] - 1, original - 1])
                             for original in range(1, K + 1))
        for original in range(1, K + 1):
            display = int(mapping[ci, original])
            assigned = float(similarity[ci, display - 1, original - 1])
            row_candidates = similarity[ci, display - 1].copy()
            row_candidates[original - 1] = -np.inf
            alternative = float(row_candidates.max())
            overlap = int(summary.contingency[ci, display - 1, original - 1])
            match_rows.append({"condition": condition, "original_cluster": original,
                               "display_cluster": display, "snv_cosine_similarity": assigned,
                               "best_other_target_snv_cosine_similarity": alternative,
                               "assigned_minus_best_other": assigned - alternative,
                               "iou": float(iou[ci, display - 1, original - 1]),
                               "overlap_pixels": overlap, "common_pixels": common_pixels,
                               "condition_overlap_pixels": matched_pixels,
                               "condition_overlap_fraction": matched_pixels / common_pixels})
            for kind, kind_name in enumerate(KINDS):
                mean, q25, q75, n = equal_sample_summary(summary.values[:, ci, original - 1, kind])
                means[ci, display - 1, kind] = mean
                intervals[ci, display - 1, kind] = np.stack((q25, q75))
                counts = (summary.pixels if kind < 2 else summary.pseudoabsorbance_pixels)
                contributors = [s for si, s in enumerate(summary.sample_ids)
                                if counts[si, ci, original - 1] > 0]
                for band, wave in enumerate(summary.wavelength):
                    aggregate_rows.append({"condition": condition, "cluster": display,
                        "kind": kind_name, "wavelength_nm": wave, "mean": mean[band],
                        "q25": q25[band], "q75": q75[band], "sample_count": n,
                        "pixel_count": int(counts[:, ci, original - 1].sum()),
                        "sample_ids": "|".join(contributors)})
                for si, sample in enumerate(summary.sample_ids):
                    count = int(counts[si, ci, original - 1])
                    sample_rows.append({"sample_id": sample, "condition": condition,
                        "cluster": display, "kind": kind_name, "pixel_count": count,
                        **dict(zip(band_names, summary.values[si, ci, original - 1, kind], strict=True))})
            for ref in range(K):
                matrix_rows.append({"condition": condition, "reference_cluster": ref + 1,
                    "original_target_cluster": original, "display_target_cluster": display,
                    "snv_cosine_similarity": similarity[ci, ref, original - 1],
                    "iou": iou[ci, ref, original - 1],
                    "pixel_count": int(summary.contingency[ci, ref, original - 1]),
                    "assigned": display == ref + 1})
        for si, sample in enumerate(summary.sample_ids):
            raw = summary.pixels[si, ci]
            pixels = int(raw.sum())
            aligned_counts = raw[np.argsort(mapping[ci, 1:])]
            diagnostics = occupancy_values(aligned_counts.tolist(), pixels, K)
            for original in range(1, K + 1):
                n = int(raw[original - 1])
                valid = int(summary.pseudoabsorbance_pixels[si, ci, original - 1])
                common = {"sample_id": sample, "condition": condition,
                          "cluster": int(mapping[ci, original]), "pixels": n}
                count_rows.append({**common, "pseudoabsorbance_pixels": valid,
                                   "pseudoabsorbance_excluded_pixels": n - valid,
                                   "pseudoabsorbance_defined": valid > 0})
                occupancy_rows.append({**common, "fraction": n / pixels,
                                       "sample_pixels": pixels, **diagnostics})
    for ci, condition in enumerate(CONDITIONS):
        if ci == REFERENCE:
            continue
        for cluster in range(K):
            for kind in PLOTTED_KINDS:
                difference = means[ci, cluster, kind] - means[REFERENCE, cluster, kind]
                for band, wave in enumerate(summary.wavelength):
                    difference_rows.append({"condition": condition,
                        "contrast": f"{condition}-M00", "cluster": cluster + 1,
                        "kind": KINDS[kind], "wavelength_nm": wave, "difference": difference[band],
                        "aggregation": "difference of equal-sample representative means; not paired"})
    tables = {
        "wavelengths": pd.DataFrame({"band": band_names, "wavelength_nm": summary.wavelength}),
        "sample_spectra": pd.DataFrame(sample_rows), "spectrum_counts": pd.DataFrame(count_rows),
        "representative_spectra": pd.DataFrame(aggregate_rows),
        "difference_spectra": pd.DataFrame(difference_rows), "matching": pd.DataFrame(match_rows),
        "matching_matrices": pd.DataFrame(matrix_rows), "occupancy": pd.DataFrame(occupancy_rows),
    }
    for name, table in tables.items():
        if name == "wavelengths":
            table.to_csv(output / f"{name}.csv", index=False)
        else:
            for condition in CONDITIONS:
                table.loc[table.condition == condition].to_csv(
                    output / condition / f"{name}.csv", index=False)
    np.savez_compressed(output / "spectral_summary.npz", wavelength_nm=summary.wavelength,
                        means=means, intervals=intervals, original_to_display=mapping,
                        conditions=np.array(CONDITIONS), kinds=np.array(KINDS))
    return means


def _save(figure: plt.Figure, path: Path, dpi: int) -> None:
    try:
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def _spectral_axis(axis: Axes, wave: np.ndarray, kind: int) -> None:
    axis.set_xlim(float(wave[0]), float(wave[-1]))
    ticks = [float(wave[0]), *[float(v) for v in range(1000, 2400, 100)
                             if wave[0] < v < wave[-1]], float(wave[-1])]
    # Keep the 2300 nm grid line, but avoid overlapping its label with 2305.59 nm.
    labels = [f"{v:.2f}" if i in (0, len(ticks) - 1)
              else (f"{v:.0f}" if min(v - wave[0], wave[-1] - v) >= 30 else "")
              for i, v in enumerate(ticks)]
    axis.set_xticks(ticks, labels, rotation=45)
    axis.set_xticks([v for v in np.arange(math.ceil(wave[0] / 50) * 50, wave[-1], 50)
                    if v % 100 != 0], minor=True)
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(which="both", direction="out", top=False, right=False, pad=4)
    axis.tick_params(which="major", length=5)
    axis.tick_params(which="minor", length=3)
    axis.set_axisbelow(True)
    axis.grid(which="major", color="0.82", linewidth=.65)
    axis.grid(which="minor", color="0.91", linewidth=.45, linestyle=":")
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel(Y_LABELS[kind])
    if kind in (0, 1):
        axis.set_ylim((0, 1) if kind == 0 else (-2, 2))
        axis.set_yticks(np.arange(0, 1.01, .2) if kind == 0 else np.arange(-2, 2.01, .5))


def _plot_spectra(output: Path, summary: SpectralSummaries, means: np.ndarray, dpi: int) -> None:
    with np.load(output / "spectral_summary.npz", allow_pickle=False) as saved:
        intervals = saved["intervals"]
    derivative = np.concatenate((means[:, :, 3].ravel(), intervals[:, :, 3].ravel()))
    finite = derivative[np.isfinite(derivative)]
    limits = None
    if len(finite):
        low, high = float(finite.min()), float(finite.max())
        margin = .05 * max(high - low, abs(low), abs(high), 1e-12)
        limits = (low - margin, high + margin)
    for ci, condition in enumerate(CONDITIONS):
        fig = plt.figure(figsize=(18, 10))
        grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.15))
        axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[1, :]))
        for axis, kind in zip(axes, (1, 0, 3), strict=True):
            for cluster in range(K):
                color = LABEL_COLORS[cluster + 1]
                axis.plot(summary.wavelength, means[ci, cluster, kind], color=color, lw=1.25)
                axis.fill_between(summary.wavelength, *intervals[ci, cluster, kind],
                                  color=color, alpha=.07, linewidth=0)
            _spectral_axis(axis, summary.wavelength, kind)
            if kind == 3 and limits is not None:
                axis.set_ylim(limits)
        fig.legend([Line2D([], [], color=LABEL_COLORS[k], lw=2) for k in range(1, K + 1)],
                   [str(k) for k in range(1, K + 1)], title="Cluster ID",
                   loc="lower center", ncol=K, frameon=False)
        fig.tight_layout(rect=(0, .065, 1, 1), h_pad=2, w_pad=2)
        _save(fig, output / condition / "01_representative_spectra.png", dpi)


def _paste_cluster_colorbar(canvas: Image.Image, *, dpi: int, height: int) -> None:
    fig = plt.figure(figsize=(canvas.width / dpi, height / dpi), dpi=dpi)
    try:
        mappable = matplotlib.cm.ScalarMappable(
            norm=BoundaryNorm(np.arange(.5, K + 1.5), K),
            cmap=ListedColormap(LABEL_COLORS[1:]))
        bar = fig.colorbar(mappable, cax=fig.add_axes((.18, .60, .64, .16)),
                           orientation="horizontal", ticks=np.arange(1, K + 1))
        bar.ax.tick_params(labelsize=80 * 72 / 240, length=5, pad=5)
        bar.set_label("Cluster ID", fontsize=80 * 72 / 240, labelpad=8)
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
            buffer.seek(0)
            with Image.open(buffer) as image:
                canvas.paste(image.convert("RGB"), (0, canvas.height - height))
    finally:
        plt.close(fig)


def _map_overview(
    labels: dict[str, np.ndarray], sample_ids: list[str], path: Path, *, rows: int,
    image_height: int, dpi: int,
) -> None:
    """Mirror sample_overviews spacing and labels, using nearest-neighbor class colors."""
    scale = dpi / 240
    width, gutter, label_height, font_size, bar_height = (
        max(1, round(value * scale)) for value in (720, 24, 128, 80, 360))
    _require(0 < len(sample_ids) <= rows * 7, "Map grid cannot hold the selected samples")
    cell_height = image_height + label_height
    size = (7 * width + 8 * gutter, rows * cell_height + (rows + 1) * gutter + bar_height)
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(str(font_path), size=font_size)
    palette = np.rint(np.array([to_rgb(color) for color in LABEL_COLORS]) * 255).astype(np.uint8)
    with Image.new("RGB", size, "white") as canvas:
        draw = ImageDraw.Draw(canvas)
        for index, sample_id in enumerate(sample_ids):
            row, column = divmod(index, 7)
            x, y = gutter + column * (width + gutter), gutter + row * (cell_height + gutter)
            with Image.fromarray(palette[labels[sample_id]]) as source:
                ratio = min(width / source.width, image_height / source.height)
                with source.resize((max(1, round(source.width * ratio)),
                                    max(1, round(source.height * ratio))),
                                   Image.Resampling.NEAREST) as display:
                    canvas.paste(display, (x + (width - display.width) // 2,
                                           y + (image_height - display.height) // 2))
            draw.text((x + width // 2, y + image_height + label_height // 2), sample_id,
                      fill="black", font=font, anchor="mm")
        _paste_cluster_colorbar(canvas, dpi=dpi, height=bar_height)
        with path.open("xb") as handle:
            canvas.save(handle, format="PNG", dpi=(dpi, dpi))


def _plot_maps(
    experiment: Path, inventory: InputInventory, output: Path, mapping: np.ndarray, dpi: int,
) -> None:
    samples = sorted(inventory.samples, key=lambda s: s.sample_id)
    _require(0 < len(samples) <= 49, "Expected at most 49 global samples")
    sample_ids = [sample.sample_id for sample in samples]
    representative_ids = [sample for _, sample in REPRESENTATIVES]
    _require(set(representative_ids) <= set(sample_ids), "Missing fixed representative samples")
    width = max(1, round(720 * dpi / 240))
    image_height = max(round(width * s.height / s.width) for s in samples)
    for ci, condition in enumerate(CONDITIONS):
        directory = output / condition / "labels"
        directory.mkdir()
        labels = {s.sample_id: mapping[ci, read_label_map(condition_paths(experiment, condition)[0]
                  / "maps" / f"{s.sample_id}.npz", s)] for s in samples}
        np.savez_compressed(output / condition / "label_maps.npz", **labels)
        for group, start in enumerate(range(0, len(samples), 7)):
            stop = min(start + 7, len(samples))
            _map_overview(labels, sample_ids[start:stop],
                          directory / f"{group:02d}_samples_{start + 1:02d}-{stop:02d}_1x7.png",
                          rows=1, image_height=image_height, dpi=dpi)
        _map_overview(labels, representative_ids, directory / REPRESENTATIVE_LABEL_FIGURE,
                      rows=1, image_height=image_height, dpi=dpi)
        _map_overview(labels, sample_ids, directory / "08_all_samples_7x7.png",
                      rows=7, image_height=image_height, dpi=dpi)
        print(f"{condition}: saved label overviews", flush=True)


def _plot_representatives(output: Path, dpi: int) -> None:
    """Compare the fixed representative columns; print sample IDs below the final row only."""
    sample_ids = [sample for _, sample in REPRESENTATIVES]
    maps = {}
    for condition in CONDITIONS:
        with np.load(output / condition / "label_maps.npz", allow_pickle=False) as saved:
            _require(set(sample_ids) <= set(saved.files),
                     f"{condition}: missing fixed representative samples")
            maps[condition] = {sample: saved[sample] for sample in sample_ids}
    width, gutter, label_height, font_size, bar_height, left = (
        max(1, round(value * dpi / 240)) for value in (720, 24, 128, 80, 360, 240))
    image_height = max(round(width * labels.shape[0] / labels.shape[1])
                       for labels in maps[CONDITIONS[0]].values())
    size = (left + 7 * width + 8 * gutter,
            len(CONDITIONS) * image_height + (len(CONDITIONS) + 1) * gutter
            + label_height + bar_height)
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(str(font_path), size=font_size)
    palette = np.rint(np.array([to_rgb(color) for color in LABEL_COLORS]) * 255).astype(np.uint8)
    with Image.new("RGB", size, "white") as canvas:
        draw = ImageDraw.Draw(canvas)
        for row, condition in enumerate(CONDITIONS):
            y = gutter + row * (image_height + gutter)
            draw.text((left // 2, y + image_height // 2), condition,
                      fill="black", font=font, anchor="mm")
            for column, sample in enumerate(sample_ids):
                x = left + gutter + column * (width + gutter)
                with Image.fromarray(palette[maps[condition][sample]]) as source:
                    ratio = min(width / source.width, image_height / source.height)
                    with source.resize((max(1, round(source.width * ratio)),
                                        max(1, round(source.height * ratio))),
                                       Image.Resampling.NEAREST) as display:
                        canvas.paste(display, (x + (width - display.width) // 2,
                                               y + (image_height - display.height) // 2))
                if row == len(CONDITIONS) - 1:
                    draw.text((x + width // 2, y + image_height + label_height // 2), sample,
                              fill="black", font=font, anchor="mm")
        _paste_cluster_colorbar(canvas, dpi=dpi, height=bar_height)
        with (output / REPRESENTATIVE_FIGURE).open("xb") as handle:
            canvas.save(handle, format="PNG", dpi=(dpi, dpi))


def _plot_matrices(
    output: Path, mapping: np.ndarray, similarity: np.ndarray, dpi: int,
) -> None:
    for ci, condition in enumerate(CONDITIONS):
        fig, axis = plt.subplots(figsize=(7.5, 6.5))
        matrix = similarity[ci][:, np.argsort(mapping[ci, 1:])]
        artist = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        for row in range(K):
            for column in range(K):
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center",
                          color="white" if abs(matrix[row, column]) > .55 else "black", fontsize=10)
        axis.set_xticks(np.arange(K), np.arange(1, K + 1))
        axis.set_yticks(np.arange(K), np.arange(1, K + 1))
        axis.set_xlabel(f"{condition} Cluster ID")
        axis.set_ylabel("M00 Cluster ID")
        fig.colorbar(artist, ax=axis, fraction=.045, pad=.04).set_label("SNV cosine similarity")
        fig.tight_layout()
        _save(fig, output / condition / "02_snv_cosine_matrix.png", dpi)


def report_contract(experiment: Path, data: GlobalData, inventory: InputInventory) -> dict[str, object]:
    sources = {}
    for condition in CONDITIONS:
        check_global_clustering(experiment, data, inventory, condition)
        sources[condition] = _digest(condition_paths(experiment, condition)[0] / "completion.json")
    return {"schema_version": 3, "scope": "global_descriptive_report", "K": K,
            "conditions": list(CONDITIONS), "reference": "M00+Cosine-KMeans", "sources": sources,
            "manifest_sha256": _digest(experiment / "manifests/complete.json"),
            "code_sha256": {name: _digest(Path(__file__).with_name(name))
                            for name in ("global_reporting.py", "oof_sanity.py")},
            "runtime": {"numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}


def render_global_report(
    experiment: Path, data: GlobalData, inventory: InputInventory, *, dpi: int = 240,
    chunk_pixels: int = 2048,
) -> dict[str, object]:
    _require(type(dpi) is int and dpi > 0, "Invalid DPI")
    output = experiment / DEFAULT_DIRECTORY
    contract = report_contract(experiment, data, inventory)
    with TemporaryDirectory(prefix=".global-report-", dir=experiment) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        for condition in CONDITIONS:
            (stage / condition).mkdir()
        summary = collect_spectra(experiment, inventory, chunk_pixels=chunk_pixels)
        mapping, similarity = match_snv(summary)
        means = _write_tables(stage, summary, mapping, similarity)
        _plot_maps(experiment, inventory, stage, mapping, dpi)
        _plot_representatives(stage, dpi)
        _plot_spectra(stage, summary, means, dpi)
        _plot_matrices(stage, mapping, similarity, dpi)
        captions = [(REPRESENTATIVE_FIGURE,
                     "Rows from top: B0, B1, A0, M00, M11. Columns: "
                     + ", ".join(sample for _, sample in REPRESENTATIVES)
                     + ". Fixed representative samples; sample IDs appear below the final row "
                     "only. Shared M00-aligned display IDs and cluster colors.")]
        for condition in CONDITIONS:
            captions.extend([
                (f"{condition}/labels/*_samples_??-??_1x7.png",
                 "Samples in ascending ID order, seven per "
                 "file; the seven rows together contain all 49 samples. Background is 0. "
                 "Display IDs aligned once to M00 by maximum summed cosine similarity "
                 "of equal-sample observed SNV representative spectra."),
                (f"{condition}/labels/{REPRESENTATIVE_LABEL_FIGURE}",
                 "Fixed representative samples, left to right: "
                 + ", ".join(sample for _, sample in REPRESENTATIVES)
                 + ". Same columns, display IDs and colors as the five-condition comparison."),
                (f"{condition}/labels/08_all_samples_7x7.png", "All 49 samples in ascending "
                 "ID order, row-major 7 by 7 grid. The same display IDs and colors apply."),
                (f"{condition}/01_representative_spectra.png", "Top left: SNV; top right: "
                 "reflectance; full-width bottom: pseudoabsorbance SG second derivative. "
                 "Colors: display clusters 1-8. Lines: equal-sample means; bands: sample "
                 "IQR, not confidence intervals. Absent curves excluded. A single contributing "
                 "sample cannot establish between-sample variation. SG(7,2,2), delta in nm. "
                 "Major wavelength grid every 100 nm, minor every 50 nm; endpoint labels "
                 "take precedence over neighboring labels within 30 nm."),
                (f"{condition}/02_snv_cosine_matrix.png", "Rows: M00 clusters; columns: target "
                 "display clusters. Cosine similarity of equal-sample observed SNV means, "
                 "all 256 bands. Hungarian assignment maximizes summed cosine similarity; "
                 "not an independent chemical equivalence test."),
            ])
        pd.DataFrame(captions, columns=("file", "caption")).to_csv(stage / "captions.csv", index=False)
        _write_json(stage / "report.json", {
            "contract": contract, "dpi": dpi, "chunk_pixels": chunk_pixels,
            "sample_ids": list(summary.sample_ids),
            "representative_sample_ids": [sample for _, sample in REPRESENTATIVES],
            "representative_row_conditions": list(CONDITIONS),
            "map_groups": [list(summary.sample_ids[start:start + 7])
                           for start in range(0, len(summary.sample_ids), 7)],
            "definitions": {"aggregation": "within-sample pixel mean, then equal-sample mean",
                "pseudoabsorbance": "pixelwise -log10(reflectance); all 256 values positive finite",
                "missing": "absent curves are NaN in NPZ and empty CSV cells; no zero fill",
                "sg": {"window_length": 7, "polyorder": 2, "deriv": 2,
                       "delta_nm": float(summary.wavelength[1] - summary.wavelength[0]),
                       "mode": "interp", "axis": -1},
                "matching": "Hungarian maximum summed SNV cosine similarity, directly to M00",
                "matching_spectrum": "observed pixel SNV; within-sample cluster mean, then "
                    "equal-sample mean over present clusters; L2 normalization; all 256 bands",
                "iou_aggregation": "diagnostic only: all common valid pixels pooled; "
                    "not equal-sample weighting",
                "empty_clusters": "stop with original condition/cluster IDs; no fabricated match",
                "undefined_snv": "stop for nonfinite or zero-norm representative spectra",
                "palette": list(LABEL_COLORS), "grid_major_nm": 100, "grid_minor_nm": 50,
                "spatial_smoothing": "none; continuous spectral index maps are a later stage"},
        })
        _require(report_contract(experiment, data, inventory) == contract,
                 "Global report source changed during rendering")
        hashes = {str(p.relative_to(stage)).replace("\\", "/"): _digest(p)
                  for p in sorted(stage.rglob("*")) if p.is_file()}
        completion = {"status": "global_report_completed", "checks_passed": True,
                      "artifact_sha256": hashes, "samples": len(summary.sample_ids),
                      "png_count": sum(name.endswith(".png") for name in hashes),
                      "csv_count": sum(name.endswith(".csv") for name in hashes)}
        _write_json(stage / "completion.json", completion)
        output.parent.mkdir(parents=True, exist_ok=True)
        _publish_report(stage, output, experiment)
    return completion


def _publish_report(stage: Path, output: Path, experiment: Path) -> None:
    """Replace only the specified report after rendering; preserve the old one on failure."""
    expected = experiment.resolve() / DEFAULT_DIRECTORY
    _require(output.absolute() == expected and output.resolve() == expected,
             "Report replacement target escapes the expected directory")
    previous = stage.parent / "previous-report"
    saved_latent = previous / "pca-latent-2d"
    staged_latent = stage / "pca-latent-2d"
    if output.exists():
        for path in (output, *output.rglob("*")):
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
            _require(not path.is_symlink()
                     and not attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
                     and path.resolve().is_relative_to(expected),
                     f"Refuse replacement through a linked artifact: {path}")
        # The enclosing TemporaryDirectory removes the old, validated tree only after success.
        output.rename(previous)
    try:
        # Latent projections have their own completion record and independent lifecycle.
        if saved_latent.exists():
            saved_latent.rename(staged_latent)
        stage.rename(output)
    except OSError:
        if staged_latent.exists():
            staged_latent.rename(saved_latent)
        if previous.exists():
            previous.rename(output)
        raise


def check_global_report(
    experiment: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    output = experiment / DEFAULT_DIRECTORY
    report, completion = _read_json(output / "report.json"), _read_json(output / "completion.json")
    _require(report.get("contract") == report_contract(experiment, data, inventory)
             and completion.get("status") == "global_report_completed"
             and completion.get("checks_passed") is True, "Global report contract differs")
    hashes = completion["artifact_sha256"]
    actual = {str(p.relative_to(output)).replace("\\", "/") for p in output.rglob("*")
              if p.is_file() and p.name != "completion.json"
              and p.relative_to(output).parts[0] != "pca-latent-2d"}
    _require(set(hashes) == actual, "Global report artifact coverage differs")
    expected_samples = sorted(sample.sample_id for sample in inventory.samples)
    _require(report["sample_ids"] == expected_samples
             and completion["samples"] == len(expected_samples)
             and completion["png_count"] == sum(name.endswith(".png") for name in hashes)
             == len(CONDITIONS) * (math.ceil(len(expected_samples) / 7) + 4) + 1
             and completion["csv_count"] == sum(name.endswith(".csv") for name in hashes) == 37,
             "Global report artifact counts differ")
    _require(report["representative_sample_ids"] == [sample for _, sample in REPRESENTATIVES]
             and report["representative_row_conditions"] == list(CONDITIONS),
             "Global representative layout differs")
    expected_png = {REPRESENTATIVE_FIGURE}
    for condition in CONDITIONS:
        expected_png.update((f"{condition}/01_representative_spectra.png",
                             f"{condition}/02_snv_cosine_matrix.png",
                             f"{condition}/labels/{REPRESENTATIVE_LABEL_FIGURE}",
                             f"{condition}/labels/08_all_samples_7x7.png"))
        for group, start in enumerate(range(0, len(expected_samples), 7)):
            stop = min(start + 7, len(expected_samples))
            expected_png.add(f"{condition}/labels/{group:02d}_samples_{start + 1:02d}-{stop:02d}_1x7.png")
    _require({name for name in hashes if name.endswith(".png")} == expected_png,
             "Global report PNG layout differs")
    for name, expected in hashes.items():
        path = (output / name).resolve()
        _require(path.is_relative_to(output.resolve()) and _digest(path) == expected,
                 f"Global report artifact changed: {name}")
    return {"status": "validated_global_report", "checks_passed": True,
            "samples": completion["samples"], "png_count": completion["png_count"],
            "csv_count": completion["csv_count"], "output": str(output)}
