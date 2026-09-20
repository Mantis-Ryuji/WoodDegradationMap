"""Observed spectral summaries, M00 matching, and static global-fit PNG/CSV reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
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
COLORS = ("#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00")
Y_LABELS = ("Reflectance", "SNV", "Pseudoabsorbance", r"$d^2A/d\lambda^2$ (nm$^{-2}$)")
DEFAULT_DIRECTORY = "results/figures/global_k8_v1"


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


def match_spectra(summary: SpectralSummaries) -> tuple[np.ndarray, np.ndarray]:
    """SNV cosine Hungarian matching; overlap is diagnostic and never the objective."""
    means = np.stack([[equal_sample_summary(summary.values[:, ci, k, 1])[0]
                       for k in range(K)] for ci in range(len(CONDITIONS))])
    norms = np.linalg.norm(means, axis=-1)
    invalid = ~np.isfinite(means).all(axis=-1) | ~np.isfinite(norms) | (norms == 0)
    _require(not invalid.any(), "Undefined SNV representative for matching: " + ", ".join(
        f"{CONDITIONS[ci]} cluster {k + 1}" for ci, k in np.argwhere(invalid)))
    normalized = means / norms[..., None]
    similarity = np.einsum("id,cjd->cij", normalized[REFERENCE], normalized)
    mapping = np.zeros((len(CONDITIONS), K + 1), dtype=np.uint8)
    for ci in range(len(CONDITIONS)):
        if ci == REFERENCE:
            mapping[ci, 1:] = np.arange(1, K + 1)
        else:
            ref, target = linear_sum_assignment(similarity[ci], maximize=True)
            mapping[ci, target + 1] = ref + 1
    return mapping, similarity


def _write_tables(
    output: Path, summary: SpectralSummaries, mapping: np.ndarray, similarity: np.ndarray,
) -> np.ndarray:
    sample_rows, count_rows, aggregate_rows, match_rows, matrix_rows = [], [], [], [], []
    occupancy_rows, difference_rows = [], []
    means = np.full((len(CONDITIONS), K, len(KINDS), 256), np.nan)
    intervals = np.full((len(CONDITIONS), K, len(KINDS), 2, 256), np.nan)
    band_names = [f"band_{i:03d}" for i in range(256)]
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
                               "display_cluster": display, "snv_cosine": assigned,
                               "best_other_target_cosine": alternative,
                               "assigned_minus_best_other": assigned - alternative,
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
                    "snv_cosine": similarity[ci, ref, original - 1],
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
                    difference_rows.append({"contrast": f"{condition}-M00", "cluster": cluster + 1,
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
        table.to_csv(output / f"{name}.csv", index=False)
    np.savez_compressed(output / "spectral_summary.npz", wavelength_nm=summary.wavelength,
                        means=means, intervals=intervals, original_to_display=mapping,
                        conditions=np.array(CONDITIONS), kinds=np.array(KINDS))
    return means


def _save(figure: plt.Figure, path: Path, dpi: int) -> None:
    try:
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def _spectral_axis(axis: Axes, wave: np.ndarray, kind: int, *, difference: bool) -> None:
    axis.set_xlim(float(wave[0]), float(wave[-1]))
    ticks = [float(wave[0]), *[float(v) for v in range(1000, 2400, 200)
                             if wave[0] < v < wave[-1]], float(wave[-1])]
    axis.set_xticks(ticks, [f"{v:.2f}" if i in (0, len(ticks) - 1) else f"{v:.0f}"
                           for i, v in enumerate(ticks)], rotation=45)
    axis.tick_params(which="both", direction="out", top=False, right=False, pad=4)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel(("Difference: " if difference else "") + Y_LABELS[kind])
    if not difference and kind in (0, 1):
        axis.set_ylim((0, 1) if kind == 0 else (-2, 2))
        axis.set_yticks(np.arange(0, 1.01, .2) if kind == 0 else np.arange(-2, 2.01, .5))


def _plot_spectra(output: Path, summary: SpectralSummaries, means: np.ndarray, dpi: int) -> None:
    with np.load(output / "spectral_summary.npz", allow_pickle=False) as saved:
        intervals = saved["intervals"]
    for difference in (False, True):
        for index, kind in enumerate(PLOTTED_KINDS, start=7 if difference else 2):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
            for cluster, axis in enumerate(axes.flat):
                for ci, condition in enumerate(CONDITIONS):
                    if difference and ci == REFERENCE:
                        continue
                    values = means[ci, cluster, kind]
                    if difference:
                        values = values - means[REFERENCE, cluster, kind]
                    axis.plot(summary.wavelength, values, color=COLORS[ci], lw=1.0)
                    if not difference:
                        axis.fill_between(summary.wavelength, *intervals[ci, cluster, kind],
                                          color=COLORS[ci], alpha=.09, linewidth=0)
                if difference:
                    axis.axhline(0, color="0.5", lw=.6)
                axis.text(.03, .96, str(cluster + 1), transform=axis.transAxes, va="top",
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
                _spectral_axis(axis, summary.wavelength, kind, difference=difference)
            selected = [i for i in range(len(CONDITIONS)) if not difference or i != REFERENCE]
            fig.legend([Line2D([], [], color=COLORS[i]) for i in selected],
                       [f"{CONDITIONS[i]} - M00" if difference else CONDITIONS[i] for i in selected],
                       loc="lower center", ncol=len(selected), frameon=False)
            fig.tight_layout(rect=(0, .045, 1, 1))
            suffix = "differences" if difference else "spectra"
            _save(fig, output / f"{index:02d}_{KINDS[kind]}_{suffix}.png", dpi)


def _map_axis(axis: Axes, labels: np.ndarray, sample: str, panel: str) -> None:
    axis.imshow(labels, cmap=ListedColormap(LABEL_COLORS),
                norm=BoundaryNorm(np.arange(-.5, K + 1.5), K + 1), interpolation="nearest")
    axis.set_axis_off()
    axis.text(.5, -.025, sample, transform=axis.transAxes, ha="center", va="top", fontsize=8)
    axis.text(.015, .98, panel, transform=axis.transAxes, ha="left", va="top", fontsize=8,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})


def _plot_maps(
    experiment: Path, inventory: InputInventory, output: Path, mapping: np.ndarray, dpi: int,
) -> None:
    (output / "maps").mkdir()
    (output / "labels").mkdir()
    representatives = {sample for _, sample in REPRESENTATIVES}
    examples = {}
    for sample in sorted(inventory.samples, key=lambda s: s.sample_id):
        labels = {condition: mapping[ci, read_label_map(
            condition_paths(experiment, condition)[0] / "maps" / f"{sample.sample_id}.npz", sample)]
            for ci, condition in enumerate(CONDITIONS)}
        np.savez_compressed(output / "labels" / f"{sample.sample_id}.npz", **labels)
        fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(15, 5))
        for ci, axis in enumerate(axes):
            _map_axis(axis, labels[CONDITIONS[ci]], sample.sample_id, chr(97 + ci))
        fig.subplots_adjust(left=.02, right=.98, top=.97, bottom=.22, wspace=.04)
        bar = fig.colorbar(axes[0].images[0], cax=fig.add_axes((.3, .10, .4, .025)),
                           orientation="horizontal", ticks=np.arange(K + 1))
        bar.set_label("M00 reference cluster ID (0 = background)")
        _save(fig, output / "maps" / f"{sample.sample_id}_k8.png", dpi)
        if sample.sample_id in representatives:
            examples[sample.sample_id] = labels
    ordered = [sample for _, sample in REPRESENTATIVES if sample in examples]
    _require(bool(ordered), "No fixed representative samples present")
    height = 3.6 * len(ordered) + .85
    fig, axes = plt.subplots(len(ordered), len(CONDITIONS), figsize=(15, height),
                             squeeze=False)
    for si, sample in enumerate(ordered):
        for ci, axis in enumerate(axes[si]):
            _map_axis(axis, examples[sample][CONDITIONS[ci]], sample, chr(97 + ci))
    fig.subplots_adjust(left=.02, right=.98, hspace=.14, wspace=.04,
                        bottom=.85 / height, top=1 - .15 / height)
    # Keep a readable bar width and a fixed physical gap below the final sample IDs.
    bar = fig.colorbar(axes[0, 0].images[0], cax=fig.add_axes((.3, .35 / height, .4, .15 / height)),
                       orientation="horizontal", ticks=np.arange(K + 1))
    bar.set_label("M00 reference cluster ID (0 = background)")
    _save(fig, output / "01_representative_maps.png", dpi)


def _plot_matrices(
    output: Path, summary: SpectralSummaries, mapping: np.ndarray, similarity: np.ndarray, dpi: int,
) -> None:
    selected = [ci for ci in range(len(CONDITIONS)) if ci != REFERENCE]
    for name, number in (("snv_similarity", 5), ("contingency", 6)):
        fig, axes = plt.subplots(1, len(selected), figsize=(16, 4.2))
        for axis, ci in zip(axes, selected, strict=True):
            order = np.argsort(mapping[ci, 1:])
            matrix = similarity[ci][:, order] if number == 5 else summary.contingency[ci][:, order]
            if number == 5:
                artist = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
            else:
                totals = matrix.sum(axis=1, keepdims=True)
                artist = axis.imshow(np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float),
                                                where=totals > 0), vmin=0, vmax=1, cmap="Blues")
            axis.set_xticks(np.arange(K), np.arange(1, K + 1))
            axis.set_yticks(np.arange(K), np.arange(1, K + 1))
            axis.set_xlabel(f"{CONDITIONS[ci]} display cluster")
            axis.set_ylabel("M00 cluster")
        bar = fig.colorbar(artist, ax=list(axes), fraction=.018, pad=.02)
        bar.set_label("SNV cosine similarity" if number == 5 else "Fraction within M00 cluster")
        _save(fig, output / f"{number:02d}_{name}.png", dpi)


def report_contract(experiment: Path, data: GlobalData, inventory: InputInventory) -> dict[str, object]:
    sources = {}
    for condition in CONDITIONS:
        check_global_clustering(experiment, data, inventory, condition)
        sources[condition] = _digest(condition_paths(experiment, condition)[0] / "completion.json")
    return {"schema_version": 1, "scope": "global_descriptive_report", "K": K,
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
    _require(not output.exists(), "Global report exists; use check, or --resume to verify/skip")
    contract = report_contract(experiment, data, inventory)
    with TemporaryDirectory(prefix=".global-report-", dir=experiment) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        summary = collect_spectra(experiment, inventory, chunk_pixels=chunk_pixels)
        mapping, similarity = match_spectra(summary)
        means = _write_tables(stage, summary, mapping, similarity)
        _plot_maps(experiment, inventory, stage, mapping, dpi)
        _plot_spectra(stage, summary, means, dpi)
        _plot_matrices(stage, summary, mapping, similarity, dpi)
        captions = [
            ("01_representative_maps.png", "Rows: fixed representative samples in design order; "
             "columns/panels a-e: B0, B1, A0, M00, M11; M00 SNV matching; 0 is background."),
            ("maps/*.png", "Panels a-e: B0, B1, A0, M00, M11; identical pixel coordinates; "
             "colors aligned by SNV representative similarity, not chemical ground truth."),
            ("05_snv_similarity.png", "Equal-sample SNV representative cosine similarity; "
             "columns reordered by Hungarian matching to M00."),
            ("06_contingency.png", "Common-pixel contingency normalized within each M00 cluster; "
             "diagnostic only. Larger samples contribute more pixels. Raw counts in CSV."),
        ]
        for i, kind in enumerate(PLOTTED_KINDS, start=2):
            captions.append((f"{i:02d}_{KINDS[kind]}_spectra.png", "Panels 1-8: display clusters; "
                "lines: equal-sample means; bands: sample IQR (not confidence intervals). "
                "Absent sample/cluster curves excluded; contribution counts in CSV. "
                "Second derivative: pixelwise -log10(R), within-sample mean, SG(7,2,2)."))
            captions.append((f"{i + 5:02d}_{KINDS[kind]}_differences.png", "Panels 1-8: display "
                "clusters. Differences of representative means relative to M00; sample "
                "membership can differ, so these are not paired chemical changes."))
        pd.DataFrame(captions, columns=("file", "caption")).to_csv(stage / "captions.csv", index=False)
        _write_json(stage / "report.json", {
            "contract": contract, "dpi": dpi, "chunk_pixels": chunk_pixels,
            "sample_ids": list(summary.sample_ids),
            "representative_sample_ids": [sample for _, sample in REPRESENTATIVES
                                           if sample in summary.sample_ids],
            "definitions": {"aggregation": "within-sample pixel mean, then equal-sample mean",
                "pseudoabsorbance": "pixelwise -log10(reflectance); all 256 values positive finite",
                "missing": "absent curves are NaN in NPZ and empty CSV cells; no zero fill",
                "sg": {"window_length": 7, "polyorder": 2, "deriv": 2,
                       "delta_nm": float(summary.wavelength[1] - summary.wavelength[0]),
                       "mode": "interp", "axis": -1},
                "matching": "Hungarian maximum observed SNV cosine, directly to M00",
                "overlap": "diagnostic only; not used for assignment",
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
        stage.rename(output)
    return completion


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
              if p.is_file() and p.name != "completion.json"}
    _require(set(hashes) == actual, "Global report artifact coverage differs")
    expected_samples = sorted(sample.sample_id for sample in inventory.samples)
    _require(report["sample_ids"] == expected_samples
             and completion["samples"] == len(expected_samples)
             and completion["png_count"] == sum(name.endswith(".png") for name in hashes)
             == 9 + len(expected_samples)
             and completion["csv_count"] == sum(name.endswith(".csv") for name in hashes) == 9,
             "Global report artifact counts differ")
    for name, expected in hashes.items():
        path = (output / name).resolve()
        _require(path.is_relative_to(output.resolve()) and _digest(path) == expected,
                 f"Global report artifact changed: {name}")
    return {"status": "validated_global_report", "checks_passed": True,
            "samples": completion["samples"], "png_count": completion["png_count"],
            "csv_count": completion["csv_count"], "output": str(output)}
