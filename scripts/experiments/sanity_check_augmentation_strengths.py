"""Illustrate SNV augmentation and summarize its effect on fixed train pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig

from wood_degradation_map.experiments.config import experiment_config
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import load_manifest_bundle
from wood_degradation_map.preprocessing.visualization import (
    _mark_outside_limits,
    configure_spectrum_axis,
    configure_wavelength_axis,
)


NOISE_ANGLES = (2.5, 5.0, 7.5)
SHIFT_MAGNITUDES = (1.0, 2.0, 3.0)
SAMPLE_COUNT = 8
PIXELS_PER_SAMPLE = 128
EXAMPLE_COUNT = 3
EXAMPLE_FIGSIZE = (14.5, 8.4)
SHIFT_EXAMPLE_SIGNS = (1.0, -1.0, 1.0)
COLORS = ("#0072B2", "#E69F00", "#CC79A7")
OBSOLETE_PLOTS = (
    "snv_noise_uniform_ranges_distributions.png",
    "snv_shift_uniform_ranges_distributions.png",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Refresh existing example figures and plot metadata without recomputing train metrics.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=root / "outputs/experiments/production_v1",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=root / "data/processed/production_v1",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "data/metadata/古材メタデータ.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/sanity_checks/augmentation_strengths_train_fold1",
    )
    return parser.parse_args()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _spaced_indices(size: int, count: int) -> np.ndarray:
    if size < count:
        raise ValueError(f"Cannot choose {count} distinct positions from {size}")
    return np.linspace(0, size - 1, count, dtype=np.int64)


def _select_measured_examples(
    spectra: np.ndarray,
    selection: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    """Choose varied measured pixels near the center of their own sample."""
    if spectra.ndim != 2 or len(spectra) != len(selection) or not np.isfinite(spectra).all():
        raise ValueError("Finite spectra must align with the saved selection rows")
    representatives: list[int] = []
    median_distances: list[float] = []
    for positions in selection.groupby("sample_id", sort=True).indices.values():
        block = spectra[positions].astype(np.float64)
        center = np.median(block, axis=0)
        distances = np.sqrt(np.mean((block - center) ** 2, axis=1))
        local_index = int(np.argmin(distances))
        representatives.append(int(positions[local_index]))
        median_distances.append(float(distances[local_index]))
    if len(representatives) < EXAMPLE_COUNT:
        raise ValueError(f"At least {EXAMPLE_COUNT} measured samples are required")
    values = spectra[representatives].astype(np.float64)
    pairwise = np.sqrt(np.mean((values[:, None] - values[None, :]) ** 2, axis=2))

    def diversity(indices: tuple[int, ...]) -> tuple[float, float]:
        distances = [pairwise[first, second] for first, second in combinations(indices, 2)]
        return float(min(distances)), float(sum(distances))

    chosen = max(combinations(range(len(representatives)), EXAMPLE_COUNT), key=diversity)
    indices = [representatives[index] for index in chosen]
    examples = selection.iloc[indices][[
        "subset_index", "sample_id", "hdf5_row", "pixel_row", "pixel_col",
    ]].copy()
    examples["distance_to_sample_median_rms"] = [median_distances[index] for index in chosen]
    details: dict[str, object] = {
        "algorithm": (
            "Within each sample, choose the measured pixel nearest its bandwise median "
            "by RMS distance. Among these sample representatives, choose three maximizing "
            "the minimum pairwise RMS distance, then the sum of pairwise distances. "
            "Ties follow sorted sample IDs and saved selection row order."
        ),
        "scope": "Illustrative selection only; no data quality exclusion or metric changes",
        "candidate_pixel_count": len(selection),
        "candidate_sample_count": len(representatives),
        "minimum_pairwise_rms": diversity(chosen)[0],
    }
    return spectra[indices], examples.reset_index(drop=True), details


def _load_saved_subset(
    output: Path,
    processed: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Read only the previously selected train rows, preserving their order."""
    selection_path = output / "selection.csv"
    selection = pd.read_csv(selection_path)
    if not np.array_equal(selection["subset_index"], np.arange(len(selection))):
        raise ValueError("selection.csv must retain contiguous subset_index order")
    spectra = np.empty((len(selection), 256), dtype=np.float32)
    wavelength: np.ndarray | None = None
    for sample_id, group in selection.groupby("sample_id", sort=False):
        rows, inverse = np.unique(group["hdf5_row"].to_numpy(dtype=np.int64), return_inverse=True)
        sample_path = (processed / "samples" / f"{sample_id}.h5").resolve()
        if not sample_path.is_relative_to((processed / "samples").resolve()):
            raise ValueError(f"Invalid saved sample ID: {sample_id}")
        with h5py.File(sample_path, "r") as handle:
            if rows[0] < 0 or rows[-1] >= len(handle["snv"]):
                raise ValueError(f"{sample_id}: saved HDF5 rows are out of bounds")
            values = handle["snv"][rows][inverse]
            coordinates = handle["pixel_row_col"][rows][inverse]
            current_wavelength = handle["wavelength_nm"][:]
        if values.dtype != np.float32 or values.shape != (len(group), 256):
            raise ValueError(f"{sample_id}: unexpected saved SNV shape or dtype")
        if not np.isfinite(values).all():
            raise ValueError(f"{sample_id}: non-finite saved SNV")
        if not np.array_equal(coordinates, group[["pixel_row", "pixel_col"]].to_numpy()):
            raise ValueError(f"{sample_id}: saved pixel coordinates differ from selection.csv")
        if wavelength is None:
            wavelength = current_wavelength
        elif not np.array_equal(wavelength, current_wavelength):
            raise ValueError(f"{sample_id}: wavelength grid differs")
        spectra[group.index] = values
    if wavelength is None:
        raise ValueError("Saved train selection is empty")
    return spectra, wavelength, selection


def _load_train_subset(
    experiment_dir: Path,
    processed_dir: Path,
    metadata: Path,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    inventory = load_input_inventory(processed_dir, metadata)
    manifest = load_manifest_bundle(experiment_dir, inventory, processed_dir, metadata)
    train = manifest.train_pixels[fold]
    train_ids = sorted(train["sample_id"].unique())
    selected_ids = [train_ids[index] for index in _spaced_indices(len(train_ids), SAMPLE_COUNT)]
    samples = {sample.sample_id: sample for sample in inventory.samples}

    spectra: list[np.ndarray] = []
    selections: list[pd.DataFrame] = []
    wavelength: np.ndarray | None = None
    for sample_id in selected_ids:
        group = train.loc[train["sample_id"] == sample_id].reset_index(drop=True)
        selected = group.iloc[_spaced_indices(len(group), PIXELS_PER_SAMPLE)].copy()
        sample = samples[sample_id]
        rows = selected["hdf5_row"].to_numpy(dtype=np.int64)
        with h5py.File(sample.path, "r") as handle:
            values = handle["snv"][rows]
            current_wavelength = handle["wavelength_nm"][:]
        if values.dtype != np.float32 or values.shape != (PIXELS_PER_SAMPLE, 256):
            raise ValueError(f"{sample_id}: unexpected SNV subset")
        if not np.isfinite(values).all():
            raise ValueError(f"{sample_id}: non-finite SNV subset")
        if wavelength is None:
            wavelength = current_wavelength
        elif not np.array_equal(wavelength, current_wavelength):
            raise ValueError(f"{sample_id}: wavelength grid differs")
        spectra.append(values)
        selections.append(selected)

    if wavelength is None:
        raise RuntimeError("No train spectra were selected")
    selection = pd.concat(selections, ignore_index=True)
    selection.insert(0, "subset_index", np.arange(len(selection), dtype=np.int64))
    return np.concatenate(spectra, axis=0), wavelength, selection


def _augmenter(
    *,
    noise_range: tuple[float, float] | None = None,
    shift_range: tuple[float, float] | None = None,
) -> SpectraAugmenter:
    if (noise_range is None) == (shift_range is None):
        raise ValueError("Specify exactly one augmentation range")
    fixed = experiment_config()["augmentation"]
    if not isinstance(fixed, dict):
        raise TypeError("Experiment augmentation settings must be a dictionary")
    configured_noise = tuple(fixed["noise_angle_deg_range"])
    configured_shift = tuple(fixed["shift_delta_range"])
    config = SpectraAugmenterConfig(
        noise_prob=float(noise_range is not None),
        shift_prob=float(shift_range is not None),
        noise_angle_deg_range=noise_range if noise_range is not None else configured_noise,
        shift_delta_range=shift_range if shift_range is not None else configured_shift,
        shuffle_order_per_batch=bool(fixed["shuffle_order_per_batch"]),
        recenter_after_each_op=bool(fixed["recenter_after_each_op"]),
        renorm_to_input_norm=bool(fixed["renorm_to_input_norm"]),
        eps=float(fixed["eps"]),
    )
    return SpectraAugmenter(config).train()


def _apply(
    spectra: np.ndarray,
    seed: int,
    *,
    noise_range: tuple[float, float] | None = None,
    shift_range: tuple[float, float] | None = None,
) -> np.ndarray:
    torch.manual_seed(seed)
    with torch.no_grad():
        result = _augmenter(
            noise_range=noise_range,
            shift_range=shift_range,
        )(torch.from_numpy(spectra.copy()))
    return result.numpy()


def _metrics(
    clean: np.ndarray,
    perturbed: np.ndarray,
    selection: pd.DataFrame,
    *,
    kind: str,
    strength: float,
) -> pd.DataFrame:
    clean64 = clean.astype(np.float64)
    perturbed64 = perturbed.astype(np.float64)
    clean_norm = np.linalg.norm(clean64, axis=1)
    perturbed_norm = np.linalg.norm(perturbed64, axis=1)
    cosine = np.sum(clean64 * perturbed64, axis=1) / (clean_norm * perturbed_norm)
    difference = perturbed64 - clean64
    result = selection[[
        "subset_index", "sample_id", "hdf5_row", "pixel_row", "pixel_col"
    ]].copy()
    result.insert(1, "kind", kind)
    result.insert(2, "strength", float(strength))
    result["spectral_angle_deg"] = np.rad2deg(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )
    result["cosine_similarity"] = cosine
    result["rms_difference_snv"] = np.sqrt(np.mean(difference**2, axis=1))
    result["max_abs_difference_snv"] = np.max(np.abs(difference), axis=1)
    result["perturbed_mean"] = perturbed64.mean(axis=1)
    result["norm_absolute_error"] = np.abs(perturbed_norm - clean_norm)
    return result


def _plot_noise_examples(
    output: Path,
    wavelength: np.ndarray,
    clean: np.ndarray,
    variants: dict[float, np.ndarray],
    examples: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        EXAMPLE_COUNT, 2, figsize=EXAMPLE_FIGSIZE, dpi=180, sharex=True,
        constrained_layout=True,
    )
    for row, spectrum in enumerate(clean):
        left, right = axes[row]
        configure_spectrum_axis(left, representation="snv")
        right.set_ylim(-0.5, 0.5)
        left.plot(wavelength, spectrum, color="0.15", linewidth=1.35, label="clean")
        _mark_outside_limits(left, wavelength, spectrum, color="0.15")
        for color, (angle, values) in zip(COLORS, variants.items(), strict=True):
            label = f"{angle:g}°"
            left.plot(wavelength, values[row], color=color, linewidth=0.95, label=label)
            right.plot(
                wavelength,
                values[row] - spectrum,
                color=color,
                linewidth=0.95,
                label=label,
            )
            _mark_outside_limits(left, wavelength, values[row], color=color)
            _mark_outside_limits(right, wavelength, values[row] - spectrum, color=color)
        left.set_ylabel(f"Example {row + 1}\nSNV spectra")
        record = examples.iloc[row]
        left.text(
            0.02, 0.05,
            f"{record['sample_id']}  /  pixel ({int(record['pixel_row'])}, "
            f"{int(record['pixel_col'])})",
            transform=left.transAxes, fontsize=9,
        )
        left.grid(alpha=0.18)
        right.axhline(0.0, color="0.4", linewidth=0.7)
        right.set_ylim(-0.5, 0.5)
        right.set_yticks(np.linspace(-0.5, 0.5, 11))
        right.set_ylabel("Perturbed - clean (SNV spectra)")
        right.grid(alpha=0.18)
        for axis in (left, right):
            configure_wavelength_axis(axis, wavelength)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("Wavelength (nm)")
    axes[-1, 1].set_xlabel("Wavelength (nm)")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_shift_examples(
    output: Path,
    wavelength: np.ndarray,
    clean: np.ndarray,
    variants: dict[float, np.ndarray],
    examples: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        EXAMPLE_COUNT, 2, figsize=EXAMPLE_FIGSIZE, dpi=180, sharex=True,
        constrained_layout=True,
    )
    for row, (spectrum, sign) in enumerate(zip(clean, SHIFT_EXAMPLE_SIGNS, strict=True)):
        left, right = axes[row]
        configure_spectrum_axis(left, representation="snv")
        right.set_ylim(-0.5, 0.5)
        left.plot(
            wavelength,
            spectrum,
            color="0.15",
            linewidth=1.35,
            label="clean",
        )
        _mark_outside_limits(left, wavelength, spectrum, color="0.15")
        for color, magnitude in zip(COLORS, SHIFT_MAGNITUDES, strict=True):
            delta = sign * magnitude
            shifted = variants[delta][row]
            left.plot(
                wavelength,
                shifted,
                color=color,
                linewidth=0.95,
                label=f"{delta:+g} channels",
            )
            right.plot(
                wavelength,
                shifted - spectrum,
                color=color,
                linewidth=0.95,
            )
            _mark_outside_limits(left, wavelength, shifted, color=color)
            _mark_outside_limits(right, wavelength, shifted - spectrum, color=color)
        direction = "+" if sign > 0 else "−"
        left.set_ylabel(f"Example {row + 1} ({direction})\nSNV spectra")
        record = examples.iloc[row]
        left.text(
            0.02, 0.05,
            f"{record['sample_id']}  /  pixel ({int(record['pixel_row'])}, "
            f"{int(record['pixel_col'])})",
            transform=left.transAxes, fontsize=9,
        )
        left.grid(alpha=0.18)
        right.axhline(0.0, color="0.4", linewidth=0.7)
        right.set_ylim(-0.5, 0.5)
        right.set_yticks(np.linspace(-0.5, 0.5, 11))
        right.set_ylabel("Shifted - clean (SNV spectra)")
        right.grid(alpha=0.18)
        for axis in (left, right):
            configure_wavelength_axis(axis, wavelength)
        left.legend(frameon=False, ncol=2, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Wavelength (nm)")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _quantiles(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "q05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "q95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _summary(metrics: pd.DataFrame, kind: str) -> dict[str, object]:
    subset = metrics.loc[metrics["kind"] == kind]
    return {
        f"{strength:g}": {
            column: _quantiles(group[column])
            for column in (
                "spectral_angle_deg",
                "rms_difference_snv",
                "max_abs_difference_snv",
            )
        }
        for strength, group in subset.groupby("strength", sort=True)
    }


def _write_example_figures(
    output: Path,
    wavelength: np.ndarray,
    seed: int,
    clean: np.ndarray,
    examples: pd.DataFrame,
) -> dict[str, str]:
    noise_variants = {
        angle: _apply(clean, seed, noise_range=(angle, angle))
        for angle in NOISE_ANGLES
    }
    shift_variants = {
        sign * magnitude: _apply(
            clean, seed, shift_range=(sign * magnitude, sign * magnitude)
        )
        for sign in (1.0, -1.0)
        for magnitude in SHIFT_MAGNITUDES
    }
    _plot_noise_examples(
        output / "snv_noise_exact_angles_examples.png", wavelength, clean, noise_variants, examples
    )
    _plot_shift_examples(
        output / "snv_shift_exact_endpoints_examples.png", wavelength, clean, shift_variants, examples
    )
    return {
        "snv_noise_exact_angles_examples.png": (
            "Three measured fold-train SNV pixels, one per row, shared with the shift figure. "
            "Sample IDs and image coordinates identify the unmodified saved spectra. Left: clean SNV "
            "and exact 2.5°, 5°, and 7.5° tangent-direction rotations; right: "
            "perturbed-minus-clean residuals. The random tangent direction is shared "
            "across the three angles within each example. SNV axes: [-2, 2], step 0.5; "
            "residual axes: [-0.5, 0.5], step 0.1. Boundary triangles mark values outside "
            "the display range; stored values are unchanged."
        ),
        "snv_shift_exact_endpoints_examples.png": (
            "The same three measured fold-train SNV pixels as the noise figure, with the "
            "same figure size and three-row layout. Rows show Example 1 (+1, +2, +3 channels), "
            "Example 2 (-1, -2, -3 channels), and Example 3 (+1, +2, +3 channels). "
            "Left: clean and shifted SNV; "
            "right: shifted-minus-clean residuals. SNV axes: [-2, 2], step 0.5; "
            "residual axes: [-0.5, 0.5], step 0.1. Sample IDs and image coordinates "
            "identify the unmodified saved spectra. Boundary triangles mark values "
            "outside the display range; stored values are unchanged."
        ),
    }


def _visualization_metadata(
    seed: int,
    examples: pd.DataFrame,
    selection_details: dict[str, object],
    output: Path,
) -> dict[str, object]:
    return {
        "example_source": "measured SNV pixels from the existing fixed fold-train selection",
        "examples_in_row_order": examples.to_dict(orient="records"),
        "example_selection": selection_details,
        "source_selection_sha256": _digest(output / "selection.csv"),
        "normalization": "saved production SNV (ddof=1); no re-SNV, smoothing, or averaging",
        "same_clean_spectra_for_noise_and_shift": True,
        "metric_summary_population": (
            "All existing exact_examples_summary and uniform_distribution_summary "
            "values describe the full fixed fold-train subset, not only the three displayed pixels."
        ),
        "snv_limits": [-2.0, 2.0],
        "snv_tick_step": 0.5,
        "residual_limits": [-0.5, 0.5],
        "residual_tick_step": 0.1,
        "tick_direction": "out",
        "tick_sides": ["bottom", "left"],
        "wavelength_axis": {
            "limits": "wavelength grid endpoints; no horizontal padding",
            "endpoint_decimal_places": 2,
            "interior_tick_start_nm": 1000.0,
            "interior_tick_step_nm": 200.0,
        },
        "noise_layout": "3 rows x 2 columns; spectrum and residual",
        "shift_layout": "3 rows x 2 columns; Example 1 (+), Example 2 (-), Example 3 (+)",
        "shift_signs_in_row_order": list(SHIFT_EXAMPLE_SIGNS),
        "seed": seed,
        "source_script_sha256": _digest(Path(__file__)),
        "chemomae": version("chemomae"),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def _refresh_existing_plots(output: Path, processed: Path) -> None:
    summary_path = output / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    if _digest(output / "selection.csv") != report["artifacts_sha256"]["selection.csv"]:
        raise ValueError("selection.csv differs from the saved sanity report")
    spectra, wavelength, selection = _load_saved_subset(output, processed)
    if len(selection) != report["selection"]["pixel_count"]:
        raise ValueError("Saved selection size differs from the sanity report")
    if not (selection["fold"] == report["fold"]).all():
        raise ValueError("Saved selection belongs to a different fold")
    if set(selection["sample_id"]) != set(report["selection"]["sample_ids"]):
        raise ValueError("Saved sample IDs differ from the sanity report")
    recorded_grid = report["wavelength_grid"]
    if not np.allclose(
        [wavelength[0], wavelength[-1], np.mean(np.diff(wavelength))],
        [recorded_grid["start_nm"], recorded_grid["end_nm"], recorded_grid["mean_step_nm"]],
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("Processed wavelength grid differs from the existing sanity report")
    seed = int(report["seed"])
    clean, examples, selection_details = _select_measured_examples(spectra, selection)
    captions = _write_example_figures(output, wavelength, seed, clean, examples)
    for name in OBSOLETE_PLOTS:
        (output / name).unlink(missing_ok=True)
        report["artifacts_sha256"].pop(name, None)
    report["captions"] = captions
    report["selection"].pop("examples_in_row_order", None)
    report["visualization"] = _visualization_metadata(seed, examples, selection_details, output)
    for name in captions:
        report["artifacts_sha256"][name] = _digest(output / name)
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "updated_figures": list(captions)}))


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if args.plots_only:
        _refresh_existing_plots(output, args.processed_dir.resolve())
        return 0
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    experiment = args.experiment_dir.resolve()
    processed = args.processed_dir.resolve()
    metadata = args.metadata.resolve()
    clean, wavelength, selection = _load_train_subset(
        experiment, processed, metadata, args.fold
    )
    noise_exact_variants = {
        angle: _apply(clean, args.seed, noise_range=(angle, angle))
        for angle in NOISE_ANGLES
    }
    noise_uniform_variants = {
        upper: _apply(clean, args.seed, noise_range=(0.0, upper))
        for upper in NOISE_ANGLES
    }
    shift_endpoints = tuple(-value for value in reversed(SHIFT_MAGNITUDES)) + SHIFT_MAGNITUDES
    shift_exact_variants = {
        delta: _apply(clean, args.seed, shift_range=(delta, delta))
        for delta in shift_endpoints
    }
    shift_uniform_variants = {
        magnitude: _apply(clean, args.seed, shift_range=(-magnitude, magnitude))
        for magnitude in SHIFT_MAGNITUDES
    }
    metric_parts = [
        _metrics(clean, values, selection, kind="noise_exact", strength=angle)
        for angle, values in noise_exact_variants.items()
    ] + [
        _metrics(clean, values, selection, kind="noise_uniform", strength=upper)
        for upper, values in noise_uniform_variants.items()
    ] + [
        _metrics(clean, values, selection, kind="shift_exact", strength=delta)
        for delta, values in shift_exact_variants.items()
    ] + [
        _metrics(clean, values, selection, kind="shift_uniform", strength=magnitude)
        for magnitude, values in shift_uniform_variants.items()
    ]
    metrics = pd.concat(metric_parts, ignore_index=True)

    for angle in NOISE_ANGLES:
        realized = metrics.loc[
            (metrics["kind"] == "noise_exact") & (metrics["strength"] == angle),
            "spectral_angle_deg",
        ]
        if float(np.max(np.abs(realized - angle))) > 2e-3:
            raise RuntimeError(f"Noise augmenter did not apply exact angle {angle:g}")
    if float(metrics["perturbed_mean"].abs().max()) > 2e-6:
        raise RuntimeError("Perturbation did not preserve the zero-mean SNV constraint")
    if float(metrics["norm_absolute_error"].max()) > 5e-6:
        raise RuntimeError("Perturbation did not preserve the input norm")

    selection.to_csv(output / "selection.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    example_spectra, examples, selection_details = _select_measured_examples(clean, selection)
    captions = _write_example_figures(output, wavelength, args.seed, example_spectra, examples)
    artifacts = (
        "selection.csv",
        "metrics.csv",
        *captions,
    )
    report = {
        "status": "augmentation_strength_sanity_check_completed",
        "scope": (
            "fixed fold-train SNV pixels; no model training, outer-test pixels, "
            "or model-evaluation metrics"
        ),
        "fold": args.fold,
        "seed": args.seed,
        "noise": {
            "visualized_exact_angles_deg": list(NOISE_ANGLES),
            "distribution_uniform_ranges_deg": [
                [0.0, upper] for upper in NOISE_ANGLES
            ],
            "noise_prob": 1.0,
            "shift_prob": 0.0,
            "same_random_draws_across_candidate_maxima": True,
            "exact_examples_summary": _summary(metrics, "noise_exact"),
            "uniform_distribution_summary": _summary(metrics, "noise_uniform"),
        },
        "shift": {
            "candidate_ranges_channels": [
                [-magnitude, magnitude] for magnitude in SHIFT_MAGNITUDES
            ],
            "visualized_exact_endpoints_channels": list(shift_endpoints),
            "distribution_uniform_ranges_channels": [
                [-magnitude, magnitude] for magnitude in SHIFT_MAGNITUDES
            ],
            "noise_prob": 0.0,
            "shift_prob": 1.0,
            "same_random_draws_across_candidate_ranges": True,
            "exact_examples_summary": _summary(metrics, "shift_exact"),
            "uniform_distribution_summary": _summary(metrics, "shift_uniform"),
        },
        "selection": {
            "algorithm": (
                "8 evenly spaced sorted fold-train sample IDs; 128 evenly spaced positions "
                "within each sample's fixed manifest rows"
            ),
            "sample_ids": list(dict.fromkeys(selection["sample_id"])),
            "sample_count": int(selection["sample_id"].nunique()),
            "pixel_count": len(selection),
        },
        "wavelength_grid": {
            "start_nm": float(wavelength[0]),
            "end_nm": float(wavelength[-1]),
            "mean_step_nm": float(np.mean(np.diff(wavelength))),
        },
        "constraint_error_max": {
            "perturbed_mean_absolute": float(metrics["perturbed_mean"].abs().max()),
            "norm_absolute": float(metrics["norm_absolute_error"].max()),
        },
        "captions": captions,
        "visualization": _visualization_metadata(args.seed, examples, selection_details, output),
        "source": {
            "experiment_dir": str(experiment),
            "processed_dir": str(processed),
            "manifest_completion_sha256": _digest(experiment / "manifests/complete.json"),
            "chemomae": version("chemomae"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "artifacts_sha256": {name: _digest(output / name) for name in artifacts},
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
