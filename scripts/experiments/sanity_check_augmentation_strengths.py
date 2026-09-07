"""Plot maximum examples and sampled augmentation ranges on fixed train SNV pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
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


NOISE_ANGLES = (2.5, 5.0, 7.5)
SHIFT_MAGNITUDES = (1.0, 2.0, 3.0)
SAMPLE_COUNT = 8
PIXELS_PER_SAMPLE = 128
EXAMPLE_COUNT = 4
COLORS = ("#0072B2", "#E69F00", "#CC79A7")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--seed", type=int, default=20260907)
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


def _example_indices() -> np.ndarray:
    starts = np.arange(SAMPLE_COUNT, dtype=np.int64) * PIXELS_PER_SAMPLE
    return starts[_spaced_indices(SAMPLE_COUNT, EXAMPLE_COUNT)] + PIXELS_PER_SAMPLE // 2


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
) -> None:
    indices = _example_indices()
    residual_limit = max(
        float(np.max(np.abs(values[indices] - clean[indices])))
        for values in variants.values()
    ) * 1.08
    figure, axes = plt.subplots(
        EXAMPLE_COUNT, 2, figsize=(14.5, 11.0), dpi=180, sharex=True,
        constrained_layout=True,
    )
    for row, spectrum_index in enumerate(indices):
        left, right = axes[row]
        left.plot(wavelength, clean[spectrum_index], color="0.15", linewidth=1.35, label="clean")
        for color, (angle, values) in zip(COLORS, variants.items(), strict=True):
            label = f"{angle:g}°"
            left.plot(wavelength, values[spectrum_index], color=color, linewidth=0.95, label=label)
            right.plot(
                wavelength,
                values[spectrum_index] - clean[spectrum_index],
                color=color,
                linewidth=0.95,
                label=label,
            )
        left.set_ylabel("SNV")
        left.grid(alpha=0.18)
        right.axhline(0.0, color="0.4", linewidth=0.7)
        right.set_ylim(-residual_limit, residual_limit)
        right.set_ylabel("Perturbed - clean (SNV)")
        right.grid(alpha=0.18)
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
) -> None:
    indices = _example_indices()
    residual_limit = max(
        float(np.max(np.abs(values[indices] - clean[indices])))
        for values in variants.values()
    ) * 1.08
    figure, axes = plt.subplots(
        EXAMPLE_COUNT, 3, figsize=(18.0, 11.0), dpi=180, sharex=True,
        constrained_layout=True,
    )
    for row, spectrum_index in enumerate(indices):
        negative, positive, residual = axes[row]
        for axis in (negative, positive):
            axis.plot(
                wavelength,
                clean[spectrum_index],
                color="0.15",
                linewidth=1.35,
                label="clean",
            )
        for color, magnitude in zip(COLORS, SHIFT_MAGNITUDES, strict=True):
            negative.plot(
                wavelength,
                variants[-magnitude][spectrum_index],
                color=color,
                linewidth=0.95,
                label=f"{-magnitude:+g} channels",
            )
            positive.plot(
                wavelength,
                variants[magnitude][spectrum_index],
                color=color,
                linewidth=0.95,
                label=f"{magnitude:+g} channels",
            )
            for delta, style in ((-magnitude, "--"), (magnitude, "-")):
                residual.plot(
                    wavelength,
                    variants[delta][spectrum_index] - clean[spectrum_index],
                    color=color,
                    linestyle=style,
                    linewidth=0.9,
                    label=f"{delta:+g}" if row == 0 else None,
                )
        negative.set_ylabel("SNV")
        negative.grid(alpha=0.18)
        positive.set_ylabel("SNV")
        positive.grid(alpha=0.18)
        residual.axhline(0.0, color="0.4", linewidth=0.7)
        residual.set_ylim(-residual_limit, residual_limit)
        residual.set_ylabel("Shifted - clean (SNV)")
        residual.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 2].legend(frameon=False, ncol=2, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Wavelength (nm)")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_distribution(
    output: Path,
    metrics: pd.DataFrame,
    *,
    kind: str,
    strengths: tuple[float, ...],
    labels: list[str],
) -> None:
    columns = (
        ("spectral_angle_deg", "Spectral angle from clean (degrees)"),
        ("rms_difference_snv", "RMS difference (SNV)"),
        ("max_abs_difference_snv", "Maximum absolute band difference (SNV)"),
    )
    subset = metrics.loc[metrics["kind"] == kind]
    color_by_magnitude = dict(zip(SHIFT_MAGNITUDES, COLORS, strict=True))
    colors = COLORS if kind.startswith("noise") else tuple(
        color_by_magnitude[abs(strength)] for strength in strengths
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.2), dpi=180, constrained_layout=True
    )
    for axis, (column, y_label) in zip(axes, columns, strict=True):
        series = [
            subset.loc[subset["strength"] == strength, column].to_numpy()
            for strength in strengths
        ]
        artists = axis.boxplot(series, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(artists["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.62)
        axis.set_ylabel(y_label)
        axis.grid(axis="y", alpha=0.22)
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


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
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
    _plot_noise_examples(
        output / "snv_noise_exact_angles_examples.png",
        wavelength,
        clean,
        noise_exact_variants,
    )
    _plot_shift_examples(
        output / "snv_shift_exact_endpoints_examples.png",
        wavelength,
        clean,
        shift_exact_variants,
    )
    _plot_distribution(
        output / "snv_noise_uniform_ranges_distributions.png",
        metrics,
        kind="noise_uniform",
        strengths=NOISE_ANGLES,
        labels=[f"U(0, {angle:g}°)" for angle in NOISE_ANGLES],
    )
    _plot_distribution(
        output / "snv_shift_uniform_ranges_distributions.png",
        metrics,
        kind="shift_uniform",
        strengths=SHIFT_MAGNITUDES,
        labels=[f"U(-{magnitude:g}, {magnitude:g})" for magnitude in SHIFT_MAGNITUDES],
    )

    captions = {
        "snv_noise_exact_angles_examples.png": (
            "Four fixed fold-1 train pixels, one per row. Left: clean SNV and exact "
            "2.5°, 5°, and 7.5° tangent-direction rotations. Right: perturbed-minus-clean "
            "residuals on one common vertical scale. The random tangent direction is shared "
            "across the three angles within each pixel."
        ),
        "snv_noise_uniform_ranges_distributions.png": (
            "Noise sampled independently per spectrum from U(0, 2.5°), U(0, 5°), or "
            "U(0, 7.5°) over 1,024 fixed fold-1 train pixels. Panels from left to right "
            "show spectral angle, per-spectrum RMS difference, and maximum absolute "
            "single-band difference."
        ),
        "snv_shift_exact_endpoints_examples.png": (
            "Four fixed fold-1 train pixels, one per row. Columns from left to right show "
            "clean SNV with negative endpoints, clean SNV with positive endpoints, and "
            "residuals. Dashed residuals are negative shifts and solid residuals are positive."
        ),
        "snv_shift_uniform_ranges_distributions.png": (
            "Shift sampled independently per spectrum from U(-1, 1), U(-2, 2), or "
            "U(-3, 3) channels over 1,024 fixed fold-1 train pixels. Panels from left to "
            "right show spectral angle, per-spectrum RMS difference, and maximum absolute "
            "single-band difference."
        ),
    }
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
            "examples_in_row_order": selection.iloc[_example_indices()][[
                "sample_id", "hdf5_row", "pixel_row", "pixel_col"
            ]].to_dict(orient="records"),
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
