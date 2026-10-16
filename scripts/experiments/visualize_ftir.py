"""Export per-sample M11 K8 maps and saved NIR derivatives for FT-IR comparison.

Only the horizontal coordinate is converted to wavenumber. The saved SG second
derivative remains with respect to wavelength (nm), without recomputation.
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
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MultipleLocator


DERIVATIVE_LIMIT = 2.5e-4
WAVENUMBER_START = 10_000.0


@dataclass(frozen=True)
class SelectedCurve:
    """An explicitly selected observed curve; notes never become legend labels."""

    source_sample_id: str
    cluster: int
    note: str

    def legend_label(self, target_sample_id: str) -> str:
        if self.source_sample_id == target_sample_id:
            return str(self.cluster)
        return f"{self.source_sample_id}: {self.cluster}"


def load_selection(path: Path, sample_ids: list[str]) -> dict[str, tuple[SelectedCurve, ...]]:
    """Validate the editable selection and retain its group/cluster order."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 \
            or payload.get("condition") != "M11":
        raise ValueError(f"Expected an M11 selection with schema_version=1: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, dict) or set(samples) != set(sample_ids):
        actual = set(samples) if isinstance(samples, dict) else set()
        raise ValueError(f"Selection IDs differ: missing={sorted(set(sample_ids) - actual)}; "
                         f"unknown={sorted(actual - set(sample_ids), key=str)}")
    selections = {}
    for sample_id in sample_ids:
        groups = samples[sample_id]
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"{sample_id}: expected a nonempty list of cluster groups")
        selected: list[SelectedCurve] = []
        seen: set[tuple[str, int]] = set()
        for group in groups:
            if not isinstance(group, dict) \
                    or set(group) - {"source_sample_id", "clusters", "note"}:
                raise ValueError(f"{sample_id}: invalid cluster group {group!r}")
            source_id = group.get("source_sample_id", sample_id)
            clusters, note = group.get("clusters"), group.get("note", "")
            if source_id not in sample_ids or not isinstance(note, str):
                raise ValueError(f"{sample_id}: invalid source_sample_id or note")
            if not isinstance(clusters, list) or not clusters \
                    or any(type(cluster) is not int or not 1 <= cluster <= 8
                           for cluster in clusters):
                raise ValueError(f"{sample_id}: clusters must be integers from 1 to 8")
            for cluster in clusters:
                key = (source_id, cluster)
                if key in seen:
                    raise ValueError(f"{sample_id}: duplicate selected curve {key}")
                seen.add(key)
                selected.append(SelectedCurve(source_id, cluster, note))
        selections[sample_id] = tuple(selected)
    return selections


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path,
        default=root / "outputs/experiments/global_v1/results/figures/global_k8_v1",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/ftir")
    parser.add_argument(
        "--selection", type=Path, default=Path(__file__).with_name("ftir_plot_selection.yaml"),
        help="Editable sample/cluster selection and notes; notes are not legend labels.",
    )
    parser.add_argument("--sample-ids", nargs="+", help="Regenerate only these target samples.")
    parser.add_argument("--nir-only", action="store_true", help="Leave sample map PNGs unchanged.")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true", help="Replace the exported PNGs.")
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("DPI must be positive")
    return args


def save_figure(
    figure: plt.Figure, path: Path, dpi: int, description: str, *, overwrite: bool = False,
    pad_inches: float = 0.1,
) -> None:
    """Save a PNG with provenance, replacing an existing file only on request."""
    try:
        with path.open("wb" if overwrite else "xb") as handle:
            figure.savefig(
                handle, format="png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches,
                facecolor="white",
                metadata={"Title": path.stem, "Description": description},
            )
    finally:
        plt.close(figure)


def render_map(
    sample_id: str, labels: np.ndarray, palette: list[str], path: Path, dpi: int,
    source: Path, *, overwrite: bool = False,
) -> None:
    """Keep the saved display IDs, colors, full field, orientation and aspect ratio."""
    figure, axis = plt.subplots(figsize=(6, 6 * labels.shape[0] / labels.shape[1] + 1.3))
    axis.imshow(
        labels, cmap=ListedColormap(palette),
        norm=BoundaryNorm(np.arange(-0.5, 9.5), 9), interpolation="nearest",
        origin="upper", aspect="equal",
    )
    axis.set_title(sample_id, fontsize=28, fontweight="bold", pad=14)
    axis.set_axis_off()
    mappable = matplotlib.cm.ScalarMappable(
        norm=BoundaryNorm(np.arange(0.5, 9.5), 8), cmap=ListedColormap(palette[1:]),
    )
    bar = figure.colorbar(
        mappable, ax=axis, orientation="horizontal", ticks=np.arange(1, 9),
        fraction=0.045, pad=0.025, aspect=30,
    )
    bar.set_label("Cluster ID", fontsize=22, labelpad=10)
    bar.ax.tick_params(labelsize=22, pad=6)
    figure.tight_layout()
    save_figure(
        figure, path, dpi,
        f"{source}; {sample_id}; M11 K8 repeat 1; saved M00-aligned display IDs; "
        "background 0; no spatial smoothing, rotation or crop.",
        overwrite=overwrite,
    )


def render_spectrum(
    sample_id: str, wavelength: np.ndarray, spectra: pd.DataFrame, counts: pd.DataFrame,
    palette: list[str], path: Path, dpi: int, source: Path,
    selection: tuple[SelectedCurve, ...], *, overwrite: bool = False,
) -> None:
    """Plot only selected observed curves, identifying any cross-sample references."""
    wavenumber = 1e7 / wavelength
    visible = wavenumber <= WAVENUMBER_START
    band_names = [f"band_{index:03d}" for index in range(len(wavelength))]
    figure, axis = plt.subplots(figsize=(13, 4.6))
    overflow_curves = []
    curve_records = []
    # Keep cross-sample reference curves on the right, after ascending local IDs.
    ordered = sorted(selection, key=lambda curve: (
        curve.source_sample_id != sample_id, curve.source_sample_id, curve.cluster,
    ))
    for selected in ordered:
        key = (selected.source_sample_id, selected.cluster)
        row, count = spectra.loc[key], counts.loc[key]
        cluster = selected.cluster
        label = selected.legend_label(sample_id)
        curve_records.append({
            "source_sample_id": selected.source_sample_id, "cluster": cluster,
            "legend_label": label, "pixel_count": int(row["pixel_count"]),
            "excluded_pixels": int(count["pseudoabsorbance_excluded_pixels"]),
            "plotted": bool(row["pixel_count"] > 0),
        })
        if row["pixel_count"] > 0:
            values = row[band_names].to_numpy(dtype=float)
            axis.plot(
                wavenumber, values,
                color=palette[cluster], linewidth=1.4, label=label,
            )
            if np.any((np.abs(values) > DERIVATIVE_LIMIT) & visible):
                overflow_curves.append(label)
            for outside, boundary, marker in (
                ((values > DERIVATIVE_LIMIT) & visible, DERIVATIVE_LIMIT, "^"),
                ((values < -DERIVATIVE_LIMIT) & visible, -DERIVATIVE_LIMIT, "v"),
            ):
                axis.plot(wavenumber[outside], np.full(outside.sum(), boundary),
                          linestyle="none", marker=marker, markersize=4,
                          color=palette[cluster], clip_on=False, zorder=4)
        else:
            print(f"{sample_id}: omit {label}; no valid pseudoabsorbance pixels", flush=True)
    axis.set_xlim(WAVENUMBER_START, float(wavenumber[-1]))
    axis.set_ylim(-DERIVATIVE_LIMIT, DERIVATIVE_LIMIT)
    axis.set_yticks(np.linspace(-DERIVATIVE_LIMIT, DERIVATIVE_LIMIT, 11))
    axis.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=12)
    axis.set_ylabel(r"$d^2A/d\lambda^2$ (nm$^{-2}$)", fontsize=12)
    axis.xaxis.set_major_locator(MultipleLocator(500))
    axis.xaxis.set_minor_locator(MultipleLocator(100))
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4), useMathText=True)
    axis.set_axisbelow(True)
    axis.grid(which="major", color="0.83", linewidth=0.7)
    axis.grid(which="minor", axis="x", color="0.91", linewidth=0.5, linestyle=":")
    axis.axhline(0, color="0.55", linewidth=0.6, zorder=0)
    if overflow_curves:
        axis.text(0.99, 0.97, "Triangles: outside y-range", transform=axis.transAxes,
                  ha="right", va="top", fontsize=8, color="0.35",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    if any(record["plotted"] for record in curve_records):
        axis.legend(
            title="Cluster ID", ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
            frameon=False, handlelength=2, columnspacing=1.2, fontsize=14, title_fontsize=16,
            borderaxespad=0, borderpad=0, labelspacing=0.25,
        )
    else:
        axis.text(0.5, 0.5, "No valid pseudoabsorbance spectra", ha="center",
                  va="center", transform=axis.transAxes)
    figure.subplots_adjust(left=0.065, right=0.992, bottom=0.225, top=0.94)
    save_figure(
        figure, path, dpi,
        f"{source}; {sample_id}; M11 K8 repeat 1; saved within-sample cluster means; "
        "pixelwise A=-log10(R); SG(window=7, polyorder=2, deriv=2, delta in nm, "
        "mode=interp); x=1e7/wavelength_nm; y remains d2A/dlambda2 in nm^-2; "
        f"x-limits={WAVENUMBER_START:g} to {wavenumber[-1]:.6f} cm^-1; "
        f"y-limits=+/-{DERIVATIVE_LIMIT}; major x-ticks=500 cm^-1; minor=100 cm^-1; "
        f"out-of-range curves={overflow_curves}; triangles mark overflow; "
        "absent clusters omitted; no IQR. "
        f"Selected curves={json.dumps(curve_records)}.",
        overwrite=overwrite, pad_inches=0.03,
    )


def main() -> int:
    args = parse_args()
    source = args.report_dir.resolve()
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    contract = report["contract"]
    if contract["K"] != 8 or contract["repeat"] != 1 or "M11" not in contract["conditions"]:
        raise ValueError("Expected the saved global M11 K8 repeat-1 report")
    sample_ids = report["sample_ids"]
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Expected unique sample IDs")
    for sample_id in sample_ids:
        if not isinstance(sample_id, str) or not sample_id.startswith("KYOw") \
                or not sample_id[4:].isdigit():
            raise ValueError(f"Invalid sample ID: {sample_id!r}")
    selections = load_selection(args.selection, sample_ids)
    target_ids = args.sample_ids if args.sample_ids is not None else sample_ids
    if len(set(target_ids)) != len(target_ids) or not set(target_ids) <= set(sample_ids):
        raise ValueError("--sample-ids must contain unique saved sample IDs")
    palette = report["definitions"]["palette"]
    if len(palette) != 9:
        raise ValueError("Expected the saved background and eight cluster colors")
    grid = pd.read_csv(source / "wavelengths.csv")
    wavelength = grid.wavelength_nm.to_numpy(dtype=float)
    band_names = [f"band_{index:03d}" for index in range(256)]
    if grid.band.tolist() != band_names or not np.isfinite(wavelength).all() \
            or not np.all(wavelength > 0) or not np.all(np.diff(wavelength) > 0):
        raise ValueError("Invalid saved wavelength grid")
    spectra = pd.read_csv(source / "M11/sample_spectra.csv")
    spectra = spectra.loc[spectra.kind == "second_derivative"]
    counts = pd.read_csv(source / "M11/spectrum_counts.csv")
    expected = pd.MultiIndex.from_product([sample_ids, range(1, 9)], names=["sample_id", "cluster"])
    for table in (spectra, counts):
        table.set_index(["sample_id", "cluster"], inplace=True)
        if not table.index.is_unique or set(table.index) != set(expected) \
                or not table.condition.eq("M11").all():
            raise ValueError("M11 sample/cluster coverage differs between saved artifacts")
    spectra, counts = spectra.loc[expected], counts.loc[expected]
    values = spectra[band_names].to_numpy(dtype=float)
    valid = counts.pseudoabsorbance_pixels.to_numpy()
    excluded = counts.pseudoabsorbance_excluded_pixels.to_numpy()
    if not np.array_equal(spectra.pixel_count, valid) or np.any(valid < 0) \
            or np.any(excluded < 0) or not np.array_equal(counts.pixels, valid + excluded) \
            or not np.isfinite(values[valid > 0]).all() \
            or not np.isnan(values[valid == 0]).all():
        raise ValueError("Invalid saved derivative curves or contributing pixel counts")
    folders = ("nir",) if args.nir_only else ("samples", "nir")
    for folder in folders:
        for sample_id in target_ids:
            path = args.output_dir / folder / f"{sample_id}.png"
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite")
    with np.load(source / "M11/label_maps.npz", allow_pickle=False) as maps:
        if set(maps.files) != set(sample_ids):
            raise ValueError("Saved label maps differ from report sample IDs")
        # Validate all small saved maps before generating the first output.
        for sample_id in sample_ids:
            labels = maps[sample_id]
            if labels.ndim != 2 or 0 in labels.shape or labels.dtype.kind not in "iu" \
                    or np.any(labels < 0) or np.any(labels > 8):
                raise ValueError(f"{sample_id}: invalid saved label map")
            if not np.array_equal(np.bincount(labels.ravel(), minlength=9)[1:],
                                  counts.loc[sample_id].pixels.to_numpy()):
                raise ValueError(f"{sample_id}: map and spectral cluster pixel counts differ")
        for folder in ("samples", "nir", "ft-ir"):
            (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
        for index, sample_id in enumerate(target_ids, start=1):
            if not args.nir_only:
                render_map(sample_id, maps[sample_id], palette,
                           args.output_dir / "samples" / f"{sample_id}.png", args.dpi, source,
                           overwrite=args.overwrite)
            render_spectrum(sample_id, wavelength, spectra, counts,
                            palette, args.output_dir / "nir" / f"{sample_id}.png", args.dpi, source,
                            selections[sample_id], overwrite=args.overwrite)
            print(f"[{index}/{len(target_ids)}] {sample_id}: {' + '.join(folders)} PNG", flush=True)
    print(f"Saved to {args.output_dir.resolve()}; ft-ir/ reserved for future figures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
