"""One static 2x5 density/cluster figure from saved CPU PCA coordinates."""

from __future__ import annotations

from pathlib import Path
import stat
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .global_pca import (
    CONDITIONS, FIGURE_DIRECTORY, INPUT_DIRECTORY, RESULT_DIRECTORY, check_artifacts,
    check_pca, digest, finish_artifacts, read_json, require, write_json,
)

FIGURE_NAME = "01_pca_density_clusters.png"
FONT_SIZES = {"condition": 22, "axis": 18, "tick": 15, "centroid": 14}


def centroids(coordinates: np.ndarray, labels: np.ndarray, condition: str) -> pd.DataFrame:
    require(coordinates.shape == (len(labels), 2) and np.isfinite(coordinates).all()
            and np.isin(labels, np.arange(1, 9)).all(), "Invalid centroid inputs")
    rows = []
    for cluster in range(1, 9):
        selected = coordinates[labels == cluster]
        center = selected.astype(np.float64).mean(axis=0) if len(selected) else (np.nan, np.nan)
        rows.append({"condition": condition, "cluster": cluster, "pixels": len(selected),
                     "pc1": center[0], "pc2": center[1]})
    return pd.DataFrame(rows)


def plot_summary(
    coordinates: dict[str, np.ndarray], pixels: pd.DataFrame, palette: list[str],
    output: Path, *, dpi: int, seed: int, explained_variance: dict[str, np.ndarray],
) -> dict:
    require(len(palette) == 8 and dpi > 0, "Invalid PCA plot style")
    for condition in CONDITIONS:
        require(coordinates[condition].shape == (len(pixels), 2)
                and np.isfinite(coordinates[condition]).all(), "Invalid PCA plot coordinates")
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), squeeze=False)
    cmap = ListedColormap(palette)
    cluster_norm = BoundaryNorm(np.arange(.5, 9.5), cmap.N)
    order = np.random.default_rng(seed).permutation(len(pixels))
    hexagons, tables, centers, limits = [], [], [], {}
    try:
        for ci, condition in enumerate(CONDITIONS):
            values = coordinates[condition]
            labels = pixels[f"{condition}_cluster"].to_numpy()
            # Raw B1 scores and normalized latent projections have different units.
            half = max(float(np.ptp(values, axis=0).max()) * .53, 1e-6)
            middle = (values.min(axis=0) + values.max(axis=0)) / 2
            extent = (float(middle[0] - half), float(middle[0] + half),
                      float(middle[1] - half), float(middle[1] + half))
            limits[condition] = extent
            axes[0, ci].set_title(condition, fontsize=FONT_SIZES["condition"],
                                  weight="bold", pad=16)
            hexbins = axes[0, ci].hexbin(values[:, 0], values[:, 1], gridsize=70,
                                       extent=extent, mincnt=1, cmap="turbo", linewidths=0)
            hexagons.append(hexbins)
            offsets, counts = hexbins.get_offsets(), np.asarray(hexbins.get_array())
            require(int(counts.sum()) == len(pixels), "Hexbin lost input pixels")
            tables.append(pd.DataFrame({"condition": condition, "pc1": offsets[:, 0],
                                        "pc2": offsets[:, 1], "pixels": counts.astype(int)}))
            scatter = axes[1, ci].scatter(values[order, 0], values[order, 1], c=labels[order],
                cmap=cmap, norm=cluster_norm, s=.6, alpha=1, linewidths=0, rasterized=True)
            center_table = centroids(values, labels, condition)
            centers.append(center_table)
            for center in center_table.itertuples(index=False):
                if center.pixels:
                    axes[1, ci].scatter(center.pc1, center.pc2, s=280,
                                       facecolors="white", edgecolors="black", linewidths=.8,
                                       zorder=3)
                    axes[1, ci].text(center.pc1, center.pc2, str(center.cluster),
                                    ha="center", va="center", fontsize=FONT_SIZES["centroid"],
                                    weight="bold", zorder=4)
            ratio = explained_variance[condition]
            require(ratio.shape == (2,) and np.isfinite(ratio).all(), "Invalid PCA variance ratios")
            for axis in axes[:, ci]:
                axis.set(xlim=extent[:2], ylim=extent[2:])
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel(f"PC1 ({100 * ratio[0]:.1f}%)", fontsize=FONT_SIZES["axis"])
                axis.set_ylabel(f"PC2 ({100 * ratio[1]:.1f}%)", fontsize=FONT_SIZES["axis"])
                axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
                axis.tick_params(direction="out", labelsize=FONT_SIZES["tick"])
                axis.xaxis.get_offset_text().set_fontsize(FONT_SIZES["tick"])
                axis.yaxis.get_offset_text().set_fontsize(FONT_SIZES["tick"])
        maximum = max(float(artist.get_array().max()) for artist in hexagons)
        density_norm = LogNorm(vmin=1, vmax=max(2, maximum))
        for artist in hexagons:
            artist.set_norm(density_norm)
        fig.subplots_adjust(left=.04, right=.92, top=.92, bottom=.10, wspace=.45, hspace=.42)
        density_bar = fig.add_axes((.943, .59, .012, .29))
        fig.colorbar(hexagons[0], cax=density_bar).set_label(
            "Pixels / hexagon (log scale)", fontsize=FONT_SIZES["axis"])
        cluster_bar = fig.add_axes((.943, .14, .012, .29))
        fig.colorbar(scatter, cax=cluster_bar, ticks=np.arange(1, 9)).set_label(
            "Cluster ID", fontsize=FONT_SIZES["axis"])
        for bar in (density_bar, cluster_bar):
            bar.tick_params(labelsize=FONT_SIZES["tick"])
        fig.savefig(output / FIGURE_NAME, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    pd.concat(tables, ignore_index=True).to_csv(output / "hexbin_counts.csv", index=False)
    pd.concat(centers, ignore_index=True).to_csv(output / "centroids.csv", index=False)
    pd.DataFrame([{"file": FIGURE_NAME, "caption":
        "Columns: B0, B1, A0, M00, M11. Top: pixel counts per hexagon, shared logarithmic "
        "turbo scale. Bottom: SNV-aligned Cluster ID; numbered circles are arithmetic "
        "means of member PCA coordinates. All conditions use the same selected pixels. "
        "B1 uses the saved baseline's top two PCA scores before L2 normalization, with its "
        "original explained variance ratios. Other conditions use centered PCA of their "
        "clustering representations. Each column has its own coordinate range and hexagon "
        "area; counts share one color scale. Axis percentages are explained variance ratios."}]).to_csv(
            output / "captions.csv", index=False)
    return {"density": {"cmap": "turbo", "normalization": "log", "vmin": 1,
                         "vmax": max(2, maximum), "gridsize": 70, "extent": limits},
            "centroid": "arithmetic mean of displayed member coordinates; not transformed centers",
            "columns": list(CONDITIONS), "rows": ["pixel count", "display Cluster ID"],
            "axis_ticks": True, "condition_titles": True, "font_sizes": FONT_SIZES,
            "extent_scope": "per condition; identical between rows",
            "point_order_seed": seed, "dpi": dpi}


def _sources(experiment: Path) -> dict:
    return {condition: digest(experiment / RESULT_DIRECTORY / condition / "completion.json")
            for condition in CONDITIONS}


def render_pca(experiment: Path, *, dpi: int = 240) -> dict:
    check_pca(experiment)
    inputs = experiment / INPUT_DIRECTORY
    record = read_json(inputs / "inputs.json")
    pixels = pd.read_csv(inputs / "pixels.csv")
    coordinates = {c: np.load(experiment / RESULT_DIRECTORY / c / "coordinates.npy",
                               allow_pickle=False) for c in CONDITIONS}
    variance_tables = {c: pd.read_csv(experiment / RESULT_DIRECTORY / c / "explained_variance.csv")
                       for c in CONDITIONS}
    explained_variance = {c: table.explained_variance_ratio.to_numpy()
                          for c, table in variance_tables.items()}
    sources = _sources(experiment)
    output = experiment / FIGURE_DIRECTORY
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pca-plot-", dir=output.parent.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        style = plot_summary(coordinates, pixels, record["palette"], stage,
                             dpi=dpi, seed=record["plot_seed"], explained_variance=explained_variance)
        pixels.to_csv(stage / "pixels.csv", index=False)
        pd.concat([table.assign(condition=c) for c, table in variance_tables.items()],
                  ignore_index=True).to_csv(stage / "explained_variance.csv", index=False)
        write_json(stage / "report.json", {"sources": sources, "pixels": len(pixels),
            "input_completion_sha256": digest(inputs / "completion.json"),
            "code_sha256": digest(Path(__file__)), "style": style,
            "runtime": {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__},
            "interpretation": "Hexbin counts describe PCA sampling density, not chemical "
                "concentration or original-space density. Embeddings have independent axes."})
        finish_artifacts(stage, "pca_figure_completed")
        require(sources == _sources(experiment), "PCA coordinates changed during rendering")
        # Replace only this figure's known directory; never touch fits or other figures.
        expected = experiment.resolve() / FIGURE_DIRECTORY
        require(output.resolve() == expected, "PCA figure path escapes output")
        previous = Path(temporary) / "previous"
        if output.exists():
            for path in (output, *output.rglob("*")):
                attributes = getattr(path.lstat(), "st_file_attributes", 0)
                require(not path.is_symlink()
                        and not attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
                        and path.resolve().is_relative_to(expected), "Linked PCA figure path")
            output.rename(previous)
        try:
            stage.rename(output)
        except OSError:
            if previous.exists():
                previous.rename(output)
            raise
    return check_pca_figure(experiment)


def check_pca_figure(experiment: Path) -> dict:
    check_pca(experiment)
    output = experiment / FIGURE_DIRECTORY
    check_artifacts(output, "pca_figure_completed")
    report = read_json(output / "report.json")
    require(report["sources"] == _sources(experiment)
            and report["code_sha256"] == digest(Path(__file__))
            and report["input_completion_sha256"] == digest(
                experiment / INPUT_DIRECTORY / "completion.json"), "PCA figure source differs")
    require([p.name for p in output.glob("*.png")] == [FIGURE_NAME], "Expected one PCA PNG")
    return {"status": "validated_pca_figure", "checks_passed": True,
            "png_count": 1, "pixels": report["pixels"], "output": str(output / FIGURE_NAME)}
