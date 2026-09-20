"""CPU checks of shared inputs, centered PCA and the single static summary figure."""

from __future__ import annotations

from pathlib import Path

from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import pytest

from wood_degradation_map.experiments import global_pca as gu
from wood_degradation_map.experiments import global_pca_reporting as ur

PALETTE = ["#0072b2", "#e69f00", "#009e73", "#cc79a7",
           "#56b4e9", "#d55e00", "#f0e442", "#332288"]


@pytest.fixture
def portable(tmp_path: Path) -> Path:
    inputs = tmp_path / gu.INPUT_DIRECTORY
    inputs.mkdir(parents=True)
    rng = np.random.default_rng(13)
    pixels = pd.DataFrame({"point_id": np.arange(64), "sample_id": ["a"] * 32 + ["b"] * 32,
        "hdf5_row": np.tile(np.arange(32), 2), "pixel_row": np.tile(np.arange(32) // 8, 2),
        "pixel_col": np.tile(np.arange(32) % 8, 2)})
    for condition in gu.CONDITIONS:
        dimension = 256 if condition == "B0" else 2 if condition == "B1" else 16
        values = rng.normal(size=(64, dimension)).astype(np.float32)
        if condition == "B1":
            values *= 10  # Raw scores have a different scale from unit vectors.
        else:
            values /= np.linalg.norm(values, axis=1, keepdims=True)
        np.save(inputs / f"{condition}.npy", values, allow_pickle=False)
        pixels[f"{condition}_cluster"] = np.tile(np.arange(1, 9), 8)
    pixels.to_csv(inputs / "pixels.csv", index=False)
    np.savez_compressed(inputs / gu.BASELINE_PROJECTION, mean=np.zeros(256, dtype=np.float32),
        components=np.eye(256, dtype=np.float32)[:2],
        explained_variance=np.array([12, 5], dtype=np.float32),
        explained_variance_ratio=np.array([.24, .10], dtype=np.float32),
        singular_values=np.sqrt(np.array([12, 5], dtype=np.float32) * 63),
        source_component_index=np.array([4, 2]))
    gu.write_json(inputs / "inputs.json", {"schema_version": 2,
        "conditions": list(gu.CONDITIONS), "pixels": 64,
        "sample_ids": ["a", "b"], "pixels_per_sample": 32, "plot_seed": 17,
        "palette": PALETTE})
    gu.finish_artifacts(inputs, "pca_inputs_completed")
    return tmp_path


def test_inputs_reject_duplicate_identity_and_nonunit_features(portable: Path) -> None:
    inputs = portable / gu.INPUT_DIRECTORY
    _, pixels = gu.check_inputs(inputs)
    pixels.loc[1, "hdf5_row"] = pixels.loc[0, "hdf5_row"]
    pixels.to_csv(inputs / "pixels.csv", index=False)
    gu.finish_artifacts(inputs, "pca_inputs_completed")
    with pytest.raises(ValueError, match="pixel identity"):
        gu.check_inputs(inputs)
    pixels.loc[1, "hdf5_row"] = 1
    pixels.to_csv(inputs / "pixels.csv", index=False)
    values = np.load(inputs / "M11.npy")
    values[7] = 0
    np.save(inputs / "M11.npy", values, allow_pickle=False)
    gu.finish_artifacts(inputs, "pca_inputs_completed")
    with pytest.raises(ValueError, match="M11: invalid unit"):
        gu.check_inputs(inputs)


def test_pca_matches_full_svd_and_retains_original_feature_scales() -> None:
    rng = np.random.default_rng(14)
    values = rng.normal(size=(96, 16)) * np.linspace(1, 8, 16) + np.arange(16)
    coordinates, projection, runtime = gu.fit_projection(values)
    centered = values - values.mean(axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    expected = centered @ components[:2].T
    np.testing.assert_allclose(coordinates @ coordinates.T, expected @ expected.T, atol=1e-10)
    np.testing.assert_allclose(projection["explained_variance"], singular[:2] ** 2 / 95)
    np.testing.assert_allclose(projection["explained_variance_ratio"],
                               singular[:2] ** 2 / np.square(singular).sum())
    np.testing.assert_allclose(projection["mean"], values.mean(axis=0))
    np.testing.assert_allclose(coordinates.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(coordinates,
                               (values - projection["mean"]) @ projection["components"].T)
    assert runtime["device"] == "cpu" and coordinates.dtype == np.float64


def test_baseline_component_selection_uses_descending_singular_values() -> None:
    singular = np.ones(16, dtype=np.float32)
    singular[[4, 2]] = [30, 20]
    np.testing.assert_array_equal(gu.baseline_component_indices(singular), [4, 2])
    singular[2] = 30
    np.testing.assert_array_equal(gu.baseline_component_indices(singular), [2, 4])
    singular[0] = np.nan
    with pytest.raises(ValueError, match="baseline PCA singular"):
        gu.baseline_component_indices(singular)


def test_check_identifies_unfinished_fit_before_rendering(portable: Path) -> None:
    with pytest.raises(FileNotFoundError, match="B0: PCA fit is not complete.*before rendering"):
        gu.check_pca(portable)


def test_centroids_are_member_means_and_keep_empty_clusters() -> None:
    coordinates = np.array([[0, 0], [6, 0], [0, 3], [10, 10]], dtype=np.float32)
    centers = ur.centroids(coordinates, np.array([2, 2, 2, 8]), "M11").set_index("cluster")
    np.testing.assert_allclose(centers.loc[2, ["pc1", "pc2"]].astype(float), [2, 1])
    assert centers.loc[2, "pixels"] == 3 and centers.loc[1, "pixels"] == 0
    assert np.isnan(centers.loc[1, "pc1"])
    np.testing.assert_allclose(centers.loc[8, ["pc1", "pc2"]].astype(float), [10, 10])


def test_resume_single_figure_shared_norm_and_tamper(
    portable: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    actual_fit = gu.fit_projection

    def fit(values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
        calls.append(values.shape)
        return actual_fit(values)

    monkeypatch.setattr(gu, "fit_projection", fit)
    assert gu.run_pca(portable)["pixels"] == 64
    assert calls == [(64, 256)] + [(64, 16)] * 3
    baseline_scores = np.load(portable / gu.INPUT_DIRECTORY / "B1.npy")
    np.testing.assert_array_equal(
        np.load(portable / gu.RESULT_DIRECTORY / "B1/coordinates.npy"), baseline_scores)
    with np.load(portable / gu.RESULT_DIRECTORY / "B1/projection.npz") as projection:
        np.testing.assert_array_equal(projection["source_component_index"], [4, 2])
        np.testing.assert_array_equal(projection["explained_variance_ratio"],
                                      np.array([.24, .10], dtype=np.float32))
        assert projection["components"].shape == (2, 256)
    gu.run_pca(portable, resume=True)
    assert len(calls) == 4
    with pytest.raises(FileExistsError):
        gu.run_pca(portable)

    actual_savefig = ur.plt.Figure.savefig
    seen = []

    def inspect_figure(figure: ur.plt.Figure, *args: object, **kwargs: object) -> None:
        panels = figure.axes[:10]
        assert len(panels) == 10
        norm = panels[0].collections[0].norm
        assert isinstance(norm, LogNorm) and norm.vmin == 1
        for condition, upper, lower in zip(gu.CONDITIONS, panels[:5], panels[5:], strict=True):
            assert upper.collections[0].norm is norm
            assert upper.collections[0].get_cmap().name == "turbo"
            assert upper.get_xlim() == lower.get_xlim() and upper.get_ylim() == lower.get_ylim()
            assert len(upper.get_xticks()) and len(lower.get_yticks())
            assert upper.get_xlabel().startswith("PC1 (")
            assert lower.get_ylabel().startswith("PC2 (")
            assert upper.get_title() == condition and not lower.get_title()
        assert np.ptp(panels[1].get_xlim()) > 5 * np.ptp(panels[0].get_xlim())
        seen.append(norm.vmax)
        actual_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(ur.plt.Figure, "savefig", inspect_figure)
    result = ur.render_pca(portable, dpi=30)
    assert result["png_count"] == 1 and len(seen) == 1
    output = portable / gu.FIGURE_DIRECTORY
    assert output.name == "pca-latent-2d"
    bins = pd.read_csv(output / "hexbin_counts.csv")
    assert (bins.groupby("condition").pixels.sum() == 64).all()
    assert seen[0] == max(2, bins.pixels.max())
    assert sorted(p.name for p in output.glob("*.png")) == [ur.FIGURE_NAME]
    centers = pd.read_csv(output / "centroids.csv")
    pixels = pd.read_csv(portable / gu.INPUT_DIRECTORY / "pixels.csv")
    for condition in gu.CONDITIONS:
        coordinates = np.load(portable / gu.RESULT_DIRECTORY / condition / "coordinates.npy")
        for row in centers.loc[centers.condition == condition].itertuples(index=False):
            expected = coordinates[pixels[f"{condition}_cluster"] == row.cluster].mean(
                axis=0, dtype=np.float64)
            np.testing.assert_allclose([row.pc1, row.pc2], expected)
    ur.render_pca(portable, dpi=30)
    assert len(calls) == 4  # B1 reuse and re-rendering never fit PCA again.
    path = portable / gu.RESULT_DIRECTORY / "A0/coordinates.csv"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Artifact changed"):
        ur.check_pca_figure(portable)


def test_failed_fit_does_not_publish_partial_output(
    portable: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic PCA failure")

    monkeypatch.setattr(gu, "fit_projection", fail)
    with pytest.raises(RuntimeError, match="synthetic PCA"):
        gu.run_pca(portable)
    assert not (portable / gu.RESULT_DIRECTORY / "B0").exists()
