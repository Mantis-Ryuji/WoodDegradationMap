"""Small CPU fixtures for train-only KMeans, exact center storage and spatial coverage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans

from wood_degradation_map.experiments.baselines import B0Baseline, RepresentationError, fit_pca
from wood_degradation_map.experiments.clustering import (
    FittedClusters, LabelMap, TrainFeatures, collect_train_features, fit_clusters,
)
from wood_degradation_map.experiments.config import kmeans_seed
from wood_degradation_map.experiments.data import FoldData
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput
from wood_degradation_map.experiments.manifests import create_cv_manifest


@pytest.fixture
def fold(tmp_path: Path) -> tuple[FoldData, InputInventory]:
    samples = []
    random = np.random.default_rng(83)
    for index in range(5):
        sample_id = f"KYOw{2700 + index:05d}"
        path = tmp_path / f"{sample_id}.h5"
        spectra = random.standard_normal((37, 256), dtype=np.float32)
        spectra -= spectra.mean(axis=1, keepdims=True)
        spectra /= spectra.std(axis=1, ddof=1, keepdims=True)
        coordinates = np.column_stack((np.arange(37) // 10, np.arange(37) % 10))
        mask = np.zeros((4, 10), dtype=np.uint8)
        mask[coordinates[:, 0], coordinates[:, 1]] = 1
        with h5py.File(path, "w") as handle:
            handle.create_dataset("snv", data=spectra)
            handle.create_dataset("pixel_row_col", data=coordinates.astype(np.int32))
            handle.create_dataset("valid_spectrum_mask", data=mask)
            handle.attrs.update(sample_id=sample_id, saved_pixel_count=37, schema_version=2)
        samples.append(SampleInput(sample_id, path, 4, 10, 37))
    inventory = InputInventory("fixture", tuple(samples), (), 900.0, 2300.0)
    # q=9 gives 36 train rows, including a partial seven-row extraction chunk.
    return FoldData(inventory, create_cv_manifest(inventory, q=9), 1), inventory


@pytest.mark.parametrize("condition", ["B0", "B1"])
def test_collect_and_fit_never_reads_test_or_drops_train_tail(
    fold: tuple[FoldData, InputInventory], condition: str,
) -> None:
    data, inventory = fold
    representation = B0Baseline() if condition == "B0" else fit_pca(data, repeat=1)
    expected = representation.transform(data.train_matrix()).values
    for sample in inventory.samples:
        if sample.sample_id in data.test_sample_ids:
            with h5py.File(sample.path, "r+") as handle:
                handle["snv"][:] = np.nan
    features = collect_train_features(data, representation, condition_id=condition, repeat=1,
                                      chunk_pixels=7)
    assert features.values.shape == (36, 256 if condition == "B0" else 16)
    np.testing.assert_allclose(features.values, expected, atol=1e-6, rtol=1e-6)
    first = fit_clusters(features, 2, device=torch.device("cpu"))
    second = fit_clusters(features, 4, device=torch.device("cpu"))
    assert first.record.train_pixels == second.record.train_pixels == 36
    assert first.record.train_sample_ids == data.train_sample_ids
    assert not set(first.record.train_sample_ids) & set(data.test_sample_ids)
    assert sum(first.record.train_occupancy) == 36


def test_pca_fit_provenance_must_match_shared_train_fold(fold: tuple[FoldData, InputInventory]) -> None:
    data, _ = fold
    pca = fit_pca(data, repeat=1)
    wrong = replace(pca, record=replace(pca.record, sample_ids=("held-out-sample",)))
    with pytest.raises(ValueError, match="shared train"):
        collect_train_features(data, wrong, condition_id="B1", repeat=1)


@pytest.fixture
def features() -> TrainFeatures:
    random = np.random.default_rng(99)
    values = random.standard_normal((40, 256), dtype=np.float32)
    normalized = B0Baseline().transform(values).values
    return TrainFeatures("B0", 1, 1, ("train-a", "train-b"), normalized)


def test_reference_fit_once_and_fixed_centers_survive_save_load(
    features: TrainFeatures, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = CosineKMeans.fit
    calls = 0

    def counted_fit(self: CosineKMeans, values: torch.Tensor, chunk: int | None = None) -> CosineKMeans:
        nonlocal calls
        calls += 1
        assert self.max_iter == 500 and self.tol == 1e-4
        assert self.random_state == kmeans_seed(1, 1, 2) and chunk is None
        assert torch.backends.cuda.matmul.fp32_precision == "ieee"
        return original_fit(self, values, chunk=chunk)

    monkeypatch.setattr(CosineKMeans, "fit", counted_fit)
    outside_rng = torch.get_rng_state().clone()
    fitted = fit_clusters(features, 2, device=torch.device("cpu"))
    assert torch.equal(outside_rng, torch.get_rng_state())
    assert calls == 1
    centers = fitted.centroids
    expected = fitted.predict(features.values)
    reference = CosineKMeans(2, device="cpu", random_state=kmeans_seed(1, 1, 2))
    original_fit(reference, torch.from_numpy(features.values))
    np.testing.assert_array_equal(centers, reference.centroids.numpy())
    np.testing.assert_array_equal(expected, reference.predict(torch.from_numpy(features.values)))
    checkpoint = tmp_path / "centroids.npz"
    fitted.save(checkpoint)
    with pytest.raises(FileExistsError):
        fitted.save(checkpoint)
    restored = FittedClusters.load(checkpoint, condition_id="B0", fold=1, repeat=1, k=2,
                                   device=torch.device("cpu"))
    np.testing.assert_array_equal(restored.centroids, centers)
    np.testing.assert_array_equal(restored.predict(features.values), expected)
    # New test inputs must neither fit again nor modify centers.
    restored.predict(-features.values)
    np.testing.assert_array_equal(restored.centroids, centers)
    assert calls == 1
    assert fitted.record.iterations is None
    assert fitted.record.stop_reason == "not_exposed_by_ChemoMAE_0.2.2"
    for mismatch in ({"repeat": 2}, {"fold": 2}, {"k": 4}, {"condition_id": "B1"}):
        arguments = {"condition_id": "B0", "fold": 1, "repeat": 1, "k": 2, **mismatch}
        with pytest.raises(ValueError, match="mismatch"):
            FittedClusters.load(checkpoint, **arguments, device=torch.device("cpu"))


@pytest.mark.parametrize("invalid", [0.0, 1e-12, np.nan, np.inf, 1e30])
def test_invalid_features_fail_without_discarding_rows(features: TrainFeatures, invalid: float) -> None:
    values = features.values.copy()
    values[3] = invalid
    with pytest.raises(RepresentationError) as error:
        fit_clusters(replace(features, values=values), 2, device=torch.device("cpu"))
    diagnostic = error.value.diagnostics
    assert 3 in (diagnostic.nonfinite_rows + diagnostic.nonfinite_norm_rows
                 + diagnostic.zero_norm_rows + diagnostic.epsilon_clamped_rows)


@pytest.mark.parametrize("invalid_k", [1, 3, 16, 2.0])
def test_unplanned_k_is_rejected(features: TrainFeatures, invalid_k: int) -> None:
    with pytest.raises(ValueError):
        fit_clusters(features, invalid_k, device=torch.device("cpu"))


def test_test_prediction_covers_all_hdf5_rows_and_restores_coordinates(
    fold: tuple[FoldData, InputInventory],
) -> None:
    data, inventory = fold
    representation = B0Baseline()
    train = collect_train_features(data, representation, condition_id="B0", repeat=1)
    fitted = fit_clusters(train, 2, device=torch.device("cpu"))
    sample = next(sample for sample in inventory.samples if sample.sample_id in data.test_sample_ids)
    with h5py.File(sample.path, "r") as handle:
        mask = handle["valid_spectrum_mask"][:]
        coordinates = handle["pixel_row_col"][:]
        expected = fitted.predict(representation.transform(handle["snv"][:]).values)
    output = LabelMap(mask, 2)
    rows = []
    for batch in data.batches("test", chunk_pixels=7):
        rows.extend(batch.hdf5_rows.tolist())
        output.add(batch.pixel_row_col, fitted.predict(representation.transform(batch.snv).values))
    restored = output.finish()
    assert rows == list(range(37))
    assert restored.dtype == np.uint8
    np.testing.assert_array_equal(restored[coordinates[:, 0], coordinates[:, 1]], expected + 1)
    assert np.all(restored[mask == 0] == 0)


def test_label_map_rejects_missing_duplicate_invalid_and_out_of_bounds_pixels() -> None:
    output = LabelMap(np.array([[1, 0], [1, 1]], dtype=np.uint8), 2)
    output.add(np.array([[1, 1]]), np.array([0]))
    with pytest.raises(ValueError, match="Missing predictions for 2"):
        output.finish()
    for coordinates, labels, message in (
        ([[1, 1]], [1], "Duplicate"),
        ([[0, 0], [0, 0]], [0, 1], "Duplicate"),
        ([[0, 1]], [0], "background"),
        ([[2, 0]], [0], "outside"),
        ([[-1, 0]], [0], "outside"),
        ([[0, 0]], [2], "0..K-1"),
        ([[0, 0]], [-1], "0..K-1"),
    ):
        with pytest.raises(ValueError, match=message):
            output.add(np.array(coordinates), np.array(labels))
    output.add(np.array([[0, 0], [1, 0]]), np.array([1, 0]))
    np.testing.assert_array_equal(output.finish(), np.array([[2, 0], [1, 1]], dtype=np.uint8))
