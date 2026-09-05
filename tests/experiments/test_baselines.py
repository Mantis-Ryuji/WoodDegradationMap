from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from sklearn.decomposition import PCA

from wood_degradation_map.experiments.baselines import (
    B0Baseline,
    PCABaseline,
    RepresentationError,
    fit_pca,
    normalize_representation,
)
from wood_degradation_map.experiments.data import FoldData, SpectrumInputError
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput
from wood_degradation_map.experiments.manifests import CVManifest, create_cv_manifest


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[InputInventory, CVManifest]:
    samples = []
    rng = np.random.default_rng(703)
    for index in range(5):
        sample_id = f"KYOw{2700 + index:05d}"
        path = tmp_path / f"{sample_id}.h5"
        coordinates = np.column_stack((np.arange(37) // 19, (np.arange(37) % 19) * 3))
        spectra = rng.standard_normal((37, 256), dtype=np.float32)
        spectra -= spectra.mean(axis=1, keepdims=True)
        spectra /= spectra.std(axis=1, ddof=1, keepdims=True)
        mask = np.zeros((2, 320), dtype=np.uint8)
        mask[coordinates[:, 0], coordinates[:, 1]] = 1
        with h5py.File(path, "w") as handle:
            handle.create_dataset("snv", data=spectra, chunks=(8, 256), compression="gzip")
            handle.create_dataset("pixel_row_col", data=coordinates.astype(np.int32))
            handle.create_dataset("valid_spectrum_mask", data=mask)
            handle.attrs.update(sample_id=sample_id, saved_pixel_count=37, schema_version=2)
        samples.append(SampleInput(sample_id, path, 2, 320, 37))
    inventory = InputInventory("fixture", tuple(samples), (), 900.0, 2300.0)
    return inventory, create_cv_manifest(inventory, q=8)


def test_loader_returns_exact_train_selection_and_all_test_rows(
    inputs: tuple[InputInventory, CVManifest],
) -> None:
    inventory, manifest = inputs
    data = FoldData(inventory, manifest, 1)
    assert not set(data.train_sample_ids) & set(data.test_sample_ids)
    for split in ("train", "test"):
        observed: dict[str, list[int]] = {}
        for batch in data.batches(split, chunk_pixels=7):
            assert 0 < len(batch.snv) <= 7
            assert batch.snv.dtype == np.float32
            observed.setdefault(batch.sample_id, []).extend(batch.hdf5_rows.tolist())
            sample = next(item for item in inventory.samples if item.sample_id == batch.sample_id)
            with h5py.File(sample.path, "r") as handle:
                np.testing.assert_array_equal(batch.snv, handle["snv"][batch.hdf5_rows])
                np.testing.assert_array_equal(batch.pixel_row_col,
                                              handle["pixel_row_col"][batch.hdf5_rows])
        if split == "train":
            for sample_id, rows in observed.items():
                expected = manifest.train_pixels[1].loc[
                    manifest.train_pixels[1]["sample_id"] == sample_id, "hdf5_row",
                ].tolist()
                assert rows == expected
        else:
            assert all(rows == list(range(37)) for rows in observed.values())
    assert data.train_matrix(chunk_pixels=7).shape == (32, 256)
    np.testing.assert_array_equal(data.train_matrix(chunk_pixels=7),
                                  data.train_matrix(chunk_pixels=13))


def test_pca_reads_only_train_rows_and_preserves_numpy_rng(
    inputs: tuple[InputInventory, CVManifest],
) -> None:
    inventory, manifest = inputs
    data = FoldData(inventory, manifest, 1)
    for sample in inventory.samples:
        with h5py.File(sample.path, "r+") as handle:
            if sample.sample_id in data.test_sample_ids:
                handle["snv"][:] = np.nan
            else:
                selected = manifest.train_pixels[1].loc[
                    manifest.train_pixels[1]["sample_id"] == sample.sample_id, "hdf5_row",
                ].to_numpy()
                unused = np.setdiff1d(np.arange(37), selected)
                handle["snv"][unused] = np.full((len(unused), 256), np.nan, dtype=np.float32)
    train = data.train_matrix()
    before = np.random.get_state()
    fitted = fit_pca(data, repeat=1)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]
    assert fitted.record.sample_ids == data.train_sample_ids
    assert fitted.record.train_pixel_count == 32
    np.testing.assert_allclose(fitted.estimator.mean_, train.mean(axis=0), atol=1e-7)
    assert fitted.estimator.random_state is None
    assert fitted.estimator.whiten is False


def test_spectrum_io_stays_within_requested_chunk_size(
    inputs: tuple[InputInventory, CVManifest], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_getitem = h5py.Dataset.__getitem__
    read_sizes: list[int] = []

    def bounded_getitem(dataset: h5py.Dataset, key: object) -> object:
        if dataset.name == "/snv":
            assert isinstance(key, slice) and key.start is not None and key.stop is not None
            assert 0 < key.stop - key.start <= 7
            read_sizes.append(key.stop - key.start)
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded_getitem)
    data = FoldData(*inputs, 1)
    data.train_matrix(chunk_pixels=7)
    list(data.batches("test", chunk_pixels=7))
    assert read_sizes


def test_loader_copies_manifest_selections(inputs: tuple[InputInventory, CVManifest]) -> None:
    inventory, manifest = inputs
    data = FoldData(inventory, manifest, 1)
    before = data.train_matrix()
    manifest.train_pixels[1].loc[:, "hdf5_row"] = 0
    np.testing.assert_array_equal(data.train_matrix(), before)


def test_rng_restored_on_fit_failure(
    inputs: tuple[InputInventory, CVManifest], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fit(self: PCA, spectra: np.ndarray) -> None:
        np.random.random(10)
        raise ValueError("synthetic fit failure")

    monkeypatch.setattr(PCA, "fit", fail_fit)
    before = np.random.get_state()
    with pytest.raises(ValueError, match="synthetic"):
        fit_pca(FoldData(*inputs, 1), repeat=1)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]


def test_baseline_shapes_normalization_and_roundtrip(
    inputs: tuple[InputInventory, CVManifest], tmp_path: Path,
) -> None:
    data = FoldData(*inputs, 1)
    spectra = data.train_matrix()
    before = spectra.copy()
    b0 = B0Baseline()
    b0.save(tmp_path / "b0.json")
    expected_b0 = spectra / np.linalg.norm(spectra, axis=1, keepdims=True)
    np.testing.assert_allclose(B0Baseline.load(tmp_path / "b0.json").transform(spectra).values,
                               expected_b0, atol=1e-6)
    pca = fit_pca(data, repeat=1)
    expected_pca = pca.estimator.transform(spectra)
    expected_pca /= np.linalg.norm(expected_pca, axis=1, keepdims=True)
    result = pca.transform(spectra)
    assert result.values.shape == (32, 16) and result.values.dtype == np.float32
    assert result.diagnostics.unit_norm_absolute_error_max < 1e-6
    np.testing.assert_allclose(result.values, expected_pca, atol=1e-6)
    pca.save(tmp_path / "pca.npz")
    restored = PCABaseline.load(tmp_path / "pca.npz", fold=1, repeat=1)
    np.testing.assert_allclose(restored.transform(spectra).values, result.values, atol=1e-6)
    np.testing.assert_array_equal(spectra, before)
    assert restored.record == pca.record
    assert restored.reusable_across_repeats
    PCABaseline.load(tmp_path / "pca.npz", fold=1, repeat=2)
    with pytest.raises(ValueError, match="fold or seed"):
        PCABaseline.load(tmp_path / "pca.npz", fold=2, repeat=1)
    with pytest.raises(FileExistsError):
        pca.save(tmp_path / "pca.npz")
    with pytest.raises(FileExistsError):
        b0.save(tmp_path / "b0.json")


def test_transform_never_refits_or_updates_train_mean(
    inputs: tuple[InputInventory, CVManifest], monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = FoldData(*inputs, 1)
    pca = fit_pca(data, repeat=1)
    mean = pca.estimator.mean_.copy()

    def forbid_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("transform must not fit")

    monkeypatch.setattr(PCA, "fit", forbid_fit)
    for batch in data.batches("test", chunk_pixels=11):
        transformed = pca.transform(batch.snv + np.float32(3))
        assert transformed.values.shape == (len(batch.snv), 16)
    np.testing.assert_array_equal(pca.estimator.mean_, mean)


def test_stochastic_checkpoint_cannot_cross_repeats(
    inputs: tuple[InputInventory, CVManifest], tmp_path: Path,
) -> None:
    pca = fit_pca(FoldData(*inputs, 1), repeat=1)
    pca = replace(pca, record=replace(pca.record, solver="randomized"))
    pca.save(tmp_path / "stochastic.npz")
    with pytest.raises(ValueError, match="different repeat"):
        PCABaseline.load(tmp_path / "stochastic.npz", fold=1, repeat=2)


@pytest.mark.parametrize("value, field", [
    (np.nan, "nonfinite_rows"), (np.inf, "nonfinite_rows"), (0.0, "zero_norm_rows"),
    (1e-10, "epsilon_clamped_rows"), (1e38, "nonfinite_norm_rows"),
])
def test_invalid_cosine_rows_are_reported(value: float, field: str) -> None:
    values = np.ones((3, 16), dtype=np.float32)
    values[1] = value
    with pytest.raises(RepresentationError) as error:
        normalize_representation(values)
    assert getattr(error.value.diagnostics, field) == (1,)
    assert error.value.diagnostics.row_count == 3


@pytest.mark.parametrize("problem", ["nonfinite", "zero", "mask", "coordinate"])
def test_loader_reports_corrupt_selected_rows(
    inputs: tuple[InputInventory, CVManifest], problem: str,
) -> None:
    inventory, manifest = inputs
    row = manifest.train_pixels[1].iloc[0]
    sample = next(item for item in inventory.samples if item.sample_id == row.sample_id)
    with h5py.File(sample.path, "r+") as handle:
        if problem == "nonfinite":
            handle["snv"][row.hdf5_row, 0] = np.nan
        elif problem == "zero":
            handle["snv"][row.hdf5_row] = 0
        elif problem == "mask":
            handle["valid_spectrum_mask"][row.pixel_row, row.pixel_col] = 0
        else:
            handle["pixel_row_col"][row.hdf5_row] = [-1, 0]
    with pytest.raises(SpectrumInputError) as error:
        FoldData(inventory, manifest, 1).train_matrix()
    assert error.value.sample_id == row.sample_id
    assert int(row.hdf5_row) in error.value.hdf5_rows


@pytest.mark.parametrize("values", [
    np.ones((2, 256), dtype=np.float64), np.empty((0, 256), dtype=np.float32),
    np.ones((2, 16), dtype=np.float32),
])
def test_b0_rejects_wrong_input_contract(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        B0Baseline().transform(values)
