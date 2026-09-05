"""Small CPU fixtures for source validation, K reuse and complete clean map persistence."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from chemomae.models.chemo_mae import ChemoMAE

from wood_degradation_map.experiments import cluster_pipeline as pipeline
from wood_degradation_map.experiments.baselines import NormalizedRepresentation, fit_pca
from wood_degradation_map.experiments.clustering import FittedClusters, TrainFeatures
from wood_degradation_map.experiments.config import CLUSTER_COUNTS, experiment_config, run_seed
from wood_degradation_map.experiments.data import FoldData, SpectrumInputError
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput
from wood_degradation_map.experiments.manifests import _digest, _read_json, _write_json, create_cv_manifest
from wood_degradation_map.experiments.neural import TorchRandomStream, extract_full_visible
from wood_degradation_map.experiments.training import _code_hashes, runtime_record

CPU = torch.device("cpu")
NeuralSource = tuple[Path, FoldData, InputInventory, ChemoMAE]


@pytest.fixture
def experiment(tmp_path: Path) -> tuple[Path, FoldData, InputInventory]:
    samples = []
    random = np.random.default_rng(413)
    coordinates = np.column_stack((np.arange(37) // 10, np.arange(37) % 10)).astype(np.int32)
    mask = np.zeros((4, 10), dtype=np.uint8)
    mask[coordinates[:, 0], coordinates[:, 1]] = 1
    for index in range(5):
        sample_id = f"KYOw{2800 + index:05d}"
        path = tmp_path / f"{sample_id}.h5"
        spectra = random.standard_normal((37, 256), dtype=np.float32)
        spectra -= spectra.mean(axis=1, keepdims=True)
        spectra /= spectra.std(axis=1, ddof=1, keepdims=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("snv", data=spectra)
            handle.create_dataset("pixel_row_col", data=coordinates)
            handle.create_dataset("valid_spectrum_mask", data=mask)
            handle.attrs.update(sample_id=sample_id, saved_pixel_count=37, schema_version=2)
        samples.append(SampleInput(sample_id, path, 4, 10, 37))
    inventory = InputInventory("fixture", tuple(samples), (), 900.0, 2300.0)
    data = FoldData(inventory, create_cv_manifest(inventory, q=9), 1)
    directory = tmp_path / "experiment"
    (directory / "manifests").mkdir(parents=True)
    _write_json(directory / "manifests/complete.json", {"artifact_sha256": {"fixture": "fixed"}})
    return directory, data, inventory


def _save_pca(experiment: tuple[Path, FoldData, InputInventory]) -> None:
    directory, data, _ = experiment
    suffix = f"baselines/fold_{data.fold}/repeat_1"
    checkpoints, results = directory / "checkpoints" / suffix, directory / "results" / suffix
    checkpoints.mkdir(parents=True)
    results.mkdir(parents=True)
    pca = fit_pca(data, repeat=1)
    pca.save(checkpoints / "pca.npz")
    _write_json(results / "fit.json", {
        "status": "fitted_and_roundtrip_checked", "manifest_artifact_sha256": {"fixture": "fixed"},
        "pca_checkpoint_sha256": _digest(checkpoints / "pca.npz"),
    })


@pytest.mark.parametrize("condition", ["B0", "B1"])
def test_complete_maps_reuse_features_across_all_k_and_check_without_fit(
    experiment: tuple[Path, FoldData, InputInventory], monkeypatch: pytest.MonkeyPatch, condition: str,
) -> None:
    directory, data, inventory = experiment
    if condition == "B1":
        _save_pca(experiment)
    representation, source = pipeline.load_representation(directory, data, condition, 1, device=CPU)
    calls = []

    class CountedRepresentation:
        def transform(self, spectra: np.ndarray) -> NormalizedRepresentation:
            calls.append(spectra.copy())
            return representation.transform(spectra)

    original_load = pipeline.load_representation
    monkeypatch.setattr(pipeline, "load_representation", lambda *args, **kwargs:
                        (CountedRepresentation(), source))
    original_fit = pipeline.fit_clusters
    feature_ids, fitted_ks = [], []

    def fit(features: TrainFeatures, k: int, *, device: torch.device) -> FittedClusters:
        feature_ids.append(id(features.values))
        fitted_ks.append(k)
        assert features.sample_ids == data.train_sample_ids
        assert not set(features.sample_ids) & set(data.test_sample_ids)
        assert len(features.values) == 36
        return original_fit(features, k, device=device)

    monkeypatch.setattr(pipeline, "fit_clusters", fit)
    results = pipeline.run_clustering(directory, data, inventory, condition, 1,
                                      device=CPU, chunk_pixels=7)
    assert fitted_ks == list(CLUSTER_COUNTS) and len(set(feature_ids)) == 1
    assert sum(len(value) for value in calls) == 36 + 37  # No K-fold duplication or tail drop.
    assert len(calls) == sum(1 for _ in data.batches("train", chunk_pixels=7)) + 6
    report = _read_json(results / "completion.json")
    assert report["status"] == "clean_test_maps_completed"
    assert report["samples"][0]["pixels"] == 37
    sample = next(s for s in inventory.samples if s.sample_id in data.test_sample_ids)
    with h5py.File(sample.path, "r") as handle, np.load(
        results / "maps" / f"{sample.sample_id}.npz", allow_pickle=False,
    ) as saved:
        coordinates = handle["pixel_row_col"][:]
        values = representation.transform(handle["snv"][:]).values
        for k in CLUSTER_COUNTS:
            cluster = pipeline.FittedClusters.load(
                directory / f"checkpoints/clustering/{condition}/fold_1/repeat_1/centers_k{k}.npz",
                condition_id=condition, fold=1, repeat=1, k=k, device=CPU,
            )
            np.testing.assert_array_equal(saved[f"labels_k{k}"][coordinates[:, 0], coordinates[:, 1]],
                                          cluster.predict(values) + 1)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("check must not fit, transform or read spectra")

    monkeypatch.setattr(pipeline, "load_representation", original_load)
    monkeypatch.setattr(pipeline, "fit_clusters", forbidden)
    monkeypatch.setattr(FoldData, "batches", forbidden)
    result = pipeline.check_clustering(directory, data, inventory, condition, 1, device=CPU)
    assert result["status"] == "validated_existing_clustering"
    assert result["test_pixels"] == 37
    with pytest.raises(FileExistsError):
        pipeline.run_clustering(directory, data, inventory, condition, 1, device=CPU)
    with (results / "maps" / f"{sample.sample_id}.npz").open("ab") as destination:
        destination.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        pipeline.check_clustering(directory, data, inventory, condition, 1, device=CPU)


def test_smoke_never_reads_test_and_does_not_claim_complete_maps(
    experiment: tuple[Path, FoldData, InputInventory],
) -> None:
    directory, data, inventory = experiment
    for sample in inventory.samples:
        if sample.sample_id in data.test_sample_ids:
            with h5py.File(sample.path, "r+") as handle:
                handle["snv"][:] = np.nan
    results = pipeline.run_clustering(directory, data, inventory, "B0", 1, device=CPU, smoke=True)
    report = _read_json(results / "completion.json")
    assert report["status"] == "clustering_smoke_completed"
    assert report["samples"] == [] and not (results / "maps").exists()
    assert all(row["sample_id"] in data.train_sample_ids for row in report["probe_rows"])
    assert sum(len(row["hdf5_rows"]) for row in report["probe_rows"]) == 36
    assert not (directory / "results/clustering").exists()


def test_failure_preserves_source_rows_and_cannot_be_checked_as_complete(
    experiment: tuple[Path, FoldData, InputInventory],
) -> None:
    directory, data, inventory = experiment
    sample = next(s for s in inventory.samples if s.sample_id in data.test_sample_ids)
    with h5py.File(sample.path, "r+") as handle:
        handle["snv"][36] = np.nan
    with pytest.raises(SpectrumInputError):
        pipeline.run_clustering(directory, data, inventory, "B0", 1, device=CPU, chunk_pixels=7)
    results = directory / "results/clustering/B0/fold_1/repeat_1"
    assert not (results / "completion.json").exists()
    failure = _read_json(results / "failure.json")
    assert failure["sample_id"] == sample.sample_id and failure["hdf5_rows"] == [36]
    with pytest.raises(FileNotFoundError):
        pipeline.check_clustering(directory, data, inventory, "B0", 1, device=CPU)


def test_pca_source_is_explicit_and_hash_bound_to_manifest(
    experiment: tuple[Path, FoldData, InputInventory],
) -> None:
    directory, data, _ = experiment
    _save_pca(experiment)
    with pytest.raises(FileNotFoundError):
        pipeline.load_representation(directory, data, "B1", 2, device=CPU)
    pca, source = pipeline.load_representation(directory, data, "B1", 2, device=CPU, pca_repeat=1)
    assert pca.reusable_across_repeats and source["source_repeat"] == 1
    report_path = directory / "results/baselines/fold_1/repeat_1/fit.json"
    report = _read_json(report_path)
    report["manifest_artifact_sha256"] = {"fixture": "changed"}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        pipeline.load_representation(directory, data, "B1", 1, device=CPU)


@pytest.fixture
def neural_source(
    experiment: tuple[Path, FoldData, InputInventory], monkeypatch: pytest.MonkeyPatch,
) -> NeuralSource:
    directory, data, inventory = experiment

    def fixture_config() -> dict[str, object]:
        config = experiment_config()
        config["training"]["batch_size"] = 4
        return config

    def tiny_model(condition: str, fold: int, repeat: int) -> ChemoMAE:
        with TorchRandomStream(run_seed("model_init", fold, repeat), CPU).scope():
            return ChemoMAE(seq_len=256, n_patches=16, d_model=8, nhead=2, num_layers=1,
                            dim_feedforward=16, dropout=0.0, latent_dim=16,
                            latent_normalize=True, decoder_num_layers=1, n_mask=8)

    monkeypatch.setattr(pipeline, "experiment_config", fixture_config)
    monkeypatch.setattr(pipeline, "build_model", tiny_model)
    model = tiny_model("M11", 1, 1)
    # Synthetic source records exercise the loader's completion gates. They do
    # not represent trained weights or evidence of an actual 800-epoch run.
    for smoke in (False, True):
        branch = "neural_smoke/fixture" if smoke else "neural"
        suffix = f"{branch}/M11/fold_1/repeat_1"
        results, weights_dir = directory / "results" / suffix, directory / "checkpoints" / suffix
        results.mkdir(parents=True)
        weights_dir.mkdir(parents=True)
        weights = weights_dir / ("smoke_model.pt" if smoke else "last_model.pt")
        torch.save(model.state_dict(), weights)
        steps = data.train_pixel_count // 4
        updates = 4 if smoke else steps * 800
        _write_json(results / "run.json", {
            "schema_version": 1, "mode": "smoke" if smoke else "training",
            "smoke_id": "fixture" if smoke else None, "smoke_batches_per_epoch": 2 if smoke else None,
            "condition": "M11", "fold": 1, "repeat": 1, "config": fixture_config(),
            "train_sample_ids": list(data.train_sample_ids), "train_pixels": data.train_pixel_count,
            "batch_size": 4, "steps_per_full_epoch": steps, "planned_production_updates": steps * 800,
            "runtime": runtime_record(CPU), "code_sha256": _code_hashes(),
            "manifest_artifact_sha256": {"fixture": "fixed"},
            "seeds": {purpose: run_seed(purpose, 1, 1)
                      for purpose in ("model_init", "pixel_order", "mask", "train_aug")},
        })
        _write_json(results / "completion.json", {
            "status": "smoke_fit_completed" if smoke else "training_completed",
            "completed_epochs": 2 if smoke else 800, "attempted_updates": updates,
            "optimizer_updates": updates, "nonzero_lr_updates": updates - 1, "amp_skips": 0,
            "weights_sha256": _digest(weights), "weights_file": str(weights),
        })
    return directory, data, inventory, model


def test_neural_raw_load_is_exact_and_all_visible_without_rng_consumption(
    neural_source: NeuralSource,
) -> None:
    directory, data, _, model = neural_source
    before = torch.get_rng_state().clone()
    representation, source = pipeline.load_representation(directory, data, "M11", 1, device=CPU)
    assert torch.equal(before, torch.get_rng_state())
    for name, value in model.state_dict().items():
        assert torch.equal(value, representation.model.state_dict()[name])
    clean = data.train_matrix()
    original = clean.copy()
    expected = extract_full_visible(model, torch.from_numpy(clean)).values
    actual = representation.transform(clean).values
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(clean, original)
    assert torch.equal(before, torch.get_rng_state())
    assert source["completed_epochs"] == 800 and not representation.model.training


@pytest.mark.parametrize("field,value", [
    ("condition", "A0"), ("fold", 2), ("repeat", 2),
    ("train_sample_ids", ["held-out"]), ("manifest_artifact_sha256", {}),
    ("code_sha256", {}), ("config", {}),
])
def test_neural_rejects_wrong_run_before_model_loading(
    neural_source: NeuralSource, field: str, value: object,
) -> None:
    directory, data, _, _ = neural_source
    path = directory / "results/neural/M11/fold_1/repeat_1/run.json"
    run = _read_json(path)
    run[field] = value
    path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        pipeline.load_representation(directory, data, "M11", 1, device=CPU)


@pytest.mark.parametrize("field,value", [("completed_epochs", 799), ("weights_sha256", "changed")])
def test_neural_rejects_incomplete_or_changed_weights(
    neural_source: NeuralSource, field: str, value: object,
) -> None:
    directory, data, _, _ = neural_source
    path = directory / "results/neural/M11/fold_1/repeat_1/completion.json"
    completion = _read_json(path)
    completion[field] = value
    path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline.load_representation(directory, data, "M11", 1, device=CPU)


def test_neural_smoke_is_explicit_and_cannot_enter_production_maps(
    neural_source: NeuralSource,
) -> None:
    directory, data, inventory, _ = neural_source
    representation, source = pipeline.load_representation(
        directory, data, "M11", 1, device=CPU, neural_smoke_id="fixture",
    )
    assert source["mode"] == "smoke" and source["completed_epochs"] == 2
    assert representation.transform(data.train_matrix()).values.shape == (36, 16)
    with pytest.raises(ValueError, match="production"):
        pipeline.run_clustering(directory, data, inventory, "M11", 1, device=CPU,
                                neural_smoke_id="fixture")
    results = pipeline.run_clustering(directory, data, inventory, "M11", 1, device=CPU,
                                      smoke=True, neural_smoke_id="fixture")
    assert _read_json(results / "completion.json")["checks_passed"]
    assert not (results / "maps").exists()


def test_neural_pipeline_reaches_full_test_maps_with_saved_raw_weights(
    neural_source: NeuralSource,
) -> None:
    directory, data, inventory, _ = neural_source
    results = pipeline.run_clustering(directory, data, inventory, "M11", 1, device=CPU,
                                      chunk_pixels=7)
    assert _read_json(results / "completion.json")["samples"][0]["pixels"] == 37
    assert pipeline.check_clustering(directory, data, inventory, "M11", 1, device=CPU)[
        "status"] == "validated_existing_clustering"


@pytest.mark.parametrize("invalid", ["half", "nan"])
def test_neural_rejects_invalid_raw_state_even_with_matching_hash(
    neural_source: NeuralSource, invalid: str,
) -> None:
    directory, data, _, model = neural_source
    weights = directory / "checkpoints/neural/M11/fold_1/repeat_1/last_model.pt"
    state = {key: value.clone() for key, value in model.state_dict().items()}
    key = next(iter(state))
    if invalid == "half":
        state[key] = state[key].half()
    else:
        state[key].fill_(float("nan"))
    torch.save(state, weights)
    path = directory / "results/neural/M11/fold_1/repeat_1/completion.json"
    completion = _read_json(path)
    completion["weights_sha256"] = _digest(weights)
    path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="raw FP32"):
        pipeline.load_representation(directory, data, "M11", 1, device=CPU)


def test_interrupt_does_not_publish_completion(
    experiment: tuple[Path, FoldData, InputInventory], monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, data, inventory = experiment

    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "collect_train_features", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_clustering(directory, data, inventory, "B0", 1, device=CPU)
    results = directory / "results/clustering/B0/fold_1/repeat_1"
    assert not (results / "completion.json").exists()
    assert _read_json(results / "failure.json")["error_type"] == "KeyboardInterrupt"
