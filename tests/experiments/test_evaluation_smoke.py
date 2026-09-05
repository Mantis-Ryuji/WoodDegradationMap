"""CPU fixtures for isolated synthetic evaluation and saved smoke provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wood_degradation_map.experiments import cluster_pipeline as clustering
from wood_degradation_map.experiments import evaluation_smoke as smoke
from wood_degradation_map.experiments.baselines import NormalizedRepresentation
from wood_degradation_map.experiments.clustering import FittedClusters
from wood_degradation_map.experiments.config import CLUSTER_COUNTS
from wood_degradation_map.experiments.data import FoldData
from wood_degradation_map.experiments.input_validation import InputInventory
from wood_degradation_map.experiments.manifests import _digest, _read_json

# Reuse small real PCA/cluster fixtures and synthetic raw neural checkpoints.
from test_cluster_pipeline import NeuralSource, _save_pca, experiment, neural_source  # noqa: F401

CPU = torch.device("cpu")
Fixture = tuple[Path, FoldData, InputInventory]


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("Synthetic evaluation must not fit or read real spectra")


def _prepare(fixture: Fixture, condition: str = "B0") -> tuple[str, Path]:
    directory, data, inventory = fixture
    if condition == "B1":
        _save_pca(fixture)
    results = clustering.run_clustering(directory, data, inventory, condition, 1,
                                        device=CPU, smoke=True)
    return results.parents[2].name, results


def test_synthetic_input_is_fixed_full_grid_snv_without_global_rng_consumption() -> None:
    before = np.random.get_state()
    left, right = smoke._synthetic_input(), smoke._synthetic_input()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(left.snv, right.snv)
    assert left.snv.shape == (1025, 256) and left.snv.dtype == np.float32
    np.testing.assert_allclose(left.snv.mean(axis=1), 0, atol=1e-7)
    np.testing.assert_allclose(left.snv.std(axis=1, ddof=1), 1, atol=2e-7)
    np.testing.assert_array_equal(left.hdf5_rows, np.arange(1025))
    assert tuple(left.pixel_row_col[-1]) == (40, 24)


@pytest.mark.parametrize("condition", ["B0", "B1"])
def test_saved_sources_reach_all_metrics_and_roundtrip_without_fit_or_real_spectra(
    experiment: Fixture, monkeypatch: pytest.MonkeyPatch, condition: str,
) -> None:
    directory, data, _ = experiment
    identifier, _ = _prepare(experiment, condition)
    original = smoke.load_representation
    calls = []

    def counted(*args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
        representation, source = original(*args, **kwargs)

        class Counted:
            def transform(self, values: np.ndarray) -> NormalizedRepresentation:
                calls.append(len(values))
                return representation.transform(values)

        return Counted(), source

    monkeypatch.setattr(smoke, "load_representation", counted)
    monkeypatch.setattr(FoldData, "batches", _forbidden)
    monkeypatch.setattr(clustering, "fit_clusters", _forbidden)
    before = torch.get_rng_state().clone()
    results = smoke.run_evaluation_smoke(directory, data, condition, 1,
                                         clustering_smoke_id=identifier, device=CPU)
    assert torch.equal(before, torch.get_rng_state())
    assert calls == [1024] * 16 + [1] * 16  # Clean + 15 draws, reused for all seven K.
    completion = _read_json(results / "completion.json")
    assert completion["status"] == "evaluation_smoke_completed" and completion["checks_passed"]
    assert completion["generation_block_sizes"] == [1024, 1]
    assert completion["synthetic_rows"] == 1025
    assert completion["saved_clean_labels_reproduced"] and completion["fixed_centers_unchanged"]
    assert completion["source"]["clustering_smoke_id"] == identifier
    for name, digest in completion["artifact_sha256"].items():
        assert _digest(results / name) == digest
    assert not (directory / "results/evaluation").exists()
    assert not (directory / "results/clustering").exists()
    assert not (results / "failure.json").exists()
    report = _read_json(results / "metrics.json")
    assert set(report["scores"]) == {str(k) for k in CLUSTER_COUNTS}
    with np.load(results / "probe.npz", allow_pickle=False) as saved:
        assert saved["clean_features"].shape == (1025, 256 if condition == "B0" else 16)
        for k in CLUSTER_COUNTS:
            score = report["scores"][str(k)]
            assert score["lfr"]["valid_pixels"] == score["spatial"]["valid_pixels"] == 1025
            assert len(score["lfr"]["draws"]) == 15
            assert score["silhouette"]["valid_pixels"] == 1025
            assert saved[f"labels_k{k}"].shape == (41, 25)
            counts = np.bincount(saved[f"labels_k{k}"].ravel(), minlength=k + 1)[1:].tolist()
            assert counts == score["lfr"]["clean_occupancy"]["counts"]


def test_neural_smoke_source_uses_saved_raw_model_and_preserves_rng(
    neural_source: NeuralSource, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, data, inventory, _ = neural_source
    monkeypatch.setattr(smoke, "experiment_config", clustering.experiment_config)
    source = clustering.run_clustering(directory, data, inventory, "M11", 1,
                                       device=CPU, smoke=True, neural_smoke_id="fixture")
    monkeypatch.setattr(FoldData, "batches", _forbidden)
    monkeypatch.setattr(clustering, "fit_clusters", _forbidden)
    before = torch.get_rng_state().clone()
    results = smoke.run_evaluation_smoke(directory, data, "M11", 1,
                                         clustering_smoke_id=source.parents[2].name, device=CPU)
    assert torch.equal(before, torch.get_rng_state())
    report = _read_json(results / "completion.json")
    assert report["checks_passed"] and report["source"]["neural_smoke_id"] == "fixture"
    assert report["source"]["representation"]["completed_epochs"] == 2


@pytest.mark.parametrize("field,value", [
    ("mode", "clean_test_maps"), ("condition", "B1"), ("fold", 2), ("repeat", 2),
    ("config", {}), ("code_sha256", {}), ("manifest_artifact_sha256", {}),
    ("train_sample_ids", ["held-out"]), ("test_pixels", 999),
])
def test_source_identity_and_recipe_rejected_even_with_updated_file_hash(
    experiment: Fixture, monkeypatch: pytest.MonkeyPatch, field: str, value: object,
) -> None:
    directory, data, _ = experiment
    identifier, source = _prepare(experiment)
    run = _read_json(source / "run.json")
    run[field] = value
    (source / "run.json").write_text(json.dumps(run), encoding="utf-8")
    completion = _read_json(source / "completion.json")
    completion["run_sha256"] = _digest(source / "run.json")
    (source / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    monkeypatch.setattr(smoke, "load_representation", _forbidden)
    with pytest.raises(ValueError, match="mismatch"):
        smoke.run_evaluation_smoke(directory, data, "B0", 1,
                                   clustering_smoke_id=identifier, device=CPU)
    assert not (directory / "results/evaluation_smoke").exists()


@pytest.mark.parametrize("damage", ["completion", "center", "fit", "probe", "failure", "source"])
def test_incomplete_or_modified_source_is_rejected_before_output(
    experiment: Fixture, damage: str,
) -> None:
    directory, data, _ = experiment
    identifier, source = _prepare(experiment)
    completion = _read_json(source / "completion.json")
    if damage == "completion":
        completion["checks_passed"] = False
    elif damage == "center":
        path = directory / f"checkpoints/clustering_smoke/{identifier}/B0/fold_1/repeat_1/centers_k2.npz"
        with path.open("ab") as destination:
            destination.write(b"modified")
    elif damage == "fit":
        fits = _read_json(source / "fits.json")
        fits["2"]["train_pixels"] = 99
        (source / "fits.json").write_text(json.dumps(fits), encoding="utf-8")
        completion["fits_sha256"] = _digest(source / "fits.json")
    elif damage == "probe":
        completion["probe_rows"][0]["sample_id"] = data.test_sample_ids[0]
    elif damage == "failure":
        (source / "failure.json").write_text("{}", encoding="utf-8")
    else:
        run = _read_json(source / "run.json")
        run["source"] = {"kind": "changed"}
        (source / "run.json").write_text(json.dumps(run), encoding="utf-8")
        completion["run_sha256"] = _digest(source / "run.json")
    (source / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        smoke.run_evaluation_smoke(directory, data, "B0", 1,
                                   clustering_smoke_id=identifier, device=CPU)
    assert not (directory / "results/evaluation_smoke").exists()


def test_collapsed_source_labels_remain_undefined_but_gpu_kernel_is_still_probed(
    experiment: Fixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, data, _ = experiment
    identifier, _ = _prepare(experiment)
    monkeypatch.setattr(FittedClusters, "predict", lambda self, values: np.zeros(len(values), dtype=np.int64))
    results = smoke.run_evaluation_smoke(directory, data, "B0", 1,
                                         clustering_smoke_id=identifier, device=CPU)
    metrics = _read_json(results / "metrics.json")
    assert metrics["silhouette_kernel_probe"]["actual_scores"] == [1.0] * 4
    for score in metrics["scores"].values():
        assert score["silhouette"]["macro_mean"] is None
        assert score["silhouette"]["undefined_reason"] == "single_cluster"
        assert score["lfr"]["mean_by_kind"] == {"noise": 0.0, "shift": 0.0, "both": 0.0}


@pytest.mark.parametrize("failure", ["interrupt", "corrupt_save"])
def test_failure_or_interrupt_never_publishes_completion(
    experiment: Fixture, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    directory, data, _ = experiment
    identifier, _ = _prepare(experiment)
    if failure == "interrupt":
        def interrupted(*args: object, **kwargs: object) -> None:
            raise KeyboardInterrupt
        monkeypatch.setattr(smoke, "accumulate_lfr_block", interrupted)
        expected = KeyboardInterrupt
    else:
        original = np.savez_compressed

        def corrupted(destination: object, **arrays: np.ndarray) -> None:
            arrays["clean_features"] = np.zeros_like(arrays["clean_features"])
            original(destination, **arrays)

        monkeypatch.setattr(smoke.np, "savez_compressed", corrupted)
        expected = ValueError
    with pytest.raises(expected):
        smoke.run_evaluation_smoke(directory, data, "B0", 1,
                                   clustering_smoke_id=identifier, device=CPU)
    assert not list((directory / "results/evaluation_smoke").rglob("completion.json"))
    failures = list((directory / "results/evaluation_smoke").rglob("failure.json"))
    assert len(failures) == 1 and _read_json(failures[0])["error_type"] == expected.__name__


def test_existing_output_is_not_overwritten(experiment: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, data, _ = experiment
    identifier, _ = _prepare(experiment)

    class FixedClock:
        @staticmethod
        def now(zone: object) -> object:
            return FixedClock()

        def strftime(self, format_string: str) -> str:
            return "existing"

    monkeypatch.setattr(smoke, "datetime", FixedClock)
    destination = directory / "results/evaluation_smoke/existing/B0/fold_1/repeat_1"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        smoke.run_evaluation_smoke(directory, data, "B0", 1,
                                   clustering_smoke_id=identifier, device=CPU)
    assert marker.read_text(encoding="utf-8") == "original"
