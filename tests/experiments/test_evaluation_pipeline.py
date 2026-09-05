"""Small CPU integration of persisted maps, shared LFR inputs and pooled metrics."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from wood_degradation_map.experiments import cluster_pipeline as clustering
from wood_degradation_map.experiments import evaluation_pipeline as evaluation
from wood_degradation_map.experiments.aggregation import REPEATED_METRICS, ScoreRecord
from wood_degradation_map.experiments.baselines import NormalizedRepresentation, fit_pca
from wood_degradation_map.experiments.config import CLUSTER_COUNTS
from wood_degradation_map.experiments.clustering import FittedClusters, Representation
from wood_degradation_map.experiments.data import FoldData, SpectrumBatch
from wood_degradation_map.experiments.diagnostic_metrics import FoldSilhouette, SampleFeatures
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput
from wood_degradation_map.experiments.manifests import _digest, _read_json, _write_json, create_cv_manifest
from wood_degradation_map.experiments.lfr import LFRAccumulator, LFRBlockPrediction
from wood_degradation_map.experiments.perturbations import DRAW_KEYS, SharedPerturbationBlock
from wood_degradation_map.experiments.spatial_metrics import local_label_agreement

# Reuse the existing synthetic raw-weight provenance fixture with this module's
# two-test-sample inventory. Its 800-epoch metadata is a test fixture, not training.
from test_cluster_pipeline import NeuralSource, neural_source  # noqa: F401

CPU = torch.device("cpu")
Fixture = tuple[Path, FoldData, InputInventory]


@pytest.fixture
def experiment(tmp_path: Path) -> Fixture:
    samples = []
    random = np.random.default_rng(932)
    # Ten samples ensure two test samples: silhouette must pool them, not run per sample.
    for index in range(10):
        sample_id = f"KYOw{2800 + index:05d}"
        n = 35 + index % 3
        coordinates = np.column_stack((np.arange(n) // 10, np.arange(n) % 10)).astype(np.int32)
        mask = np.zeros((4, 10), dtype=np.uint8)
        mask[coordinates[:, 0], coordinates[:, 1]] = 1
        spectra = random.standard_normal((n, 256), dtype=np.float32)
        spectra -= spectra.mean(axis=1, keepdims=True)
        spectra /= spectra.std(axis=1, ddof=1, keepdims=True)
        path = tmp_path / f"{sample_id}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("snv", data=spectra)
            handle.create_dataset("pixel_row_col", data=coordinates)
            handle.create_dataset("valid_spectrum_mask", data=mask)
            handle.attrs.update(sample_id=sample_id, saved_pixel_count=n, schema_version=2)
        samples.append(SampleInput(sample_id, path, 4, 10, n))
    inventory = InputInventory("fixture", tuple(samples), (), 900.0, 2300.0)
    data = FoldData(inventory, create_cv_manifest(inventory, q=9), 1)
    directory = tmp_path / "experiment"
    (directory / "manifests").mkdir(parents=True)
    _write_json(directory / "manifests/complete.json", {"artifact_sha256": {"fixture": "fixed"}})
    return directory, data, inventory


def _cluster(experiment: Fixture, conditions: tuple[str, ...] = ("B0",),
             repeats: tuple[int, ...] = (1,)) -> None:
    directory, data, inventory = experiment
    if "B1" in conditions:
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
    for condition in conditions:
        for repeat in repeats:
            clustering.run_clustering(directory, data, inventory, condition, repeat, device=CPU,
                                      pca_repeat=1 if condition == "B1" else None, chunk_pixels=7)


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("Evaluation/check must not fit, train or recompute checked metrics")


def test_shared_blocks_all_k_saved_scores_and_read_only_check(
    experiment: Fixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, data, inventory = experiment
    conditions, repeats = ("B0", "B1"), (1, 2)
    _cluster(experiment, conditions, repeats)
    original_batches = FoldData.batches
    roles = []

    def test_only(self: FoldData, role: str, **kwargs: object) -> Iterator[SpectrumBatch]:
        roles.append(role)
        assert role == "test"
        yield from original_batches(self, role, **kwargs)

    monkeypatch.setattr(FoldData, "batches", test_only)
    monkeypatch.setattr(clustering, "fit_clusters", _forbidden)
    original_load = evaluation.load_representation
    extracted = Counter()

    def counted_load(*args: object, **kwargs: object) -> tuple[Representation, dict[str, object]]:
        representation, source = original_load(*args, **kwargs)
        key = args[2], args[3]

        class Counted:
            def transform(self, values: np.ndarray) -> NormalizedRepresentation:
                extracted[key] += len(values)
                return representation.transform(values)

        return Counted(), source

    monkeypatch.setattr(evaluation, "load_representation", counted_load)
    original_accumulate = evaluation.accumulate_lfr_block
    shared = {}

    def track_block(
        block: SharedPerturbationBlock, representation: Representation,
        clusters: Mapping[int, FittedClusters], accumulators: Mapping[int, LFRAccumulator],
    ) -> LFRBlockPrediction:
        key = block.clean.sample_id, int(block.clean.hdf5_rows[0])
        if key in shared:
            assert shared[key] is block
        else:
            shared[key] = block
        return original_accumulate(block, representation, clusters, accumulators)

    monkeypatch.setattr(evaluation, "accumulate_lfr_block", track_block)
    original_silhouette = evaluation.fold_silhouette
    silhouette_calls = []

    def pooled(samples: tuple[SampleFeatures, ...], **kwargs: object) -> FoldSilhouette:
        assert tuple(item.pixels.sample_id for item in samples) == data.test_sample_ids
        assert sum(len(item.values) for item in samples) == data.test_pixel_count
        silhouette_calls.append((kwargs["condition_id"], kwargs["repeat"], kwargs["k"]))
        return original_silhouette(samples, **kwargs)

    monkeypatch.setattr(evaluation, "fold_silhouette", pooled)
    rng = torch.get_rng_state().clone()
    paths = evaluation.run_evaluation(directory, data, inventory, conditions, repeats,
                                     device=CPU, chunk_pixels=7, silhouette_chunk_pixels=11)
    assert torch.equal(torch.get_rng_state(), rng)
    assert roles == ["test"]
    assert len(silhouette_calls) == 4 * len(CLUSTER_COUNTS)
    assert extracted == Counter({(c, r): 16 * data.test_pixel_count for c in conditions for r in repeats})
    assert len({_digest(path / "shared_inputs.json") for path in paths}) == 1
    for path in paths:
        run = _read_json(path / "run.json")
        rows = [ScoreRecord(**item) for item in _read_json(path / "scores.json")["records"]]
        assert len(rows) == len(data.test_sample_ids) * len(CLUSTER_COUNTS) * len(REPEATED_METRICS)
        for sample_id in data.test_sample_ids:
            detail = _read_json(path / "samples" / f"{sample_id}.json")
            for k in CLUSTER_COUNTS:
                lfr = detail["lfr"][str(k)]
                assert {(item["kind"], item["draw"]) for item in lfr["draws"]} == set(DRAW_KEYS)
                assert sum(lfr["clean_occupancy"]["counts"]) == detail["valid_pixels"]
                for item in lfr["draws"]:
                    assert item["rate"] == float(np.float32(item["flipped_pixels"]) / np.float32(detail["valid_pixels"]))
                sample = next(item for item in inventory.samples if item.sample_id == sample_id)
                _, maps, _ = evaluation._paths(directory, run["condition"], data.fold, run["repeat"])
                with h5py.File(sample.path, "r") as handle, np.load(maps / "maps" / f"{sample_id}.npz") as saved:
                    spatial = local_label_agreement(saved[f"labels_k{k}"], handle["valid_spectrum_mask"][:], k=k)
                assert detail["spatial"][str(k)]["windows"][0]["lla"] == spatial.windows[0].lla
    monkeypatch.setattr(FoldData, "batches", _forbidden)
    monkeypatch.setattr(evaluation, "fold_silhouette", _forbidden)
    monkeypatch.setattr(evaluation, "accumulate_lfr_block", _forbidden)
    for condition in conditions:
        for repeat in repeats:
            report = evaluation.check_evaluation(directory, data, inventory, condition, repeat)
            assert report["status"] == "validated_existing_evaluation"
    assert len(evaluation.check_evaluations(directory, data, inventory, conditions, repeats)) == 4
    with pytest.raises(FileExistsError):
        evaluation.run_evaluation(directory, data, inventory, conditions, repeats, device=CPU)


@pytest.mark.parametrize("failure", ["source", "center", "map", "smoke"])
def test_invalid_or_smoke_source_rejected_before_output(experiment: Fixture, failure: str) -> None:
    directory, data, inventory = experiment
    if failure == "smoke":
        clustering.run_clustering(directory, data, inventory, "B0", 1, device=CPU, smoke=True)
    else:
        _cluster(experiment)
        _, maps, checkpoints = evaluation._paths(directory, "B0", 1, 1)
        path = (maps / "completion.json" if failure == "source" else
                checkpoints / "centers_k2.npz" if failure == "center" else
                maps / "maps" / f"{data.test_sample_ids[0]}.npz")
        if failure == "source":
            record = _read_json(path)
            record["status"] = "clustering_smoke_completed"
            path.write_text(json.dumps(record), encoding="utf-8")
        else:
            with path.open("ab") as handle:
                handle.write(b"changed")
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.run_evaluation(directory, data, inventory, ("B0",), (1,), device=CPU)
    assert not (directory / "results/evaluation").exists()


@pytest.mark.parametrize("failure", ["nan", "prediction", "interrupt", "missing_sample"])
def test_incomplete_evaluation_preserves_failure_and_never_publishes_completion(
    experiment: Fixture, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    directory, data, inventory = experiment
    _cluster(experiment)
    if failure == "nan":
        sample = next(item for item in inventory.samples if item.sample_id == data.test_sample_ids[0])
        with h5py.File(sample.path, "r+") as handle:
            handle["snv"][0] = np.nan
    elif failure == "prediction":
        original = evaluation.accumulate_lfr_block

        def changed(*args: object, **kwargs: object) -> LFRBlockPrediction:
            result = original(*args, **kwargs)
            result.clean[2].labels[0] = 3 - result.clean[2].labels[0]
            return result

        monkeypatch.setattr(evaluation, "accumulate_lfr_block", changed)
    elif failure == "interrupt":
        def interrupt(*args: object, **kwargs: object) -> None:
            raise KeyboardInterrupt()

        monkeypatch.setattr(evaluation, "accumulate_lfr_block", interrupt)
    else:
        original = FoldData.batches

        def missing(self: FoldData, role: str, **kwargs: object) -> Iterator[SpectrumBatch]:
            for batch in original(self, role, **kwargs):
                if batch.sample_id != data.test_sample_ids[-1]:
                    yield batch

        monkeypatch.setattr(FoldData, "batches", missing)
    error = KeyboardInterrupt if failure == "interrupt" else ValueError
    with pytest.raises(error):
        evaluation.run_evaluation(directory, data, inventory, ("B0",), (1,), device=CPU, chunk_pixels=7)
    path, _, _ = evaluation._paths(directory, "B0", 1, 1)
    assert not (path / "completion.json").exists()
    record = _read_json(path / "failure.json")
    assert record["status"] == "failed_or_interrupted"
    if failure in ("nan", "prediction"):
        assert record["sample_id"] == data.test_sample_ids[0]
        assert record["hdf5_rows"] == [0]
    with pytest.raises(FileNotFoundError):
        evaluation.check_evaluation(directory, data, inventory, "B0", 1)


@pytest.mark.parametrize("artifact", ["scores", "shared_inputs", "sample", "silhouette", "pixels"])
def test_changed_saved_results_are_rejected(experiment: Fixture, artifact: str) -> None:
    directory, data, inventory = experiment
    _cluster(experiment)
    path, = evaluation.run_evaluation(directory, data, inventory, ("B0",), (1,), device=CPU)
    target = {"scores": path / "scores.json", "shared_inputs": path / "shared_inputs.json",
              "sample": path / "samples" / f"{data.test_sample_ids[0]}.json",
              "silhouette": path / "silhouette/k2.json", "pixels": path / "silhouette/pixels_k2.npy"}[artifact]
    assert target.exists()
    with target.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="hash"):
        evaluation.check_evaluation(directory, data, inventory, "B0", 1)


@pytest.mark.parametrize("conditions,repeats", [((), (1,)), (("B0", "B0"), (1,)),
                                               (("B0",), (1, 1)), (("B0",), (True,)),
                                               (("unknown",), (1,))])
def test_invalid_selection_fails_before_io(
    experiment: Fixture, conditions: tuple[str, ...], repeats: tuple[int, ...],
) -> None:
    directory, data, inventory = experiment
    with pytest.raises(ValueError):
        evaluation.run_evaluation(directory, data, inventory, conditions, repeats, device=CPU)
    assert not (directory / "results/evaluation").exists()


def test_neural_saved_weights_reach_shared_evaluation_without_decoder(
    neural_source: NeuralSource, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, data, inventory, _ = neural_source
    clustering.run_clustering(directory, data, inventory, "M11", 1, device=CPU)
    original = evaluation.load_representation
    batches = []

    def load(*args: object, **kwargs: object) -> tuple[Representation, dict[str, object]]:
        representation, source = original(*args, **kwargs)

        def inspect(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            assert not module.training and inputs[0].dtype == torch.float32
            assert bool(inputs[1].all()) and not torch.is_autocast_enabled("cpu")
            batches.append(len(inputs[0]))

        representation.model.encoder.register_forward_pre_hook(inspect)
        representation.model.decoder.register_forward_pre_hook(_forbidden)
        return representation, source

    monkeypatch.setattr(evaluation, "load_representation", load)
    path, = evaluation.run_evaluation(directory, data, inventory, ("M11",), (1,), device=CPU)
    assert sum(batches) == 16 * data.test_pixel_count
    assert evaluation.check_evaluation(directory, data, inventory, "M11", 1)["test_pixels"] == data.test_pixel_count
    assert _read_json(path / "completion.json")["checks_passed"]


def test_single_cluster_is_saved_with_explicit_undefined_scores(
    experiment: Fixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wood_degradation_map.experiments import diagnostic_metrics

    directory, data, inventory = experiment
    monkeypatch.setattr(clustering.FittedClusters, "predict",
                        lambda self, values: np.zeros(len(values), dtype=np.int64))
    _cluster(experiment)
    monkeypatch.setattr(diagnostic_metrics, "silhouette_samples_cosine_gpu", _forbidden)
    path, = evaluation.run_evaluation(directory, data, inventory, ("B0",), (1,), device=CPU)
    rows = _read_json(path / "scores.json")["records"]
    for row in rows:
        if row["metric"] == "silhouette" or row["metric"].startswith("adjusted_lla_"):
            assert row["value"] is None and row["status"] == "undefined"
            assert "single_cluster" in row["reason"]
        else:
            assert row["status"] == "defined"
            assert row["value"] == (0 if row["metric"].startswith("lfr_") else 1)
    assert not list((path / "silhouette").glob("*.npy"))
    assert evaluation.check_evaluation(directory, data, inventory, "B0", 1)["sample_count"] == 2


@pytest.mark.parametrize("change", ["missing_score", "wrong_repeat", "missing_draw"])
def test_semantic_coverage_checks_reject_rehashed_incomplete_records(
    experiment: Fixture, change: str,
) -> None:
    directory, data, inventory = experiment
    _cluster(experiment)
    path, = evaluation.run_evaluation(directory, data, inventory, ("B0",), (1,), device=CPU)
    name = "shared_inputs" if change == "missing_draw" else "scores"
    target = path / f"{name}.json"
    record = _read_json(target)
    if change == "missing_draw":
        record["samples"][0]["draws_sha256"].pop()
    elif change == "missing_score":
        record["records"].pop()
    else:
        record["records"][0]["repeat"] = 2
    target.write_text(json.dumps(record), encoding="utf-8")
    done_path = path / "completion.json"
    done = _read_json(done_path)
    done[f"{name}_sha256"] = _digest(target)
    done_path.write_text(json.dumps(done), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.check_evaluation(directory, data, inventory, "B0", 1)


def test_separate_invocations_share_realizations_and_group_check_detects_drift(
    experiment: Fixture,
) -> None:
    directory, data, inventory = experiment
    _cluster(experiment, repeats=(1, 2))
    evaluation.run_evaluation(directory, data, inventory, ("B0",), (1,), device=CPU, chunk_pixels=7)
    second, = evaluation.run_evaluation(directory, data, inventory, ("B0",), (2,), device=CPU, chunk_pixels=13)
    assert len(evaluation.check_evaluations(directory, data, inventory, ("B0",), (1, 2))) == 2
    path = second / "shared_inputs.json"
    shared = _read_json(path)
    shared["samples"][0]["draws_sha256"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(shared), encoding="utf-8")
    completion_path = second / "completion.json"
    done = _read_json(completion_path)
    done["shared_inputs_sha256"] = _digest(path)
    completion_path.write_text(json.dumps(done), encoding="utf-8")
    with pytest.raises(ValueError, match="different realized shared inputs"):
        evaluation.check_evaluations(directory, data, inventory, ("B0",), (1, 2))
