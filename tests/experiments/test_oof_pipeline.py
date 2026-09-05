"""CPU persistence tests and hand-calculated factorial contrasts; no real data."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from wood_degradation_map.experiments import oof_pipeline as pipeline
from wood_degradation_map.experiments.aggregation import REPEATED_METRICS, ScoreRecord
from wood_degradation_map.experiments.config import CLUSTER_COUNTS, FOLDS, REPEATS
from wood_degradation_map.experiments.data import FoldData
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput
from wood_degradation_map.experiments.manifests import CVManifest, _digest, _read_json, _write_json, create_cv_manifest

Fixture = tuple[Path, InputInventory, CVManifest]
SELECTED = ("B0", "M00", "M10", "M01", "M11")


def _records(metric: str = "lla_3") -> list[ScoreRecord]:
    values = {"M11": 0.8, "M10": 0.3, "M01": 0.4, "M00": 0.2}
    return [ScoreRecord(sample, fold, condition, 4, metric, repeat, "defined", value)
            for sample, fold in (("a", 1), ("b", 2)) for repeat in REPEATS
            for condition, value in values.items()]


@pytest.mark.parametrize("metric", pipeline.PRIMARY_METRICS)
def test_interaction_is_per_sample_repeat_with_original_metric_sign(metric: str) -> None:
    result = pipeline.interaction_contrast(_records(metric), expected_test_folds={"a": 1, "b": 2},
                                           metric=metric, k=4)
    assert result.mean == pytest.approx(0.3)
    assert result.sample_sd == result.repeat_sd == 0
    assert result.common_samples == 2
    assert len(result.availability) == 12
    assert [row.value for row in result.rows] == pytest.approx([0.3] * 6)


def test_interaction_intersects_all_four_conditions_and_retains_excluded_rows() -> None:
    records = _records()
    records = [replace(row, value=None, status="undefined", reason="fixture_undefined")
               if (row.sample_id, row.condition_id, row.repeat) == ("b", "M01", 2) else row
               for row in records]
    result = pipeline.interaction_contrast(records, expected_test_folds={"a": 1, "b": 2}, metric="lla_3", k=4)
    assert result.common_sample_ids == ("a",) and result.sample_sd is None
    assert result.rows[-1].value == pytest.approx(0.3)
    assert result.rows[-2].value is None
    assert result.rows[-2].unavailable_sources[0].condition_id == "M01"
    records = [replace(row, value=None, status="failed", reason="fixture_failure")
               if (row.sample_id, row.condition_id, row.repeat) == ("a", "M00", 3) else row
               for row in records]
    result = pipeline.interaction_contrast(records, expected_test_folds={"a": 1, "b": 2}, metric="lla_3", k=4)
    assert result.mean is None and result.common_samples == 0
    assert result.has_failed_or_interrupted_sources


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_fold", "ari", "silhouette"])
def test_interaction_rejects_incomplete_or_unplanned_inputs(change: str) -> None:
    records = _records()
    metric = "lla_3"
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records.append(records[0])
    elif change == "wrong_fold":
        records[0] = replace(records[0], fold=5)
    else:
        metric = change
    with pytest.raises(ValueError):
        pipeline.interaction_contrast(records, expected_test_folds={"a": 1, "b": 2}, metric=metric, k=4)


@pytest.fixture
def inventory_fixture(tmp_path: Path) -> Fixture:
    random = np.random.default_rng(213)
    samples = []
    coordinates = np.column_stack((np.arange(17) // 6, np.arange(17) % 6)).astype(np.int32)
    mask = np.zeros((3, 6), dtype=np.uint8)
    mask[coordinates[:, 0], coordinates[:, 1]] = 1
    for index in range(6):
        sample_id = f"KYOw{2800 + index:05d}"
        path = tmp_path / f"{sample_id}.h5"
        spectra = random.standard_normal((17, 256), dtype=np.float32)
        spectra -= spectra.mean(axis=1, keepdims=True)
        spectra /= spectra.std(axis=1, ddof=1, keepdims=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("snv", data=spectra)
            handle.create_dataset("pixel_row_col", data=coordinates)
            handle.create_dataset("valid_spectrum_mask", data=mask)
            handle.attrs.update(sample_id=sample_id, saved_pixel_count=17, schema_version=2)
        samples.append(SampleInput(sample_id, path, 3, 6, 17))
    inventory = InputInventory("fixture", tuple(samples), (), 900.0, 2300.0)
    manifest = create_cv_manifest(inventory, q=8)
    directory = tmp_path / "experiment"
    (directory / "manifests").mkdir(parents=True)
    _write_json(directory / "manifests/complete.json", {"artifact_sha256": {"fixture": "fixed"}})
    return directory, inventory, manifest


@pytest.fixture
def saved_sources(
    inventory_fixture: Fixture, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
) -> Fixture:
    """Synthetic score files isolate aggregation from model/evaluation computation.

    The boundary adapter below is only for these persistence tests. The final
    integration test uses actual clustering/evaluation source checks at all folds.
    """
    directory, inventory, manifest = inventory_fixture
    conditions = getattr(request, "param", ("B0",))
    samples = {sample.sample_id: sample for sample in inventory.samples}
    for fold in FOLDS:
        data = FoldData(inventory, manifest, fold)
        for condition in conditions:
            for repeat in REPEATS:
                results = directory / f"results/evaluation/{condition}/fold_{fold}/repeat_{repeat}"
                maps = directory / f"results/clustering/{condition}/fold_{fold}/repeat_{repeat}/maps"
                results.mkdir(parents=True)
                maps.mkdir(parents=True)
                rows = []
                for sample_id in data.test_sample_ids:
                    # Vary by sample, condition, and repeat to detect incorrect reduction order.
                    value = (int(sample_id[-1]) + repeat + SELECTED.index(condition)) / 16
                    for k in CLUSTER_COUNTS:
                        for metric in REPEATED_METRICS:
                            rows.append(asdict(ScoreRecord(sample_id, fold, condition, k, metric,
                                                           repeat, "defined", value)))
                    with h5py.File(samples[sample_id].path, "r") as handle:
                        coords = handle["pixel_row_col"][:]
                    label_maps = {}
                    for k in CLUSTER_COUNTS:
                        labels = np.zeros((3, 6), dtype=np.uint8)
                        # Label permutations produce ARI=1 without Hungarian matching.
                        labels[coords[:, 0], coords[:, 1]] = (np.arange(17) + repeat) % k + 1
                        label_maps[f"labels_k{k}"] = labels
                    np.savez(maps / f"{sample_id}.npz", **label_maps)
                _write_json(results / "scores.json", {"records": rows})
                _write_json(results / "completion.json", {"status": "fixture_complete", "revision": 1})

    def validate(experiment: Path, data: FoldData, inventory: InputInventory,
                 conditions: tuple[str, ...], repeats: tuple[int, ...]) -> tuple[dict[str, object], ...]:
        assert repeats == REPEATS
        reports = []
        for condition in conditions:
            for repeat in repeats:
                done = _read_json(experiment / f"results/evaluation/{condition}/fold_{data.fold}/repeat_{repeat}/completion.json")
                if done["status"] != "fixture_complete":
                    raise ValueError("Incomplete source run")
                reports.append({"condition": condition, "fold": data.fold, "repeat": repeat,
                                "shared_inputs_sha256": f"same_inputs_fold_{data.fold}"})
        return tuple(reports)

    monkeypatch.setattr(pipeline, "check_evaluations", validate)
    return inventory_fixture


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("OOF/check must not read spectra, train, predict or recompute checked summaries")


@pytest.mark.parametrize("saved_sources", [SELECTED], indirect=True)
def test_full_oof_save_check_pairs_interaction_and_ari(saved_sources: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, inventory, manifest = saved_sources
    monkeypatch.setattr(FoldData, "batches", _forbidden)
    output = pipeline.run_oof(directory, inventory, manifest, SELECTED, snapshot="fixture")
    report = _read_json(output / "completion.json")
    assert report["sample_count"] == 6 and report["source_run_count"] == 75
    assert report["score_record_count"] == 6 * 5 * 3 * 7 * 10
    summary = _read_json(output / "summaries/M11/k4.json")["metrics"]["lla_3"]
    assert summary["common_samples"] == 6
    assert summary["mean"] == pytest.approx((2.5 + 2 + 4) / 16)
    assert summary["sample_sd"] == pytest.approx(np.std(np.arange(6) / 16, ddof=1))
    assert summary["repeat_sd"] == pytest.approx(1 / 16)
    paired = _read_json(output / "comparisons/M11_minus_B0/k4.json")["metrics"]["lfr_noise"]
    assert paired["mean"] == pytest.approx(4 / 16)  # No LFR sign reversal.
    interaction = _read_json(output / "interaction/k4.json")["metrics"]["lla_3"]
    assert interaction["mean"] == 0
    ari = _read_json(output / "ari/M11/k4.json")
    assert ari["mean"] == 1 and ari["sample_sd"] == 0 and "repeat_sd" not in ari
    assert len(ari["samples"]) == 6 and all(len(row["pairs"]) == 3 for row in ari["samples"])
    monkeypatch.setattr(pipeline, "repeat_ari", _forbidden)
    monkeypatch.setattr(pipeline, "aggregate_scores", _forbidden)
    assert pipeline.check_oof(directory, inventory, manifest, "fixture")["status"] == "validated_existing_oof"
    with pytest.raises(FileExistsError):
        pipeline.run_oof(directory, inventory, manifest, SELECTED, snapshot="fixture")


def test_selected_conditions_are_explicit_and_unrequested_comparisons_not_invented(saved_sources: Fixture) -> None:
    directory, inventory, manifest = saved_sources
    output = pipeline.run_oof(directory, inventory, manifest, ("B0",), snapshot="baseline")
    run = _read_json(output / "run.json")
    assert run["conditions"] == ["B0"] and len(run["sources"]) == 15
    assert run["comparison_plan"]["interaction"] == "conditions_not_selected"
    assert all(row["status"] == "conditions_not_selected" for row in run["comparison_plan"]["paired"])
    assert not (output / "comparisons").exists()


@pytest.mark.parametrize("failure", ["missing_repeat", "failed_run", "changed_shared_inputs"])
def test_source_failure_stops_before_snapshot(saved_sources: Fixture, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    directory, inventory, manifest = saved_sources
    path = directory / "results/evaluation/B0/fold_5/repeat_3/completion.json"
    if failure == "missing_repeat":
        path.unlink()  # Disposable fixture only.
    elif failure == "failed_run":
        path.write_text(json.dumps({"status": "failed_or_interrupted"}), encoding="utf-8")
    else:
        def mismatch(*args: object, **kwargs: object) -> None:
            raise ValueError("Different realized shared inputs")
        monkeypatch.setattr(pipeline, "check_evaluations", mismatch)
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.run_oof(directory, inventory, manifest, ("B0",), snapshot="incomplete")
    assert not (directory / "results/oof/incomplete").exists()


@pytest.mark.parametrize("artifact", ["scores.json", "ari/B0/k4.json", "summaries/B0/k4.json", "source"])
def test_snapshot_or_source_change_invalidates_check(saved_sources: Fixture, artifact: str) -> None:
    directory, inventory, manifest = saved_sources
    output = pipeline.run_oof(directory, inventory, manifest, ("B0",), snapshot="saved")
    path = (directory / "results/evaluation/B0/fold_1/repeat_1/completion.json"
            if artifact == "source" else output / artifact)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError):
        pipeline.check_oof(directory, inventory, manifest, "saved")


def test_interrupt_does_not_publish_completion(saved_sources: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, inventory, manifest = saved_sources
    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt()
    monkeypatch.setattr(pipeline, "repeat_ari", interrupt)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_oof(directory, inventory, manifest, ("B0",), snapshot="interrupted")
    output = directory / "results/oof/interrupted"
    assert not (output / "completion.json").exists()
    assert _read_json(output / "failure.json")["error_type"] == "KeyboardInterrupt"


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_sample_fold", "failed"])
def test_incomplete_or_misidentified_score_rows_never_complete(saved_sources: Fixture, change: str) -> None:
    directory, inventory, manifest = saved_sources
    path = directory / "results/evaluation/B0/fold_1/repeat_1/scores.json"
    content = _read_json(path)
    if change == "missing":
        content["records"].pop()
    elif change == "duplicate":
        content["records"].append(content["records"][0])
    elif change == "wrong_sample_fold":
        content["records"][0]["fold"] = 5
    else:
        content["records"][0].update(status="failed", value=None, reason="fixture_failure")
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline.run_oof(directory, inventory, manifest, ("B0",), snapshot="bad_scores")
    assert not (directory / "results/oof/bad_scores/completion.json").exists()


def test_ari_invalid_pixel_cannot_be_dropped(saved_sources: Fixture) -> None:
    directory, inventory, manifest = saved_sources
    sample = FoldData(inventory, manifest, 1).test_sample_ids[0]
    path = directory / f"results/clustering/B0/fold_1/repeat_2/maps/{sample}.npz"
    with np.load(path) as stored:
        maps = {name: stored[name] for name in stored.files}
    maps["labels_k4"][0, 0] = 0
    np.savez(path, **maps)
    with pytest.raises(ValueError, match="integer labels"):
        pipeline.run_oof(directory, inventory, manifest, ("B0",), snapshot="bad_pixels")
    assert not (directory / "results/oof/bad_pixels/completion.json").exists()


@pytest.mark.parametrize("snapshot", ["../outside", ".", "C:/outside"])
def test_snapshot_path_cannot_escape_output_root(inventory_fixture: Fixture, snapshot: str) -> None:
    directory, inventory, manifest = inventory_fixture
    with pytest.raises(ValueError, match="Snapshot"):
        pipeline.run_oof(directory, inventory, manifest, ("B0",), snapshot=snapshot)


def test_actual_evaluation_artifacts_connect_across_all_folds(inventory_fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from wood_degradation_map.experiments.cluster_pipeline import run_clustering
    from wood_degradation_map.experiments.evaluation_pipeline import run_evaluation

    directory, inventory, manifest = inventory_fixture
    for fold in FOLDS:
        data = FoldData(inventory, manifest, fold)
        for repeat in REPEATS:
            run_clustering(directory, data, inventory, "B0", repeat, device=torch.device("cpu"))
        run_evaluation(directory, data, inventory, ("B0",), REPEATS, device=torch.device("cpu"))
    monkeypatch.setattr(FoldData, "batches", _forbidden)
    output = pipeline.run_oof(directory, inventory, manifest, ("B0",), snapshot="integration")
    assert pipeline.check_oof(directory, inventory, manifest, output.name)["source_run_count"] == 15
