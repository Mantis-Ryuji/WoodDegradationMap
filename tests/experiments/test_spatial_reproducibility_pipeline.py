"""Synthetic file roundtrips for the isolated spatial reproducibility pipeline."""

from __future__ import annotations

import csv
import importlib.util
from collections import Counter
from itertools import combinations
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest

from wood_degradation_map.experiments import spatial_reproducibility_pipeline as pipeline
from wood_degradation_map.experiments.global_manifest import GlobalData, create_global_manifest
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput
from wood_degradation_map.experiments.manifests import _read_json, _write_json


def _fixture(
    root: Path,
) -> tuple[Path, Path, GlobalData, InputInventory, np.ndarray]:
    """Two measured grids; the second has an empty boundary only in its initial map."""
    experiment, source = root / "supplement", root / "source"
    experiment.mkdir()
    source.mkdir()
    mask = np.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=np.uint8)
    labels = np.array([[1, 1, 2, 2], [1, 1, 2, 0]], dtype=np.uint8)
    samples = []
    for index, sample_id in enumerate(("KYOw02752", "KYOw02772")):
        path = root / f"{sample_id}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("valid_spectrum_mask", data=mask)
            handle.create_dataset("pixel_row_col", data=np.argwhere(mask).astype(np.int32))
            handle.attrs.update(sample_id=sample_id, saved_pixel_count=7, schema_version=2)
        samples.append(SampleInput(sample_id, path, 2, 4, 7))
        for encoder, kmeans in pipeline.MAP_IDS:
            output = pipeline.map_path(experiment, source, sample_id, encoder, kmeans)
            output.parent.mkdir(parents=True, exist_ok=True)
            saved = mask if index == 1 and (encoder, kmeans) == (1, 1) else labels
            np.savez_compressed(output, labels_k8=saved)
    inventory = InputInventory("fixture", tuple(samples), (), 900.0, 2300.0)
    data = GlobalData(inventory, create_global_manifest(inventory, q=2))
    return experiment, source, data, inventory, labels


def _mock_evaluation_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real map reads, metrics, aggregation, serialization, and seal verification."""
    def contract(
        experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
        *, include_overlap: bool,
    ) -> dict[str, object]:
        assert experiment != source
        assert set(data.train_sample_ids) == {sample.sample_id for sample in inventory.samples}
        return {"fixture": "frozen-provenance", "include_overlap": include_overlap,
                "sample_ids": list(data.train_sample_ids), "pairs": pipeline.pair_plan()}

    monkeypatch.setattr(pipeline, "_evaluation_contract", contract)
    monkeypatch.setattr(pipeline, "runtime_record", lambda device: {"device": str(device)})


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pair_plan_has_exact_groups_and_initial_comparisons_are_a_subset() -> None:
    pairs = pipeline.pair_plan()
    identities = [((pair["encoder_a"], pair["kmeans_a"]),
                   (pair["encoder_b"], pair["kmeans_b"])) for pair in pairs]
    assert len(pairs) == len(set(identities)) == 36
    assert identities == list(combinations(pipeline.MAP_IDS, 2))
    assert Counter(pair["group"] for pair in pairs) == {
        "within_encoder": 9, "between_encoder": 27,
    }
    initial = [identity for pair, identity in zip(pairs, identities, strict=True)
               if pair["initial_reference"]]
    assert initial == [((1, 1), identity) for identity in pipeline.MAP_IDS[1:]]
    for pair in pairs:
        assert pair["initial_reference"] is (
            (pair["encoder_a"], pair["kmeans_a"]) == (1, 1)
        )
        assert (pair["group"] == "within_encoder") == (pair["encoder_a"] == pair["encoder_b"])


def test_source_paths_keep_original_initial_map_and_added_encoders_separate(tmp_path: Path) -> None:
    experiment, source = tmp_path / "new", tmp_path / "old"
    sample = "KYOw02752"
    assert pipeline.map_path(experiment, source, sample, 1, 1) == (
        source / "results/clustering/M11/repeat_1/maps/KYOw02752.npz"
    )
    for kmeans in (2, 3):
        assert pipeline.map_path(experiment, source, sample, 1, kmeans) == (
            source / "results/global_k8_3seed_v1/clustering/M11"
            / f"repeat_{kmeans}/maps/{sample}.npz"
        )
    for encoder in (2, 3):
        for kmeans in (1, 2, 3):
            result, weights = pipeline.cluster_paths(experiment, encoder, kmeans)
            assert result == (experiment / "results/clustering/M11"
                              / f"encoder_{encoder}/kmeans_{kmeans}")
            assert weights == (experiment / "checkpoints/clustering/M11"
                               / f"encoder_{encoder}/kmeans_{kmeans}")
            assert pipeline.map_path(experiment, source, sample, encoder, kmeans) == (
                result / "maps" / f"{sample}.npz"
            )
    with pytest.raises(ValueError, match="added map identity"):
        pipeline.cluster_paths(experiment, 1, 1)
    with pytest.raises(ValueError, match="map identity"):
        pipeline.map_path(experiment, source, sample, True, 1)
    with pytest.raises(ValueError, match="sample ID"):
        pipeline.map_path(experiment, source, "../outside", 1, 1)


@pytest.mark.parametrize("output_name,source_name", [
    ("same", "same"), ("same/nested", "same"), ("same", "same/nested"),
    ("same/../same", "same"),
])
def test_source_and_output_must_not_overlap(
    tmp_path: Path, output_name: str, source_name: str,
) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        pipeline.separate_roots(tmp_path / output_name, tmp_path / source_name)
    assert not list(tmp_path.iterdir())


def test_disjoint_siblings_are_resolved_without_creating_directories(tmp_path: Path) -> None:
    output, source = pipeline.separate_roots(tmp_path / "new", tmp_path / "old")
    assert (output, source) == ((tmp_path / "new").resolve(), (tmp_path / "old").resolve())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", ["modify", "remove", "add"])
def test_seal_rejects_changed_missing_or_added_artifacts(tmp_path: Path, change: str) -> None:
    contract = {"fixture": "fixed", "windows": [3, 5, 9]}
    _write_json(tmp_path / "run.json", {"contract": contract})
    payload = tmp_path / "metrics.csv"
    payload.write_text("value\n0.5\n", encoding="utf-8")
    completion = pipeline.seal(tmp_path, "fixture_completed")
    assert set(completion["artifact_sha256"]) == {"run.json", "metrics.csv"}
    assert pipeline.check_sealed(tmp_path, "fixture_completed", contract) == completion
    with pytest.raises(ValueError, match="already sealed"):
        pipeline.seal(tmp_path, "fixture_completed")
    with pytest.raises(ValueError, match="contract/completion differs"):
        pipeline.check_sealed(tmp_path, "different_status", contract)
    with pytest.raises(ValueError, match="contract/completion differs"):
        pipeline.check_sealed(tmp_path, "fixture_completed", {"fixture": "different"})
    if change == "modify":
        payload.write_text("value\n1.0\n", encoding="utf-8")
    elif change == "remove":
        payload.unlink()
    else:
        (tmp_path / "unexpected.txt").write_text("new", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        pipeline.check_sealed(tmp_path, "fixture_completed", contract)


@pytest.mark.parametrize("change,message", [
    ("missing_prediction", "Missing predictions"), ("outside_mask", "Background"),
    ("label_range", "0..8"), ("shape", "matching"),
    ("dtype", "uint8"), ("extra_field", "label fields"),
])
def test_map_loading_rejects_mask_coverage_schema_and_value_changes(
    tmp_path: Path, change: str, message: str,
) -> None:
    experiment, source, _, inventory, original = _fixture(tmp_path)
    sample = inventory.samples[0]
    maps, mask = pipeline._read_maps(experiment, source, sample)
    assert tuple(maps) == pipeline.MAP_IDS and mask.dtype == np.bool_
    assert len(maps) == 9 and int(mask.sum()) == sample.saved_pixel_count
    changed = original.copy()
    if change == "missing_prediction":
        changed[0, 0] = 0
    elif change == "outside_mask":
        changed[1, 3] = 1
    elif change == "label_range":
        changed[0, 0] = 9
    elif change == "shape":
        changed = changed[:, :-1]
    elif change == "dtype":
        changed = changed.astype(np.int64)
    path = pipeline.map_path(experiment, source, sample.sample_id, 3, 3)
    fields = {"labels_k8": changed}
    if change == "extra_field":
        fields["unexpected"] = np.array([1])
    np.savez_compressed(path, **fields)
    with pytest.raises(ValueError, match=message):
        pipeline._read_maps(experiment, source, sample)


def test_changed_source_mask_is_not_hidden_by_an_intersection(tmp_path: Path) -> None:
    experiment, source, _, inventory, _ = _fixture(tmp_path)
    sample = inventory.samples[0]
    with h5py.File(sample.path, "r+") as handle:
        handle["valid_spectrum_mask"][0, 0] = 0
        handle["valid_spectrum_mask"][1, 3] = 1
    with pytest.raises(ValueError, match="Background|Missing predictions"):
        pipeline._read_maps(experiment, source, sample)


def test_missing_map_does_not_become_an_empty_boundary(tmp_path: Path) -> None:
    experiment, source, _, inventory, _ = _fixture(tmp_path)
    sample = inventory.samples[0]
    pipeline.map_path(experiment, source, sample.sample_id, 2, 2).unlink()
    with pytest.raises(FileNotFoundError):
        pipeline._read_maps(experiment, source, sample)


def test_evaluation_roundtrip_preserves_missingness_directions_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, source, data, inventory, labels = _fixture(tmp_path)
    _mock_evaluation_provenance(monkeypatch)
    source_before = {path: path.read_bytes() for path in source.rglob("*.npz")}
    completion = pipeline.evaluate(experiment, source, data, inventory, include_overlap=True)
    assert completion["status"] == "spatial_evaluation_completed"
    assert pipeline.check_evaluation(experiment, source, data, inventory) == completion
    output = experiment / "results/evaluation"
    summary = _read_json(output / "summary.json")
    assert summary["sample_count"] == 2
    assert summary["sample_ids"] == list(data.train_sample_ids)
    for group in ("within_encoder", "between_encoder"):
        assert summary["metrics"]["ari"][group]["sample_count"] == 2
        boundary = summary["metrics"]["boundary_3_lower"][group]
        assert boundary["sample_count"] == 1
        assert boundary["common_sample_ids"] == [data.train_sample_ids[0]]
        assert boundary["excluded_sample_ids"] == [data.train_sample_ids[1]]
        assert boundary["macro_mean"] == 1.0
    for sample_id in data.train_sample_ids:
        sample = _read_json(output / "samples" / f"{sample_id}.json")
        assert len(sample["maps"]) == 9 and len(sample["pairs"]) == 36
        assert sum(pair["initial_reference"] for pair in sample["pairs"]) == 8
        assert all("overlap" in pair for pair in sample["pairs"])
    map_records = _read_json(output / "map_sources.json")["maps"]
    assert len(map_records) == 18
    assert map_records[0]["path"] == str(pipeline.map_path(
        experiment, source, data.train_sample_ids[0], 1, 1,
    ).resolve())
    rows = _csv_rows(output / "pair_metrics.csv")
    assert len(rows) == 2 * 36 * 3 * 2
    assert len(_csv_rows(output / "map_diagnostics.csv")) == 18
    initial = _csv_rows(output / "initial_reference_metrics.csv")
    assert len(initial) == 2 * 8 * 3 * 2
    assert all(row["initial_reference"] == "True" for row in initial)
    empty_initial = [row for row in initial if row["sample_id"] == data.train_sample_ids[1]]
    forward = [row for row in empty_initial if row["direction"] == "a_to_b"]
    reverse = [row for row in empty_initial if row["direction"] == "b_to_a"]
    assert len(forward) == len(reverse) == 24
    assert all(row["value"] == "" and row["boundary_count"] == "0"
               and row["undefined_reason"] == "empty_source_boundary" for row in forward)
    assert all(row["value"] == "0.0" and row["matched_count"] == "0"
               and row["undefined_reason"] == "" for row in reverse)
    assert all(row["lower"] == row["upper"] == "" for row in empty_initial)
    assert all(path.read_bytes() == before for path, before in source_before.items())
    with pytest.raises(ValueError, match="Evaluation exists"):
        pipeline.evaluate(experiment, source, data, inventory, include_overlap=True)
    # Even an equivalent relabeling must not silently replace a frozen source artifact.
    path = pipeline.map_path(experiment, source, data.train_sample_ids[0], 1, 1)
    np.savez_compressed(path, labels_k8=np.array([0, 2, 1], dtype=np.uint8)[labels])
    with pytest.raises(ValueError, match="map source changed"):
        pipeline.check_evaluation(experiment, source, data, inventory)


def test_changed_evaluation_contract_never_publishes_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, source, data, inventory, _ = _fixture(tmp_path)
    _mock_evaluation_provenance(monkeypatch)
    calls = 0

    def changed_contract(*args: object, include_overlap: bool) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"revision": calls, "include_overlap": include_overlap}

    monkeypatch.setattr(pipeline, "_evaluation_contract", changed_contract)
    with pytest.raises(ValueError, match="inputs changed"):
        pipeline.evaluate(experiment, source, data, inventory)
    assert calls == 2
    assert not (experiment / "results/evaluation").exists()
    assert not list(experiment.glob(".spatial-evaluation-*"))


def _cli() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/experiments/global_spatial_reproducibility.py"
    spec = importlib.util.spec_from_file_location("spatial_reproducibility_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exposes_fixed_repeats_and_separate_output_defaults() -> None:
    module = _cli()
    default = module.parse_args(["evaluate", "--include-overlap"])
    assert default.include_overlap and default.encoders == [2, 3]
    assert default.experiment_dir.name == "global_spatial_reproducibility_v1"
    assert default.source_experiment_dir.name == "global_v1"
    assert default.experiment_dir != default.source_experiment_dir
    assert module.parse_args(["train", "--encoders", "3", "--resume"]).encoders == [3]
    assert module.parse_args(["report", "--dpi", "180"]).dpi == 180


@pytest.mark.parametrize("argv", [
    ["train", "--encoders", "1"], ["train", "--encoders", "2", "2"],
    ["evaluate", "--encoders", "2"], ["train", "--include-overlap"],
    ["check", "--resume"], ["training-check", "--resume"],
    ["cluster", "--chunk-pixels", "0"], ["report", "--dpi", "0"],
    ["train", "--device", "-1"],
])
def test_cli_rejects_inapplicable_or_invalid_options(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        _cli().parse_args(argv)
