"""Synthetic file roundtrips for the isolated spatial reproducibility pipeline."""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from wood_degradation_map.experiments import spatial_reproducibility_pipeline as pipeline
from wood_degradation_map.experiments.global_clustering import GlobalClusterRecord, GlobalClusters
from wood_degradation_map.experiments.global_manifest import GlobalData, create_global_manifest
from wood_degradation_map.experiments.global_seed_plan import global_kmeans_seed
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput
from wood_degradation_map.experiments.manifests import _digest, _read_json, _write_json
from wood_degradation_map.experiments.spatial_reproducibility_reporting import GlobalDisplay


def _fixture(
    root: Path,
) -> tuple[Path, Path, GlobalData, InputInventory, np.ndarray, pipeline.SpatialSources]:
    """Two measured grids; the second has an empty boundary only in its initial map."""
    experiment, source = root / "supplement", root / "source"
    sources = pipeline.SpatialSources(root / "artifacts")
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
            output = pipeline.map_path(experiment, source, sample_id, encoder, kmeans, sources=sources)
            output.parent.mkdir(parents=True, exist_ok=True)
            saved = mask if index == 1 and (encoder, kmeans) == (1, 1) else labels
            np.savez_compressed(output, labels_k8=saved)
    inventory = InputInventory("fixture", tuple(samples), (), 900.0, 2300.0)
    data = GlobalData(inventory, create_global_manifest(inventory, q=2))
    return experiment, source, data, inventory, labels, sources


def _mock_evaluation_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real map reads, metrics, aggregation, serialization, and seal verification."""
    def contract(
        experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
        *, include_overlap: bool, sources: pipeline.SpatialSources,
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


def test_clustering_seeds_are_fixed_distinct_and_shared_across_encoders() -> None:
    plan = pipeline.clustering_seed_plan()
    records = [row for row in plan["records"] if row["purpose"] == "kmeans"]
    assert len(records) == 3
    assert {row["kmeans_repeat"]: row["seed"] for row in records} == {
        repeat: pipeline.clustering_seed(repeat) for repeat in (1, 2, 3)
    }
    assert len({row["seed"] for row in plan["records"]}) == len(plan["records"])
    assert pipeline.clustering_seed(1) == global_kmeans_seed(1)
    assert pipeline.clustering_seed(3) == global_kmeans_seed(3)
    assert pipeline.clustering_seed_plan() == plan


def test_seed_collision_stops_before_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "clustering_seed", lambda repeat: global_kmeans_seed(1))
    with pytest.raises(ValueError, match="collision|collides"):
        pipeline.clustering_seed_plan()


def test_fitted_centers_roundtrip_keeps_seed_and_encoder_identity(tmp_path: Path) -> None:
    features = np.tile(np.eye(16, dtype=np.float32)[:8], (2, 1))
    fitted = pipeline.fit_clusters(features, ("fixture",), device=torch.device("cpu"))
    assert fitted.record.seed == pipeline.clustering_seed(2)
    assert fitted.record.fit_pixels == len(features)
    assert sum(fitted.record.fit_occupancy) == len(features)
    for encoder in (1, 2, 3):
        path = tmp_path / f"encoder_{encoder}.npz"
        pipeline.save_clusters(fitted, path, encoder)
        restored = pipeline.load_clusters(path, encoder, device=torch.device("cpu"))
        assert restored.record == fitted.record
        np.testing.assert_array_equal(restored.centroids, fitted.centroids)
        np.testing.assert_array_equal(restored.predict(features), fitted.predict(features))
        with pytest.raises(ValueError, match="contract differs"):
            pipeline.load_clusters(path, encoder % 3 + 1, device=torch.device("cpu"))
    with np.load(tmp_path / "encoder_1.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        centers = saved["centroids"].copy()
    metadata["fit"]["seed"] = pipeline.clustering_seed(1)
    np.savez_compressed(tmp_path / "wrong_seed.npz", centroids=centers, metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="Invalid .*centers"):
        pipeline.load_clusters(tmp_path / "wrong_seed.npz", 1, device=torch.device("cpu"))


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


def test_source_paths_reuse_six_maps_and_write_three_shared_seed_maps(tmp_path: Path) -> None:
    experiment, source = tmp_path / "new", tmp_path / "old"
    sources = pipeline.SpatialSources(tmp_path / "artifacts")
    sample = "KYOw02752"
    assert pipeline.map_path(experiment, source, sample, 1, 1, sources=sources) == (
        source / "results/clustering/M11/repeat_1/maps/KYOw02752.npz"
    )
    assert pipeline.map_path(experiment, source, sample, 1, 3, sources=sources) == (
        source / "results/global_k8_3seed_v1/clustering/M11/repeat_3/maps/KYOw02752.npz"
    )
    for encoder in (1, 2, 3):
        for kmeans in (1, 2, 3):
            result, weights = pipeline.cluster_paths(experiment, encoder, kmeans)
            assert result == (experiment / "results/clustering/M11"
                              / f"encoder_{encoder}/kmeans_{kmeans}")
            assert weights == (experiment / "checkpoints/clustering/M11"
                               / f"encoder_{encoder}/kmeans_{kmeans}")
            if kmeans == 2:
                expected = result / "maps" / f"{sample}.npz"
            elif encoder > 1:
                cached = pipeline.cluster_paths(sources.artifacts, encoder, kmeans)[0]
                expected = cached / "maps" / f"{sample}.npz"
            else:
                continue
            assert pipeline.map_path(experiment, source, sample, encoder, kmeans, sources=sources) == expected
    with pytest.raises(ValueError, match="map identity"):
        pipeline.cluster_paths(experiment, 4, 1)
    with pytest.raises(ValueError, match="map identity"):
        pipeline.map_path(experiment, source, sample, True, 1, sources=sources)
    with pytest.raises(ValueError, match="sample ID"):
        pipeline.map_path(experiment, source, "../outside", 1, 1, sources=sources)
    assert not list(tmp_path.iterdir())


def test_all_three_input_and_output_roots_must_be_disjoint(tmp_path: Path) -> None:
    sources = pipeline.SpatialSources(tmp_path / "artifacts")
    source = tmp_path / "global"
    for output in (sources.artifacts, sources.artifacts / "child", tmp_path, source):
        with pytest.raises(ValueError, match="disjoint"):
            sources.validate_roots(output, source)
    with pytest.raises(ValueError, match="disjoint"):
        sources.validate_roots(tmp_path / "output", sources.artifacts / "child")
    assert not list(tmp_path.iterdir())


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
    experiment, source, _, inventory, original, sources = _fixture(tmp_path)
    sample = inventory.samples[0]
    maps, mask = pipeline._read_maps(experiment, source, sample, sources=sources)
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
    path = pipeline.map_path(experiment, source, sample.sample_id, 3, 3, sources=sources)
    fields = {"labels_k8": changed}
    if change == "extra_field":
        fields["unexpected"] = np.array([1])
    np.savez_compressed(path, **fields)
    with pytest.raises(ValueError, match=message):
        pipeline._read_maps(experiment, source, sample, sources=sources)


def test_changed_source_mask_is_not_hidden_by_an_intersection(tmp_path: Path) -> None:
    experiment, source, _, inventory, _, sources = _fixture(tmp_path)
    sample = inventory.samples[0]
    with h5py.File(sample.path, "r+") as handle:
        handle["valid_spectrum_mask"][0, 0] = 0
        handle["valid_spectrum_mask"][1, 3] = 1
    with pytest.raises(ValueError, match="Background|Missing predictions"):
        pipeline._read_maps(experiment, source, sample, sources=sources)


def test_missing_map_does_not_become_an_empty_boundary(tmp_path: Path) -> None:
    experiment, source, _, inventory, _, sources = _fixture(tmp_path)
    sample = inventory.samples[0]
    pipeline.map_path(experiment, source, sample.sample_id, 2, 2, sources=sources).unlink()
    with pytest.raises(FileNotFoundError):
        pipeline._read_maps(experiment, source, sample, sources=sources)


def test_evaluation_roundtrip_preserves_missingness_directions_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, source, data, inventory, labels, sources = _fixture(tmp_path)
    _mock_evaluation_provenance(monkeypatch)
    source_before = {path: path.read_bytes() for root in (source, sources.artifacts)
                     for path in root.rglob("*.npz")}
    completion = pipeline.evaluate(experiment, source, data, inventory, sources=sources, include_overlap=True)
    assert completion["status"] == "spatial_evaluation_completed"
    assert pipeline.check_evaluation(experiment, source, data, inventory, sources=sources) == completion
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
        experiment, source, data.train_sample_ids[0], 1, 1, sources=sources,
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
        pipeline.evaluate(experiment, source, data, inventory, sources=sources, include_overlap=True)
    # Even an equivalent relabeling must not silently replace a frozen source artifact.
    path = pipeline.map_path(experiment, source, data.train_sample_ids[0], 1, 1, sources=sources)
    np.savez_compressed(path, labels_k8=np.array([0, 2, 1], dtype=np.uint8)[labels])
    with pytest.raises(ValueError, match="map source changed"):
        pipeline.check_evaluation(experiment, source, data, inventory, sources=sources)


def test_changed_evaluation_contract_never_publishes_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, source, data, inventory, _, sources = _fixture(tmp_path)
    _mock_evaluation_provenance(monkeypatch)
    calls = 0

    def changed_contract(
        *args: object, include_overlap: bool, sources: pipeline.SpatialSources,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"snapshot": calls, "include_overlap": include_overlap}

    monkeypatch.setattr(pipeline, "_evaluation_contract", changed_contract)
    with pytest.raises(ValueError, match="inputs changed"):
        pipeline.evaluate(experiment, source, data, inventory, sources=sources)
    assert calls == 2
    assert not (experiment / "results/evaluation").exists()
    assert not list(experiment.glob(".spatial-evaluation-*"))


def test_report_saves_one_pooled_color_mapping_without_changing_evaluation_or_source_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, source, data, inventory, _, sources = _fixture(tmp_path)
    _mock_evaluation_provenance(monkeypatch)
    pipeline.evaluate(experiment, source, data, inventory, sources=sources)
    evaluation_root = experiment / "results/evaluation"
    evaluation_before = {path: path.read_bytes() for path in evaluation_root.rglob("*")
                         if path.is_file()}
    maps_before = {path: path.read_bytes() for directory in (experiment, source, sources.artifacts)
                   for path in directory.rglob("*.npz")}
    mappings_seen = []
    global_lookup = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8], dtype=np.uint8)
    palette = ("white", "blue", "orange", "green", "pink", "cyan", "red", "yellow", "purple")
    reference = GlobalDisplay(global_lookup, palette, {"fixture": "saved global display"})
    monkeypatch.setattr(pipeline, "load_global_display", lambda source, samples: reference)

    def render_small(
        sample_id: str, maps: dict, mask: np.ndarray, evaluation: dict, output: Path,
        *, dpi: int, display_lookups: dict, palette: tuple[str, ...],
    ) -> list[Path]:
        # Actual raster colors and invariance are covered by the renderer tests.
        mappings_seen.append({identity: lookup.tolist()
                              for identity, lookup in display_lookups.items()})
        assert palette == reference.palette
        output.mkdir(parents=True)
        names = ["01_nine_maps.png", "03_boundary_frequency.png", "boundary_frequency.npz"]
        for name in names:
            (output / name).write_bytes(b"rendered fixture")
        return [output / name for name in names]

    def summary_small(summary: dict, output: Path, *, dpi: int) -> list[Path]:
        output.mkdir(parents=True)
        names = ["01_ari_vs_boundary_coverage.png"]
        for name in names:
            (output / name).write_bytes(b"summary fixture")
        return [output / name for name in names]

    monkeypatch.setattr(pipeline, "render_sample", render_small)
    monkeypatch.setattr(pipeline, "render_summary", summary_small)
    completion = pipeline.write_report(experiment, source, data, inventory, sources=sources, dpi=40)
    assert completion["status"] == "spatial_report_completed"
    assert "color_matching.json" in completion["artifact_sha256"]
    assert pipeline.check_report(experiment, source, data, inventory, sources=sources)["samples"] == 2
    assert len(mappings_seen) == 2 and mappings_seen[0] == mappings_seen[1]
    matching = _read_json(experiment / "results/report/color_matching.json")
    assert matching["sample_ids"] == list(data.train_sample_ids)
    assert len(matching["mappings"]) == 9
    assert matching["contract"]["reference"] == [1, 1]
    assert matching["mappings"][0]["target_to_display_including_background"] == global_lookup.tolist()
    assert all(row["valid_pixels"] == 14 for row in matching["initial_mappings"])
    assert matching["global_display"] == reference.provenance
    assert "captions.json" in completion["artifact_sha256"]
    figures = list((experiment / "results/report").rglob("*.png"))
    assert len(figures) == 2 * len(inventory.samples) + 1
    assert all(path.read_bytes() == value for path, value in evaluation_before.items())
    assert all(path.read_bytes() == value for path, value in maps_before.items())


def _cluster_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[pipeline.SpatialSources, Path, Path, GlobalData, InputInventory]:
    sources = pipeline.SpatialSources(root / "artifacts")
    source, experiment = root / "global", root / "output"
    mask = np.ones((2, 4), dtype=np.uint8)
    labels = np.tile(np.arange(1, 5, dtype=np.uint8), (2, 1))
    sample_id = "KYOw02752"
    sample_path = root / f"{sample_id}.h5"
    with h5py.File(sample_path, "w") as handle:
        handle.create_dataset("valid_spectrum_mask", data=mask)
        handle.create_dataset("pixel_row_col", data=np.argwhere(mask).astype(np.int32))
        handle.attrs.update(sample_id=sample_id, saved_pixel_count=8, schema_version=2)
    inventory = InputInventory("fixture", (SampleInput(sample_id, sample_path, 2, 4, 8),), (), 900.0, 2300.0)
    data = GlobalData(inventory, create_global_manifest(inventory, q=8))
    for encoder, kmeans in pipeline.MAP_IDS:
        if kmeans == 2:
            continue
        path = pipeline.map_path(experiment, source, sample_id, encoder, kmeans, sources=sources)
        path.parent.mkdir(parents=True)
        np.savez_compressed(path, labels_k8=labels)
        result = path.parent.parent
        _write_json(result / "run.json", {"contract": {"fixture": [encoder, kmeans]}})
        pipeline.seal(result, "spatial_clustering_completed")
    for encoder in (2, 3):
        checkpoint = sources.artifacts / f"checkpoints/neural/M11/encoder_{encoder}/final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"fixture encoder {encoder}".encode())

    def snapshot(
        binding: pipeline.SpatialSources, reference: Path,
        fit_data: GlobalData, inputs: InputInventory,
    ) -> dict[str, object]:
        assert binding == sources and reference == source
        assert fit_data is data and inputs is inventory
        return {
            "global": {"regenerated_initial_available": False},
            "encoders": {str(encoder): {"encoder": encoder} for encoder in (1, 2, 3)},
            "artifact_sha256": {
                str(path.resolve()): _digest(path) for directory in (source, sources.artifacts)
                for path in sorted(directory.rglob("*")) if path.is_file()
            },
        }

    monkeypatch.setattr(pipeline.SpatialSources, "_snapshot", snapshot)
    monkeypatch.setattr(pipeline, "training_source", lambda output, src, data, e: {"encoder": e})
    return sources, experiment, source, data, inventory


def test_three_fits_resume_evaluate_and_report_without_mutating_consumed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, experiment, source, data, inventory = _cluster_fixture(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for root in (sources.artifacts, source)
              for path in root.rglob("*") if path.is_file()}
    sources.prepare(experiment, source, data, inventory)
    sources.check_prepared(experiment, source, data, inventory)
    with pytest.raises(FileNotFoundError):
        pipeline._read_maps(experiment, source, inventory.samples[0], sources=sources)
    features = np.eye(16, dtype=np.float32)[:8]
    batch = SimpleNamespace(snv=np.zeros((8, 256), dtype=np.float32),
                            pixel_row_col=np.argwhere(np.ones((2, 4), dtype=bool)))
    monkeypatch.setattr(pipeline, "load_representation",
                        lambda base, src, data, e, **kwargs: (object(), {"encoder": e}))
    monkeypatch.setattr(pipeline, "collect_global_features", lambda *args, **kwargs: features)
    monkeypatch.setattr(pipeline, "AllPixelData",
                        lambda samples: SimpleNamespace(all_batches=lambda chunk_pixels: iter([batch])))
    monkeypatch.setattr(pipeline, "_transform_batch", lambda *args: SimpleNamespace(values=features))
    monkeypatch.setattr(pipeline, "runtime_record", lambda device: {"fixture_device": str(device)})
    original_fit, original_load = pipeline.fit_clusters, pipeline.load_clusters
    fitted_seeds = []

    def fit_on_cpu(values: np.ndarray, samples: tuple[str, ...], **kwargs: object) -> GlobalClusters:
        fitted = original_fit(values, samples, device=torch.device("cpu"))
        fitted_seeds.append(fitted.record.seed)
        return fitted

    monkeypatch.setattr(pipeline, "fit_clusters", fit_on_cpu)
    monkeypatch.setattr(pipeline, "load_clusters",
                        lambda path, e, **kwargs: original_load(path, e, device=torch.device("cpu")))
    for encoder in (1, 2, 3):
        sources.cluster(experiment, source, data, inventory, encoder, device=torch.device("cuda:0"))
        sources.cluster(experiment, source, data, inventory, encoder,
                        device=torch.device("cuda:0"), resume=True)
    assert fitted_seeds == [pipeline.clustering_seed(2)] * 3
    pipeline.evaluate(experiment, source, data, inventory, sources=sources, include_overlap=True)
    pipeline.check_evaluation(experiment, source, data, inventory, sources=sources)
    evaluation = experiment / "results/evaluation"
    records = _read_json(evaluation / "map_sources.json")["maps"]
    assert len(records) == 9
    assert sum(Path(row["path"]).is_relative_to(experiment) for row in records) == 3
    assert all(Path(row["path"]).is_relative_to(experiment) == (row["kmeans_repeat"] == 2) for row in records)
    assert _read_json(evaluation / "run.json")["contract"]["scope"] == pipeline.SCOPE
    assert len(_read_json(evaluation / "samples/KYOw02752.json")["pairs"]) == 36
    reference = GlobalDisplay(np.arange(9, dtype=np.uint8),
                              ("white", "blue", "orange", "green", "pink", "cyan", "red", "yellow", "purple"),
                              {"fixture": "saved global display"})
    monkeypatch.setattr(pipeline, "load_global_display", lambda *args: reference)
    pipeline.write_report(experiment, source, data, inventory, sources=sources, dpi=20)
    assert pipeline.check_report(experiment, source, data, inventory, sources=sources)["samples"] == 1
    assert len(list((experiment / "results/report").rglob("*.png"))) == 2 * len(inventory.samples) + 1
    assert all(path.read_bytes() == value for path, value in before.items())
    initial = pipeline.map_path(experiment, source, "KYOw02752", 1, 1, sources=sources)
    np.savez_compressed(initial, labels_k8=np.ones((2, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="contract/completion differs"):
        sources.check_prepared(experiment, source, data, inventory)


def test_cached_clusters_keep_producer_identity_but_require_current_numerical_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, _, source, data, inventory = _cluster_fixture(tmp_path, monkeypatch)
    (sources.artifacts / "protocol").mkdir()
    _write_json(sources.artifacts / "protocol/completion.json", {"fixture": "input selection"})
    result, weights = pipeline.cluster_paths(sources.artifacts, 3, 1)
    weights.mkdir(parents=True)
    center_path = weights / "centers_k8.npz"
    center_path.write_bytes(b"fixture centers")
    record = GlobalClusterRecord("M11", 16, 8, global_kmeans_seed(1), data.train_sample_ids, 8,
                                 (1,) * 8, 0.1, 0.1, 500, 1e-4, None, "fixture", 0.01, "cpu", 0)
    monkeypatch.setattr(pipeline, "load_repeat_clusters",
                        lambda *args, **kwargs: SimpleNamespace(record=record))
    (result / "run.json").unlink()
    _write_json(result / "run.json", {"contract": {"code_sha256": {
        "spatial_reproducibility_pipeline.py": "a" * 64,
    }}})
    contract = sources._cached_cluster_contract(source, data, 3, 1)
    assert contract["code_sha256"]["spatial_reproducibility_pipeline.py"] == "a" * 64
    run = {"contract": contract, "fit": asdict(record), "center_roundtrip_exact": True,
           "centers_sha256": _digest(center_path)}
    (result / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (result / "completion.json").unlink()
    completion = pipeline.seal(result, "spatial_clustering_completed")
    assert sources._check_cached_cluster(source, data, inventory, 3, 1) == completion
    run["contract"]["code_sha256"]["global_k8_repeats.py"] = "changed"
    (result / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (result / "completion.json").unlink()
    pipeline.seal(result, "spatial_clustering_completed")
    with pytest.raises(ValueError, match="contract/completion differs"):
        sources._check_cached_cluster(source, data, inventory, 3, 1)


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
    assert default.include_overlap
    assert default.experiment_dir.name == "global_spatial_reproducibility_v2"
    assert default.source_experiment_dir.name == "global_v1"
    assert default.artifacts_dir.name == "global_spatial_reproducibility_v1"
    assert default.experiment_dir != default.source_experiment_dir
    assert module.parse_args(["cluster"]).encoders == [1, 2, 3]
    assert module.parse_args(["cluster", "--encoders", "3", "--resume"]).encoders == [3]
    assert module.parse_args(["report", "--dpi", "180"]).dpi == 180
    assert module.parse_args(["cluster", "--artifacts-dir", "cached"]).artifacts_dir == Path("cached")
    assert module.parse_args(["cluster", "--encoders", "1"]).encoders == [1]


@pytest.mark.parametrize("argv", [
    ["train"], ["training-check"], ["cluster", "--encoders", "2", "2"],
    ["evaluate", "--encoders", "2"], ["cluster", "--include-overlap"],
    ["check", "--resume"],
    ["cluster", "--chunk-pixels", "0"], ["report", "--dpi", "0"],
    ["cluster", "--device", "-1"], ["cluster", "--reseed-from", "original"],
    ["cluster", "--encoders", "4"],
])
def test_cli_rejects_inapplicable_or_invalid_options(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        _cli().parse_args(argv)
