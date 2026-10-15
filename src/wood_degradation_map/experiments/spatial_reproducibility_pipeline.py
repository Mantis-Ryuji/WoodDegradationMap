"""Isolated provenance, clustering, and evaluation for the 3 x 3 spatial experiment."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch

from .cluster_pipeline import _transform_batch
from .clustering import LabelMap, cluster_contract
from .config import experiment_config
from .global_clustering import (
    AllPixelData, check_global_clustering, collect_global_features, condition_paths,
    read_label_map, source_record,
)
from .global_k8_repeats import (
    check_repeat_clustering, fit_repeat_clusters, load_repeat_clusters, repeat_paths,
    save_repeat_clusters,
)
from .global_manifest import GlobalData
from .input_validation import InputInventory, SampleInput
from .manifests import _digest, _read_json, _require, _write_json
from .spatial_reproducibility_metrics import WINDOWS, compare_maps, map_diagnostics, validate_map
from .spatial_reproducibility_reporting import render_sample, render_summary, summarize_samples
from .spatial_reproducibility_training import load_representation, seed_plan, training_source
from .training import runtime_record

SCOPE = "global_spatial_reproducibility_v1"
MAP_IDS = tuple((encoder, kmeans) for encoder in (1, 2, 3) for kmeans in (1, 2, 3))


def separate_roots(experiment: Path, source: Path) -> tuple[Path, Path]:
    """Reject writes inside, around, or directly onto the immutable source tree."""
    experiment, source = experiment.resolve(), source.resolve()
    _require(not experiment.is_relative_to(source) and not source.is_relative_to(experiment),
             "Supplement and source experiment directories must be disjoint")
    return experiment, source


def pair_plan() -> list[dict[str, object]]:
    """Return nine within-encoder and 27 between-encoder pairs, each exactly once."""
    return [{"encoder_a": left[0], "kmeans_a": left[1],
             "encoder_b": right[0], "kmeans_b": right[1],
             "group": "within_encoder" if left[0] == right[0] else "between_encoder",
             "initial_reference": left == (1, 1)}
            for left, right in combinations(MAP_IDS, 2)]


def cluster_paths(experiment: Path, encoder: int, kmeans: int) -> tuple[Path, Path]:
    _require(type(encoder) is int and encoder in (2, 3)
             and type(kmeans) is int and kmeans in (1, 2, 3), "Invalid added map identity")
    suffix = f"clustering/M11/encoder_{encoder}/kmeans_{kmeans}"
    return experiment / "results" / suffix, experiment / "checkpoints" / suffix


def map_path(experiment: Path, source: Path, sample_id: str, encoder: int, kmeans: int) -> Path:
    _require(type(encoder) is int and type(kmeans) is int
             and (encoder, kmeans) in MAP_IDS, "Invalid map identity")
    _require(bool(sample_id) and Path(sample_id).name == sample_id
             and sample_id not in (".", ".."), "Invalid sample ID")
    if encoder == 1:
        result = (condition_paths(source, "M11")[0] if kmeans == 1
                  else repeat_paths(source, "M11", kmeans)[0])
    else:
        result = cluster_paths(experiment, encoder, kmeans)[0]
    return result / "maps" / f"{sample_id}.npz"


def _code_hashes(names: tuple[str, ...]) -> dict[str, str]:
    return {name: _digest(Path(__file__).with_name(name)) for name in names}


def seal(output: Path, status: str) -> dict[str, object]:
    _require(not (output / "completion.json").exists(), "Output is already sealed")
    hashes = {path.relative_to(output).as_posix(): _digest(path)
              for path in sorted(output.rglob("*")) if path.is_file()}
    completion = {"status": status, "artifact_sha256": hashes}
    _write_json(output / "completion.json", completion)
    return completion


def check_sealed(output: Path, status: str, contract: dict[str, object]) -> dict[str, object]:
    completion = _read_json(output / "completion.json")
    _require(completion.get("status") == status
             and _read_json(output / "run.json").get("contract") == contract,
             f"Spatial experiment contract/completion differs: {output}")
    files = {path.relative_to(output).as_posix(): path
             for path in output.rglob("*") if path.is_file()
             and path != output / "completion.json"}
    hashes = completion.get("artifact_sha256")
    _require(isinstance(hashes, dict) and set(files) == set(hashes)
             and all(_digest(path) == hashes[name] for name, path in files.items()),
             f"Spatial experiment artifact changed: {output}")
    return completion


def _source_snapshot(source: Path, data: GlobalData, inventory: InputInventory) -> dict[str, object]:
    """Audit the three canonical maps and the separately regenerated k=1 copy."""
    _require(tuple(sorted(sample.sample_id for sample in inventory.samples)) == data.train_sample_ids,
             "Prediction samples differ from the fixed global manifest")
    check_global_clustering(source, data, inventory, "M11")
    entries = [("canonical_initial", *condition_paths(source, "M11"))]
    for kmeans in (2, 3):
        check_repeat_clustering(source, data, inventory, "M11", kmeans)
        entries.append((f"kmeans_{kmeans}", *repeat_paths(source, "M11", kmeans)))
    audit_results, audit_weights = repeat_paths(source, "M11", 1)
    if audit_results.exists() or audit_weights.exists():
        check_repeat_clustering(source, data, inventory, "M11", 1)
        entries.append(("regenerated_initial_audit_only", audit_results, audit_weights))
    return {
        "source_experiment": str(source.resolve()),
        "manifest_sha256": _digest(source / "manifests/complete.json"),
        "manifest_artifact_sha256": _read_json(source / "manifests/complete.json")["artifact_sha256"],
        "encoder_1": source_record(source, data, "M11"),
        "clusters": [{"role": role, "results": str(result.resolve()),
                      "centers": str((weights / "centers_k8.npz").resolve()),
                      "run_sha256": _digest(result / "run.json"),
                      "completion_sha256": _digest(result / "completion.json"),
                      "centers_sha256": _digest(weights / "centers_k8.npz")}
                     for role, result, weights in entries],
        "regenerated_initial_available": audit_results.exists(),
    }


def protocol_contract(source: Path, data: GlobalData, inventory: InputInventory) -> dict[str, object]:
    return {
        "schema_version": 1, "scope": SCOPE, "condition": "M11", "K": 8,
        "plan": "docs/design/spatial_map_reproducibility_plan.md",
        "definition_version": "valid-edge-midpoint-chebyshev-v1",
        "windows": list(WINDOWS), "radii_pixels": [1, 2, 4],
        "boundary_matching": "many-to-one proximity; both directions; no orientation matching",
        "map_ids": [list(identity) for identity in MAP_IDS], "pairs": pair_plan(),
        "sample_ids": list(data.train_sample_ids), "fit_pixels": data.train_pixel_count,
        "prediction_pixels": sum(sample.saved_pixel_count for sample in inventory.samples),
        "seed_plan": seed_plan(), "source": _source_snapshot(source, data, inventory),
        "primary_aggregation": "all expected pairs defined; equal sample weight; no imputation",
        "held_out": False, "scientific_pass_threshold": None,
    }


def prepare(experiment: Path, source: Path, data: GlobalData, inventory: InputInventory) -> dict:
    """Freeze source references and seeds before either added encoder is trained."""
    experiment, source = separate_roots(experiment, source)
    output = experiment / "protocol"
    _require(not output.exists(), "Protocol exists; use --resume to verify it")
    contract = protocol_contract(source, data, inventory)
    audit = []
    if contract["source"]["regenerated_initial_available"]:
        for sample in inventory.samples:
            original = read_label_map(map_path(experiment, source, sample.sample_id, 1, 1), sample)
            regenerated = read_label_map(repeat_paths(source, "M11", 1)[0] / "maps"
                                         / f"{sample.sample_id}.npz", sample)
            difference = compare_maps(original, regenerated, original > 0)
            audit.append({"sample_id": sample.sample_id,
                          "exact_label_array_equal": bool(np.array_equal(original, regenerated)),
                          "raw_label_changed_pixels": int(np.count_nonzero(original != regenerated)),
                          "comparison": difference})
    _require(protocol_contract(source, data, inventory) == contract,
             "Source artifacts changed during preparation")
    experiment.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".spatial-prepare-", dir=experiment) as temporary:
        stage = Path(temporary) / "protocol"
        stage.mkdir()
        _write_json(stage / "run.json", {"contract": contract})
        _write_json(stage / "initial_map_audit.json", {
            "status": "compared" if audit else "regenerated_copy_not_available",
            "canonical": "original global_v1/results/clustering/M11/repeat_1",
            "regenerated_copy_is_excluded_from_nine_maps": True, "samples": audit,
        })
        completion = seal(stage, "spatial_protocol_prepared")
        stage.rename(output)
    return completion


def check_prepared(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    experiment, source = separate_roots(experiment, source)
    return check_sealed(experiment / "protocol", "spatial_protocol_prepared",
                        protocol_contract(source, data, inventory))


def _cluster_contract(experiment: Path, source: Path, data: GlobalData, encoder: int) -> dict:
    return {
        "schema_version": 1, "scope": SCOPE, "condition": "M11", "K": 8,
        "encoder_repeat": encoder, "protocol_sha256": _digest(experiment / "protocol/completion.json"),
        "source": training_source(experiment, source, data, encoder),
        "sample_ids": list(data.train_sample_ids), "fit_pixels": data.train_pixel_count,
        "settings": experiment_config()["clustering"], "runtime": cluster_contract(),
        "seed_plan": seed_plan(),
        "code_sha256": _code_hashes((Path(__file__).name, "global_k8_repeats.py",
            "global_clustering.py", "cluster_pipeline.py", "clustering.py", "neural.py",
            "spatial_reproducibility_training.py")),
    }


def check_cluster(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
    encoder: int, kmeans: int,
) -> dict[str, object]:
    result, weights = cluster_paths(experiment, encoder, kmeans)
    contract = {**_cluster_contract(experiment, source, data, encoder), "kmeans_repeat": kmeans}
    completion = check_sealed(result, "spatial_clustering_completed", contract)
    run = _read_json(result / "run.json")
    center_path = weights / "centers_k8.npz"
    _require(run["centers_sha256"] == _digest(center_path)
             and run.get("center_roundtrip_exact") is True, "Saved center payload changed")
    centers = load_repeat_clusters(center_path, "M11", kmeans, device=torch.device("cpu"))
    _require(centers.record.sample_ids == data.train_sample_ids
             and centers.record.fit_pixels == data.train_pixel_count
             and all(type(count) is int and count >= 0 for count in centers.record.fit_occupancy)
             and json.loads(json.dumps(asdict(centers.record))) == run["fit"],
             "Saved centers have different fit provenance")
    expected = {f"{sample.sample_id}.npz" for sample in inventory.samples}
    _require({path.name for path in (result / "maps").iterdir()} == expected,
             "Added map sample coverage differs")
    for sample in inventory.samples:
        read_label_map(result / "maps" / f"{sample.sample_id}.npz", sample)
    return completion


def cluster(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory, encoder: int,
    *, device: torch.device, chunk_pixels: int = 1024, resume: bool = False,
) -> None:
    """Extract selected features once, then fit the three fixed initializations."""
    experiment, source = separate_roots(experiment, source)
    _require(type(chunk_pixels) is int and chunk_pixels > 0, "Invalid chunk size")
    _require(device.type == "cuda", "Production clustering requires a single CUDA device")
    check_prepared(experiment, source, data, inventory)
    pending = []
    for kmeans in (1, 2, 3):
        results, weights = cluster_paths(experiment, encoder, kmeans)
        if results.exists() or weights.exists():
            _require(resume, "Cluster output exists; use --resume to verify and skip")
            check_cluster(experiment, source, data, inventory, encoder, kmeans)
            print(f"encoder {encoder}, kmeans {kmeans}: verified and skipped", flush=True)
        else:
            pending.append(kmeans)
    if not pending:
        return
    contract = _cluster_contract(experiment, source, data, encoder)
    representation, reference = load_representation(experiment, source, data, encoder, device=device)
    _require(reference == contract["source"], "Representation source differs")
    features = collect_global_features(data, representation, "M11", chunk_pixels=chunk_pixels)
    samples = {sample.sample_id: sample for sample in inventory.samples}
    for kmeans in pending:
        started = time.perf_counter()
        results, weights = cluster_paths(experiment, encoder, kmeans)
        with TemporaryDirectory(prefix=".spatial-cluster-", dir=experiment) as temporary:
            stage, weight_stage = Path(temporary) / "results", Path(temporary) / "weights"
            (stage / "maps").mkdir(parents=True)
            weight_stage.mkdir()
            fitted = fit_repeat_clusters(features, "M11", data.train_sample_ids, kmeans, device=device)
            centers = weight_stage / "centers_k8.npz"
            # This payload format describes one KMeans fit. Encoder identity and its
            # weights hash are bound by run.json and the enclosing result contract.
            save_repeat_clusters(fitted, centers, kmeans)
            restored = load_repeat_clusters(centers, "M11", kmeans, device=device)
            _require(np.array_equal(fitted.centroids, restored.centroids)
                     and np.array_equal(fitted.predict(features[:32]), restored.predict(features[:32])),
                     "Center save/load changed predictions")
            del fitted
            for sample_id in data.train_sample_ids:
                sample = samples[sample_id]
                with h5py.File(sample.path, "r") as handle:
                    output = LabelMap(handle["valid_spectrum_mask"][:], 8)
                count = 0
                for batch in AllPixelData((sample,)).all_batches(chunk_pixels=chunk_pixels):
                    transformed = _transform_batch(representation, batch, "M11")
                    output.add(batch.pixel_row_col, restored.predict(transformed.values))
                    count += len(batch.snv)
                _require(count == sample.saved_pixel_count, f"{sample_id}: incomplete prediction")
                np.savez_compressed(stage / "maps" / f"{sample_id}.npz", labels_k8=output.finish())
                print(f"encoder {encoder}, kmeans {kmeans}: {sample_id}", flush=True)
            _require(_cluster_contract(experiment, source, data, encoder) == contract,
                     "Clustering inputs changed during computation")
            _write_json(stage / "run.json", {
                "contract": {**contract, "kmeans_repeat": kmeans},
                "execution": runtime_record(device), "fit": asdict(restored.record),
                "chunk_pixels": chunk_pixels, "centers_sha256": _digest(centers),
                "center_roundtrip_exact": True, "wall_seconds": time.perf_counter() - started,
            })
            seal(stage, "spatial_clustering_completed")
            results.parent.mkdir(parents=True, exist_ok=True)
            weights.parent.mkdir(parents=True, exist_ok=True)
            weight_stage.rename(weights)
            try:
                stage.rename(results)
            except OSError:
                weights.rename(weight_stage)
                raise
        check_cluster(experiment, source, data, inventory, encoder, kmeans)
    check_prepared(experiment, source, data, inventory)


def _read_maps(
    experiment: Path, source: Path, sample: SampleInput,
) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray]:
    with h5py.File(sample.path, "r") as handle:
        mask = handle["valid_spectrum_mask"][:]
    _require(mask.shape == (sample.height, sample.width) and np.isin(mask, (0, 1)).all()
             and int(np.count_nonzero(mask)) == sample.saved_pixel_count, "Source mask differs")
    mask = mask.astype(bool)
    maps = {}
    for encoder, kmeans in MAP_IDS:
        path = map_path(experiment, source, sample.sample_id, encoder, kmeans)
        with np.load(path, allow_pickle=False) as saved:
            _require(set(saved.files) == {"labels_k8"}, "Unexpected label fields")
            labels = saved["labels_k8"].copy()
        _require(labels.dtype == np.uint8, "Expected saved uint8 labels")
        validate_map(labels, mask)
        maps[encoder, kmeans] = labels
    return maps, mask


def _evaluation_contract(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
    *, include_overlap: bool,
) -> dict:
    check_prepared(experiment, source, data, inventory)
    maps = []
    for encoder, kmeans in MAP_IDS:
        if encoder > 1:
            check_cluster(experiment, source, data, inventory, encoder, kmeans)
        result = map_path(experiment, source, data.train_sample_ids[0], encoder, kmeans).parent.parent
        maps.append({"encoder_repeat": encoder, "kmeans_repeat": kmeans,
                     "results": str(result.resolve()),
                     "completion_sha256": _digest(result / "completion.json")})
    return {
        "schema_version": 1, "scope": SCOPE,
        "protocol_sha256": _digest(experiment / "protocol/completion.json"),
        "sample_ids": list(data.train_sample_ids), "maps": maps, "pairs": pair_plan(),
        "windows": list(WINDOWS), "include_overlap": include_overlap,
        "code_sha256": _code_hashes((Path(__file__).name, "spatial_reproducibility_metrics.py",
                                     "spatial_reproducibility_reporting.py")),
        "numpy": np.__version__,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    _require(bool(rows), "Cannot write an empty table without a schema")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flat_pair_rows(evaluations: list[dict]) -> list[dict[str, object]]:
    rows = []
    for sample in evaluations:
        for pair in sample["pairs"]:
            identity = {key: pair[key] for key in (
                "encoder_a", "kmeans_a", "encoder_b", "kmeans_b", "group", "initial_reference")}
            for boundary in pair["boundaries"]:
                for direction in ("a_to_b", "b_to_a"):
                    rows.append({"sample_id": sample["sample_id"], **identity,
                                 "ari": pair["ari"]["value"],
                                 "ari_undefined_reason": pair["ari"]["undefined_reason"],
                                 "window": boundary["window"],
                                 "radius_pixels": boundary["radius_pixels"],
                                 "direction": direction, **boundary[direction],
                                 "lower": boundary["lower"], "upper": boundary["upper"]})
    return rows


def evaluate(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
    *, include_overlap: bool = False,
) -> dict[str, object]:
    """Evaluate every full sample; missing maps fail instead of becoming empty boundaries."""
    experiment, source = separate_roots(experiment, source)
    output = experiment / "results/evaluation"
    _require(not output.exists(), "Evaluation exists; use --resume to verify and skip")
    contract = _evaluation_contract(experiment, source, data, inventory, include_overlap=include_overlap)
    started = time.perf_counter()
    evaluations, diagnostics, map_records = [], [], []
    samples = {sample.sample_id: sample for sample in inventory.samples}
    with TemporaryDirectory(prefix=".spatial-evaluation-", dir=experiment) as temporary:
        stage = Path(temporary) / "evaluation"
        (stage / "samples").mkdir(parents=True)
        for sample_id in data.train_sample_ids:
            maps, mask = _read_maps(experiment, source, samples[sample_id])
            map_rows = []
            for (encoder, kmeans), labels in maps.items():
                row = {"sample_id": sample_id, "encoder_repeat": encoder,
                       "kmeans_repeat": kmeans, **map_diagnostics(labels, mask)}
                map_rows.append(row)
                diagnostics.append({key: json.dumps(value) if isinstance(value, list) else value
                                    for key, value in row.items()})
                path = map_path(experiment, source, sample_id, encoder, kmeans)
                map_records.append({"sample_id": sample_id, "encoder_repeat": encoder,
                                    "kmeans_repeat": kmeans, "path": str(path.resolve()),
                                    "sha256": _digest(path), "shape": list(labels.shape)})
            pairs = [{**pair, **compare_maps(
                maps[pair["encoder_a"], pair["kmeans_a"]],
                maps[pair["encoder_b"], pair["kmeans_b"]], mask,
                include_overlap=include_overlap)} for pair in pair_plan()]
            evaluation = {"sample_id": sample_id, "maps": map_rows, "pairs": pairs}
            evaluations.append(evaluation)
            _write_json(stage / "samples" / f"{sample_id}.json", evaluation)
            print(f"{sample_id}: evaluated all 36 map pairs", flush=True)
        summary = summarize_samples(evaluations)
        _write_json(stage / "summary.json", summary)
        _write_json(stage / "map_sources.json", {"maps": map_records})
        _write_csv(stage / "map_diagnostics.csv", diagnostics)
        flat = _flat_pair_rows(evaluations)
        _write_csv(stage / "pair_metrics.csv", flat)
        _write_csv(stage / "initial_reference_metrics.csv", [row for row in flat
                                                           if row["initial_reference"]])
        _require(_evaluation_contract(experiment, source, data, inventory,
                                      include_overlap=include_overlap) == contract,
                 "Evaluation inputs changed during computation")
        _write_json(stage / "run.json", {"contract": contract,
                    "execution": runtime_record(torch.device("cpu")),
                    "wall_seconds": time.perf_counter() - started})
        completion = seal(stage, "spatial_evaluation_completed")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output)
    return completion


def check_evaluation(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    output = experiment / "results/evaluation"
    include_overlap = _read_json(output / "run.json")["contract"]["include_overlap"]
    _require(type(include_overlap) is bool, "Invalid overlap option")
    contract = _evaluation_contract(experiment, source, data, inventory, include_overlap=include_overlap)
    completion = check_sealed(output, "spatial_evaluation_completed", contract)
    records = _read_json(output / "map_sources.json")["maps"]
    _require([(row["sample_id"], row["encoder_repeat"], row["kmeans_repeat"]) for row in records]
             == [(sample, e, k) for sample in data.train_sample_ids for e, k in MAP_IDS],
             "Evaluation map coverage differs")
    for row in records:
        path = map_path(experiment, source, row["sample_id"], row["encoder_repeat"],
                        row["kmeans_repeat"])
        _require(row["path"] == str(path.resolve()) and row["sha256"] == _digest(path),
                 "Evaluation map source changed")
    evaluations = [_read_json(output / "samples" / f"{sample_id}.json")
                   for sample_id in data.train_sample_ids]
    for sample_id, evaluation in zip(data.train_sample_ids, evaluations, strict=True):
        _require(evaluation["sample_id"] == sample_id
                 and [(row["encoder_repeat"], row["kmeans_repeat"])
                      for row in evaluation["maps"]] == list(MAP_IDS)
                 and [{key: row[key] for key in pair_plan()[0]}
                      for row in evaluation["pairs"]] == pair_plan(),
                 "Evaluation must contain all nine maps and all 36 planned pairs")
    _require(summarize_samples(evaluations) == _read_json(output / "summary.json"),
             "Evaluation summary differs from pair records")
    return completion


def _report_contract(experiment: Path, *, dpi: int) -> dict[str, object]:
    return {"schema_version": 1, "scope": SCOPE, "dpi": dpi,
            "evaluation_sha256": _digest(experiment / "results/evaluation/completion.json"),
            "matplotlib": version("matplotlib"), "numpy": np.__version__,
            "code_sha256": _code_hashes((Path(__file__).name,
                                          "spatial_reproducibility_reporting.py",
                                          "spatial_reproducibility_metrics.py"))}


def write_report(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory, *, dpi: int = 160,
) -> dict[str, object]:
    experiment, source = separate_roots(experiment, source)
    _require(type(dpi) is int and dpi > 0, "DPI must be a positive integer")
    output = experiment / "results/report"
    _require(not output.exists(), "Report exists; use --resume to verify and skip")
    check_evaluation(experiment, source, data, inventory)
    contract = _report_contract(experiment, dpi=dpi)
    evaluation_root = experiment / "results/evaluation"
    samples = {sample.sample_id: sample for sample in inventory.samples}
    with TemporaryDirectory(prefix=".spatial-report-", dir=experiment) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        for sample_id in data.train_sample_ids:
            maps, mask = _read_maps(experiment, source, samples[sample_id])
            evaluation = _read_json(evaluation_root / "samples" / f"{sample_id}.json")
            render_sample(sample_id, maps, mask, evaluation, stage / "samples" / sample_id, dpi=dpi)
            print(f"{sample_id}: spatial figures saved", flush=True)
        render_summary(_read_json(evaluation_root / "summary.json"), stage / "summary", dpi=dpi)
        check_evaluation(experiment, source, data, inventory)
        _require(_report_contract(experiment, dpi=dpi) == contract, "Report source changed")
        _write_json(stage / "run.json", {"contract": contract, "sample_ids": list(data.train_sample_ids),
                                        "execution": runtime_record(torch.device("cpu")),
                                        "label_colors": "raw IDs; no cross-map color alignment"})
        completion = seal(stage, "spatial_report_completed")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output)
    return completion


def check_report(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    check_evaluation(experiment, source, data, inventory)
    output = experiment / "results/report"
    run = _read_json(output / "run.json")
    completion = check_sealed(output, "spatial_report_completed",
                              _report_contract(experiment, dpi=run["contract"]["dpi"]))
    _require(run["sample_ids"] == list(data.train_sample_ids), "Report sample coverage differs")
    _require({path.name for path in (output / "samples").iterdir()}
             == set(data.train_sample_ids), "Report sample directories differ")
    expected = {"01_nine_maps.png", "02_boundary_overlays.png", "03_boundary_frequency.png",
                "boundary_frequency.npz",
                *(f"initial_vs_e{encoder}_k{kmeans}.png" for encoder, kmeans in MAP_IDS[1:])}
    for sample_id in data.train_sample_ids:
        _require({path.name for path in (output / "samples" / sample_id).iterdir()} == expected,
                 f"{sample_id}: report figure/frequency coverage differs")
    _require({path.name for path in (output / "summary").iterdir()} == {
        "01_ari_vs_boundary_coverage.png", "02_boundary_sample_distributions.png",
        "03_ari_sample_distribution.png"}, "Report summary figure coverage differs")
    return {"status": completion["status"], "samples": len(data.train_sample_ids),
            "maps_per_sample": 9, "pairs_per_sample": 36, "scientific_pass_threshold": None}
