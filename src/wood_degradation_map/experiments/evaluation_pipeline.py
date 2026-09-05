"""Evaluate completed clean-map runs with shared perturbations and fixed centers.

No fit, training, subset selection or CV aggregation occurs here. All selected
consumers receive the very same 15-input block. Only clean representations are
retained on CPU for pooled-fold silhouette; augmented spectra are not persisted.
"""

from __future__ import annotations

import hashlib
import sys
import time
from contextlib import closing
from dataclasses import asdict, dataclass, fields
from itertools import groupby, product
from pathlib import Path

import h5py
import numpy as np
import torch

from .aggregation import REPEATED_METRICS, ScoreRecord
from .cluster_pipeline import check_clustering, load_representation
from .clustering import FittedClusters, Representation, _dimension
from .config import CLUSTER_COUNTS, REPEATS, experiment_config
from .data import FoldData, SpectrumInputError
from .diagnostic_metrics import SampleFeatures, fold_silhouette
from .input_validation import InputInventory
from .lfr import LFRAccumulator, PixelLabels, accumulate_lfr_block
from .manifests import _digest, _read_json, _write_json
from .neural import fp32_inference
from .perturbations import DRAW_KEYS, SharedPerturbations
from .spatial_metrics import local_label_agreement
from .training import _code_hashes as training_code_hashes, runtime_record


def _paths(experiment: Path, condition: str, fold: int, repeat: int) -> tuple[Path, Path, Path]:
    suffix = f"{condition}/fold_{fold}/repeat_{repeat}"
    return (experiment / "results/evaluation" / suffix,
            experiment / "results/clustering" / suffix,
            experiment / "checkpoints/clustering" / suffix)


def _code_hashes() -> dict[str, str]:
    names = ("evaluation_pipeline", "perturbations", "lfr", "spatial_metrics",
             "diagnostic_metrics", "aggregation", "cluster_pipeline", "clustering")
    return {**training_code_hashes(),
            **{f"{name}.py": _digest(Path(__file__).with_name(f"{name}.py")) for name in names}}


@dataclass
class _Consumer:
    condition: str
    repeat: int
    results: Path
    maps: Path
    representation: Representation
    clusters: dict[int, FittedClusters]
    features: dict[str, np.ndarray]
    scores: list[ScoreRecord]
    sample_reports: list[dict[str, object]]


def _score(sample: str, data: FoldData, consumer: _Consumer, k: int,
           metric: str, value: float | None, reason: str | None = None) -> ScoreRecord:
    return ScoreRecord(sample, data.fold, consumer.condition, k, metric, consumer.repeat,
                       "undefined" if value is None else "defined", value, reason)


def run_evaluation(
    experiment: Path, data: FoldData, inventory: InputInventory,
    conditions: tuple[str, ...], repeats: tuple[int, ...], *, device: torch.device,
    chunk_pixels: int = 1024, silhouette_chunk_pixels: int = 1_000_000,
) -> tuple[Path, ...]:
    """Evaluate every test row for selected condition/repeat consumers, at all K.

    Production CLI requires CUDA; CPU is supported for small fixtures. All source
    runs must have complete production maps. A training/clustering smoke cannot
    enter this path. Existing output directories are never overwritten/resumed.

    CPU clean-feature storage is sum(N_test * dimension * 4) over consumers.
    Models/centers remain on the selected device. Silhouette additionally builds
    full-fold arrays, one consumer/K at a time; its chunk bounds only N x K tiles.
    There is no automatic subsampling, CPU fallback or batch-size adaptation.
    """
    if (not conditions or len(set(conditions)) != len(conditions)
            or not repeats or len(set(repeats)) != len(repeats)
            or any(type(repeat) is not int or repeat not in REPEATS for repeat in repeats)):
        raise ValueError("Expected nonempty unique planned conditions/repeats")
    for condition in conditions:
        _dimension(condition)
    if (device.type not in ("cpu", "cuda")
            or type(chunk_pixels) is not int or chunk_pixels < 1
            or type(silhouette_chunk_pixels) is not int or silhouette_chunk_pixels < 1):
        raise ValueError("Expected CPU/CUDA and positive integer chunk sizes")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    identities = tuple(product(conditions, repeats))
    # Validate every source before creating any output or generating perturbations.
    for condition, repeat in identities:
        results, _, _ = _paths(experiment, condition, data.fold, repeat)
        if results.exists():
            raise FileExistsError(f"Evaluation output already exists: {results}")
        check_clustering(experiment, data, inventory, condition, repeat, device=torch.device("cpu"))
    samples = {sample.sample_id: sample for sample in inventory.samples
               if sample.sample_id in data.test_sample_ids}
    expected = {sample: samples[sample].saved_pixel_count for sample in data.test_sample_ids}
    artifacts = _read_json(experiment / "manifests/complete.json")["artifact_sha256"]
    feature_bytes = sum(data.test_pixel_count * _dimension(condition) * 4
                        for condition, _ in identities)
    print(f"Evaluation: {len(identities)} consumers; {data.test_pixel_count} test rows; "
          f"{feature_bytes} bytes of retained CPU clean features", flush=True)
    started = time.perf_counter()
    consumers: list[_Consumer] = []
    created: list[Path] = []
    shared_inputs: list[dict[str, object]] = []
    try:
        with fp32_inference(device):
            evaluation_runtime = runtime_record(device)
        for condition, repeat in identities:
            results, maps, checkpoints = _paths(experiment, condition, data.fold, repeat)
            cluster_run = _read_json(maps / "run.json")
            representation, source = load_representation(
                experiment, data, condition, repeat, device=device,
                pca_repeat=cluster_run["source_pca_repeat"],
            )
            if source != cluster_run["source"]:
                raise ValueError("Evaluation representation differs from saved clean-map source")
            clusters = {k: FittedClusters.load(
                checkpoints / f"centers_k{k}.npz", condition_id=condition,
                fold=data.fold, repeat=repeat, k=k, device=device,
            ) for k in CLUSTER_COUNTS}
            results.mkdir(parents=True)
            created.append(results)
            (results / "samples").mkdir()
            (results / "silhouette").mkdir()
            _write_json(results / "run.json", {
                "schema_version": 1, "mode": "full_test_evaluation", "condition": condition,
                "fold": data.fold, "repeat": repeat, "config": experiment_config(),
                "test_sample_ids": list(data.test_sample_ids), "test_pixels": data.test_pixel_count,
                "manifest_artifact_sha256": artifacts, "code_sha256": _code_hashes(),
                "clustering_completion_sha256": _digest(maps / "completion.json"),
                "source": source, "runtime": evaluation_runtime,
                "shared_consumers": [{"condition": c, "repeat": r} for c, r in identities],
                "chunk_pixels": chunk_pixels, "silhouette_chunk_pixels": silhouette_chunk_pixels,
                "retained_cpu_feature_bytes_all_consumers": feature_bytes,
                "memory_scope": "excludes models, maps, coordinate arrays and silhouette workspace",
            })
            consumers.append(_Consumer(condition, repeat, results, maps, representation,
                                       clusters, {}, [], []))

        seen = []
        with closing(data.batches("test", chunk_pixels=chunk_pixels)) as source_batches:
            for sample_id, batches in groupby(source_batches, key=lambda batch: batch.sample_id):
                if sample_id not in expected or sample_id in seen:
                    raise ValueError("Unexpected/duplicate test sample in evaluation input")
                seen.append(sample_id)
                sample = samples[sample_id]
                with h5py.File(sample.path, "r") as handle:
                    coordinates = handle["pixel_row_col"][:]
                    valid = handle["valid_spectrum_mask"][:]
                maps_by_run = []
                accumulators = []
                for consumer in consumers:
                    with np.load(consumer.maps / "maps" / f"{sample_id}.npz", allow_pickle=False) as saved:
                        maps_by_run.append({k: saved[f"labels_k{k}"] for k in CLUSTER_COUNTS})
                    accumulators.append({k: LFRAccumulator(sample_id, expected[sample_id], k=k)
                                         for k in CLUSTER_COUNTS})
                    consumer.features[sample_id] = np.empty(
                        (expected[sample_id], _dimension(consumer.condition)), dtype=np.float32)
                generator = SharedPerturbations(sample_id, expected[sample_id], device=device)
                # Independent streams make fingerprints independent of loader chunk boundaries.
                clean_hash, coordinate_hash = hashlib.sha256(), hashlib.sha256()
                draw_hashes = {(item["kind"], item["draw"]): hashlib.sha256()
                               for item in generator.record()["draws"]}
                for block in generator.batches(batches):
                    rows, coords = block.clean.hdf5_rows, block.clean.pixel_row_col
                    if not np.array_equal(coords, coordinates[rows]):
                        raise ValueError("Evaluation source coordinates differ from saved HDF5 rows")
                    clean_hash.update(block.clean.snv.astype("<f4", copy=False).tobytes())
                    coordinate_hash.update(coords.astype("<i8", copy=False).tobytes())
                    for item in block.perturbed:
                        draw_hashes[item.kind, item.draw].update(item.batch.snv.astype("<f4", copy=False).tobytes())
                    for index, consumer in enumerate(consumers):
                        predicted = accumulate_lfr_block(
                            block, consumer.representation, consumer.clusters, accumulators[index])
                        for k in CLUSTER_COUNTS:
                            saved = maps_by_run[index][k][coords[:, 0], coords[:, 1]]
                            mismatch = predicted.clean[k].labels != saved
                            if mismatch.any():
                                raise SpectrumInputError(sample_id, rows[mismatch],
                                                         f"Clean labels changed: {consumer.condition}, "
                                                         f"repeat={consumer.repeat}, K={k}")
                        consumer.features[sample_id][rows] = predicted.values
                shared_inputs.append({
                    "generation": generator.record(), "clean_sha256": clean_hash.hexdigest(),
                    "coordinates_sha256": coordinate_hash.hexdigest(),
                    "draws_sha256": [{"kind": kind, "draw": draw, "sha256": digest.hexdigest()}
                                    for (kind, draw), digest in draw_hashes.items()],
                })
                for index, consumer in enumerate(consumers):
                    detail = {"sample_id": sample_id, "valid_pixels": expected[sample_id],
                              "spatial": {}, "lfr": {}}
                    for k in CLUSTER_COUNTS:
                        spatial = local_label_agreement(maps_by_run[index][k], valid, k=k)
                        lfr = accumulators[index][k].finish()
                        detail["spatial"][str(k)], detail["lfr"][str(k)] = asdict(spatial), asdict(lfr)
                        for window in spatial.windows:
                            consumer.scores.extend((
                                _score(sample_id, data, consumer, k, f"lla_{window.window}",
                                       window.lla, window.lla_undefined_reason),
                                _score(sample_id, data, consumer, k, f"adjusted_lla_{window.window}",
                                       window.adjusted_lla,
                                       ";".join(window.adjusted_undefined_reasons) or None),
                            ))
                        for kind, value in lfr.mean_by_kind.items():
                            consumer.scores.append(_score(sample_id, data, consumer, k, f"lfr_{kind}", value))
                    path = consumer.results / "samples" / f"{sample_id}.json"
                    _write_json(path, detail)
                    consumer.sample_reports.append({"sample_id": sample_id, "sha256": _digest(path)})
                print(f"{sample_id}: all valid pixels, 15 shared draws and all K completed", flush=True)
        if seen != list(data.test_sample_ids):
            raise ValueError("Incomplete test sample coverage; no evaluation completion")

        for consumer in consumers:
            silhouette_reports = []
            for k in CLUSTER_COUNTS:
                inputs = []
                for sample_id in data.test_sample_ids:
                    with h5py.File(samples[sample_id].path, "r") as handle:
                        coords = handle["pixel_row_col"][:]
                    with np.load(consumer.maps / "maps" / f"{sample_id}.npz", allow_pickle=False) as saved:
                        labels = saved[f"labels_k{k}"][coords[:, 0], coords[:, 1]]
                    pixels = PixelLabels(sample_id, np.arange(expected[sample_id]), coords, labels)
                    inputs.append(SampleFeatures(pixels, consumer.features[sample_id]))
                print(f"{consumer.condition}/repeat_{consumer.repeat}: pooled silhouette K={k}", flush=True)
                result = fold_silhouette(
                    tuple(inputs), expected_test_pixels=expected, condition_id=consumer.condition,
                    fold=data.fold, repeat=consumer.repeat, k=k, device=device,
                    chunk_pixels=silhouette_chunk_pixels,
                )
                # Avoid dataclasses.asdict copying the full per-pixel score array into JSON.
                detail = {field.name: getattr(result, field.name) for field in fields(result)
                          if field.name not in ("pixel_scores", "samples")}
                detail["samples"] = [asdict(sample) for sample in result.samples]
                detail["pixel_scores_sha256"] = None
                if result.pixel_scores is not None:
                    path = consumer.results / "silhouette" / f"pixels_k{k}.npy"
                    with path.open("xb") as handle:
                        np.save(handle, result.pixel_scores, allow_pickle=False)
                    detail["pixel_scores_sha256"] = _digest(path)
                for sample in result.samples:
                    consumer.scores.append(_score(sample.sample_id, data, consumer, k, "silhouette",
                                                  sample.mean, sample.undefined_reason))
                path = consumer.results / "silhouette" / f"k{k}.json"
                _write_json(path, detail)
                silhouette_reports.append({"k": k, "sha256": _digest(path)})
            consumer.features.clear()
            _write_json(consumer.results / "scores.json", {"records": [asdict(row) for row in consumer.scores]})
            _write_json(consumer.results / "shared_inputs.json", {"samples": shared_inputs})
            # Validate persisted coverage/values before publishing completion.
            _check_scores(consumer.results, data, consumer.condition, consumer.repeat)
            _write_json(consumer.results / "completion.json", {
                "status": "full_test_evaluation_completed", "checks_passed": True,
                "scope": "all test pixels and all K; no cross-fold/repeat aggregation or ARI",
                "run_sha256": _digest(consumer.results / "run.json"),
                "scores_sha256": _digest(consumer.results / "scores.json"),
                "shared_inputs_sha256": _digest(consumer.results / "shared_inputs.json"),
                "samples": consumer.sample_reports, "silhouette": silhouette_reports,
                "wall_seconds": time.perf_counter() - started,
                "timing_scope": "joint evaluation elapsed to this consumer's save; excludes source validation",
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            })
    finally:
        error = sys.exc_info()[1]
        for results in created:
            if (results / "completion.json").exists():
                continue
            failure = {"status": "failed_or_interrupted", "error_type": type(error).__name__,
                       "reason": str(error), "wall_seconds": time.perf_counter() - started}
            if isinstance(error, SpectrumInputError):
                failure.update(sample_id=error.sample_id, hdf5_rows=list(error.hdf5_rows), reason=error.reason)
            try:
                _write_json(results / "failure.json", failure)
            except OSError as logging_error:
                print(f"Could not save evaluation failure record: {logging_error}", file=sys.stderr)
    return tuple(consumer.results for consumer in consumers)


def _check_scores(results: Path, data: FoldData, condition: str, repeat: int) -> None:
    records = _read_json(results / "scores.json")["records"]
    expected = {(sample, k, metric) for sample in data.test_sample_ids
                for k in CLUSTER_COUNTS for metric in REPEATED_METRICS}
    seen = set()
    for item in records:
        row = ScoreRecord(**item)
        key = (row.sample_id, row.k, row.metric)
        if (key not in expected or key in seen or row.condition_id != condition
                or type(row.fold) is not int or row.fold != data.fold
                or type(row.repeat) is not int or row.repeat != repeat or type(row.k) is not int):
            raise ValueError("Evaluation score identity/coverage mismatch")
        seen.add(key)
        if row.status == "defined":
            lower = -np.inf if row.metric.startswith("adjusted_lla_") else -1 if row.metric == "silhouette" else 0
            if (type(row.value) not in (float, int) or not np.isfinite(row.value)
                    or abs(row.value) > np.finfo(np.float32).max or not lower <= row.value <= 1
                    or row.reason is not None):
                raise ValueError("Invalid defined evaluation score")
        elif row.status != "undefined" or row.value is not None or not isinstance(row.reason, str) or not row.reason.strip():
            raise ValueError("Completed evaluation requires defined/undefined scores with explicit reasons")
    if seen != expected:
        raise ValueError("Incomplete evaluation scores")


def check_evaluation(
    experiment: Path, data: FoldData, inventory: InputInventory, condition: str, repeat: int,
) -> dict[str, object]:
    """Check source binding, hashes and coverage without spectra extraction or metric recomputation."""
    check_clustering(experiment, data, inventory, condition, repeat, device=torch.device("cpu"))
    results, maps, _ = _paths(experiment, condition, data.fold, repeat)
    run, done = _read_json(results / "run.json"), _read_json(results / "completion.json")
    expected = {"schema_version": 1, "mode": "full_test_evaluation", "condition": condition,
                "fold": data.fold, "repeat": repeat, "config": experiment_config(),
                "test_sample_ids": list(data.test_sample_ids), "test_pixels": data.test_pixel_count,
                "manifest_artifact_sha256": _read_json(experiment / "manifests/complete.json")["artifact_sha256"],
                "code_sha256": _code_hashes(),
                "clustering_completion_sha256": _digest(maps / "completion.json"),
                "source": _read_json(maps / "run.json")["source"]}
    if (any(run.get(key) != value for key, value in expected.items())
            or done.get("status") != "full_test_evaluation_completed" or done.get("checks_passed") is not True
            or (results / "failure.json").exists()):
        raise ValueError("Evaluation completion/run/source mismatch")
    for name in ("run", "scores", "shared_inputs"):
        if done[f"{name}_sha256"] != _digest(results / f"{name}.json"):
            raise ValueError(f"Evaluation {name} hash mismatch")
    if ([row["sample_id"] for row in done["samples"]] != list(data.test_sample_ids)
            or [row["k"] for row in done["silhouette"]] != list(CLUSTER_COUNTS)):
        raise ValueError("Evaluation sample/K coverage mismatch")
    shared = _read_json(results / "shared_inputs.json")["samples"]
    if [row["generation"]["sample_id"] for row in shared] != list(data.test_sample_ids):
        raise ValueError("Shared input sample coverage mismatch")
    samples = {sample.sample_id: sample for sample in inventory.samples}
    for row in shared:
        sample_id = row["generation"]["sample_id"]
        expected_generation = SharedPerturbations(
            sample_id, samples[sample_id].saved_pixel_count,
            device=torch.device(run["runtime"]["device"]),
        ).record()
        draws = row["draws_sha256"]
        if (row["generation"] != expected_generation
                or [(item["kind"], item["draw"]) for item in draws] != list(DRAW_KEYS)):
            raise ValueError("Shared input generation/draw contract mismatch")
        digests = [row["clean_sha256"], row["coordinates_sha256"],
                   *[item["sha256"] for item in draws]]
        if any(not isinstance(value, str) or len(value) != 64
               or any(char not in "0123456789abcdef" for char in value) for value in digests):
            raise ValueError("Invalid shared input fingerprint")
    _check_scores(results, data, condition, repeat)
    for row in done["samples"]:
        if row["sha256"] != _digest(results / "samples" / f"{row['sample_id']}.json"):
            raise ValueError("Evaluation sample detail hash mismatch")
    for row in done["silhouette"]:
        path = results / "silhouette" / f"k{row['k']}.json"
        if row["sha256"] != _digest(path):
            raise ValueError("Silhouette detail hash mismatch")
        detail = _read_json(path)
        pixels = results / "silhouette" / f"pixels_k{row['k']}.npy"
        if detail["pixel_scores_sha256"] is None:
            if detail["undefined_reason"] is None or pixels.exists():
                raise ValueError("Undefined silhouette persistence mismatch")
        else:
            if detail["pixel_scores_sha256"] != _digest(pixels):
                raise ValueError("Silhouette pixel hash mismatch")
            values = np.load(pixels, mmap_mode="r", allow_pickle=False)
            if (values.dtype != np.float32 or values.shape != (data.test_pixel_count,)
                    or not np.isfinite(values).all() or np.any(np.abs(values) > 1)):
                raise ValueError("Invalid saved silhouette pixel scores")
            del values
    return {"status": "validated_existing_evaluation", "condition": condition, "fold": data.fold,
            "repeat": repeat, "sample_count": len(data.test_sample_ids),
            "test_pixels": data.test_pixel_count, "cluster_counts": list(CLUSTER_COUNTS),
            "shared_inputs_sha256": done["shared_inputs_sha256"]}


def check_evaluations(
    experiment: Path, data: FoldData, inventory: InputInventory,
    conditions: tuple[str, ...], repeats: tuple[int, ...],
) -> tuple[dict[str, object], ...]:
    """Also verify exact common inputs across independently evaluated consumers.

    Downstream comparisons must use this group check, not assume seed equality
    implies identical realized perturbations across devices/software/invocations.
    """
    if (not conditions or len(set(conditions)) != len(conditions)
            or not repeats or len(set(repeats)) != len(repeats)):
        raise ValueError("Expected nonempty unique conditions/repeats for the group check")
    reports = tuple(check_evaluation(experiment, data, inventory, condition, repeat)
                    for condition, repeat in product(conditions, repeats))
    if len({report["shared_inputs_sha256"] for report in reports}) != 1:
        raise ValueError("Evaluation consumers used different realized shared inputs or generation settings")
    return reports
