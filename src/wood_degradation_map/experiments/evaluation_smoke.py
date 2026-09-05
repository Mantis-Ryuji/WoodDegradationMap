"""Bounded synthetic GPU evaluation using saved preflight models and centers.

This is an engineering probe, never a partial test-fold evaluation. Synthetic
row indices satisfy the metric interfaces but do not identify real HDF5 rows.
The production evaluation and OOF readers cannot consume these outputs.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .cluster_pipeline import _simple_id, load_representation
from .clustering import FittedClusters, Representation, _dimension
from .config import CLUSTER_COUNTS, experiment_config, kmeans_seed
from .data import FoldData, SpectrumBatch
from .diagnostic_metrics import SampleFeatures, fold_silhouette
from .lfr import LFRAccumulator, PixelLabels, accumulate_lfr_block
from .manifests import _digest, _read_json, _write_json
from .neural import fp32_inference
from .perturbations import SharedPerturbations
from .spatial_metrics import local_label_agreement
from .training import _code_hashes as training_code_hashes, runtime_record

PROBE_ID = "synthetic-evaluation-probe"
PROBE_SEED = 20260906
PROBE_SHAPE = (41, 25)  # 1024 + 1: exercise a complete generation block and its tail.
SCOPE = "synthetic engineering probe; no real test evaluation, CV metrics or condition selection"


def _cluster_code_hashes() -> dict[str, str]:
    return {**training_code_hashes(), **{
        name: _digest(Path(__file__).with_name(name))
        for name in ("clustering.py", "cluster_pipeline.py")
    }}


def _load_smoke_source(
    experiment: Path, data: FoldData, condition: str, repeat: int, smoke_id: str,
    *, device: torch.device,
) -> tuple[Representation, dict[int, FittedClusters], dict[str, object]]:
    """Validate saved smoke provenance without fitting or iterating real spectra."""
    _dimension(condition)
    kmeans_seed(data.fold, repeat, CLUSTER_COUNTS[0])
    _simple_id(smoke_id)
    suffix = f"clustering_smoke/{smoke_id}/{condition}/fold_{data.fold}/repeat_{repeat}"
    results, checkpoints = experiment / "results" / suffix, experiment / "checkpoints" / suffix
    run, completion = _read_json(results / "run.json"), _read_json(results / "completion.json")
    expected = {
        "schema_version": 1, "mode": "smoke", "condition": condition,
        "fold": data.fold, "repeat": repeat, "config": experiment_config(),
        "cluster_counts": list(CLUSTER_COUNTS), "code_sha256": _cluster_code_hashes(),
        "manifest_artifact_sha256": _read_json(experiment / "manifests/complete.json")["artifact_sha256"],
        "train_sample_ids": list(data.train_sample_ids), "train_pixels": data.train_pixel_count,
        "test_sample_ids": list(data.test_sample_ids), "test_pixels": data.test_pixel_count,
    }
    if (any(run.get(key) != value for key, value in expected.items())
            or completion.get("status") != "clustering_smoke_completed"
            or completion.get("checks_passed") is not True
            or completion.get("centers_and_probe_labels_save_load_exact") is not True
            or completion.get("samples") != [] or (results / "failure.json").exists()
            or completion.get("run_sha256") != _digest(results / "run.json")
            or completion.get("fits_sha256") != _digest(results / "fits.json")):
        raise ValueError("Clustering smoke completion/run/manifest/hash mismatch")
    if condition not in ("B0", "B1") and not run.get("source_neural_smoke_id"):
        raise ValueError("Neural evaluation smoke requires an explicit saved neural smoke source")
    probe_rows = completion["probe_rows"]
    ids = tuple(dict.fromkeys(row["sample_id"] for row in probe_rows))
    identities = [(row["sample_id"], index) for row in probe_rows for index in row["hdf5_rows"]]
    if (not ids or not set(ids).issubset(data.train_sample_ids)
            or len(identities) != min(64, data.train_pixel_count)
            or len(set(identities)) != len(identities)
            or any(type(index) is not int or index < 0 for _, index in identities)):
        raise ValueError("Clustering smoke train probe provenance mismatch")
    representation, source = load_representation(
        experiment, data, condition, repeat, device=device,
        neural_smoke_id=run["source_neural_smoke_id"], pca_repeat=run["source_pca_repeat"],
    )
    if source != run["source"]:
        raise ValueError("Clustering smoke representation source mismatch")
    fits, clusters = _read_json(results / "fits.json"), {}
    if set(fits) != {str(k) for k in CLUSTER_COUNTS}:
        raise ValueError("Clustering smoke fit K coverage mismatch")
    for k in CLUSTER_COUNTS:
        path = checkpoints / f"centers_k{k}.npz"
        if completion["centers_sha256"].get(str(k)) != _digest(path):
            raise ValueError("Clustering smoke center hash mismatch")
        cluster = FittedClusters.load(path, condition_id=condition, fold=data.fold,
                                      repeat=repeat, k=k, device=device)
        if (cluster.record.train_sample_ids != ids
                or cluster.record.train_pixels != len(identities)
                or json.loads(json.dumps(asdict(cluster.record))) != fits[str(k)]):
            raise ValueError("Clustering smoke center fit provenance mismatch")
        clusters[k] = cluster
    return representation, clusters, {
        "clustering_smoke_id": smoke_id, "clustering_completion_sha256": _digest(results / "completion.json"),
        "clustering_run_sha256": completion["run_sha256"],
        "clustering_fits_sha256": completion["fits_sha256"],
        "centers_sha256": completion["centers_sha256"], "representation": source,
        "neural_smoke_id": run["source_neural_smoke_id"],
        "manifest_artifact_sha256": expected["manifest_artifact_sha256"],
    }


def _synthetic_input() -> SpectrumBatch:
    n = PROBE_SHAPE[0] * PROBE_SHAPE[1]
    values = np.random.default_rng(PROBE_SEED).standard_normal((n, 256), dtype=np.float32)
    values -= values.mean(axis=1, keepdims=True)
    values /= values.std(axis=1, ddof=1, keepdims=True)
    rows = np.arange(n, dtype=np.int64)
    coordinates = np.column_stack((rows // PROBE_SHAPE[1], rows % PROBE_SHAPE[1]))
    return SpectrumBatch(PROBE_ID, rows, coordinates, values)


def _silhouette_kernel_probe(device: torch.device) -> dict[str, object]:
    # Source predictions can legitimately collapse to one cluster. This separate
    # known geometry still exercises the reference silhouette kernel on device.
    values = np.zeros((4, 256), dtype=np.float32)
    values[:2, 0], values[2:, 1] = 1, 1
    pixels = PixelLabels("synthetic-silhouette-oracle", np.arange(4),
                         np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([1, 1, 2, 2]))
    result = fold_silhouette(
        (SampleFeatures(pixels, values),), expected_test_pixels={pixels.sample_id: 4},
        condition_id="B0", fold=1, repeat=1, k=2, device=device, chunk_pixels=3,
    )
    if result.pixel_scores is None or not np.allclose(result.pixel_scores, 1, rtol=0, atol=1e-6):
        raise ValueError("Known-geometry silhouette kernel probe failed")
    return {"expected_scores": [1.0] * 4, "actual_scores": result.pixel_scores.tolist(),
            "device": str(device), "absolute_tolerance": 1e-6, "checks_passed": True}


def run_evaluation_smoke(
    experiment: Path, data: FoldData, condition: str, repeat: int, *,
    clustering_smoke_id: str, device: torch.device,
) -> Path:
    """Evaluate a fixed 1025-row synthetic fixture; CPU is only for unit tests.

    Load the explicitly identified completed clustering smoke and its original
    representation. Do not fit, read FoldData batches, or publish production
    evaluation records. Each clean/perturbed representation is reused across K.
    Timing and peak allocation describe this bounded probe, not full-fold load.
    """
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Expected CPU fixture or single CUDA device")
    started = time.perf_counter()
    representation, clusters, source = _load_smoke_source(
        experiment, data, condition, repeat, clustering_smoke_id, device=device,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    results = experiment / f"results/evaluation_smoke/{stamp}/{condition}/fold_{data.fold}/repeat_{repeat}"
    results.mkdir(parents=True, exist_ok=False)
    print(f"Results: {results}", flush=True)
    completed = False
    try:
        batch = _synthetic_input()
        n = len(batch.snv)
        generator = SharedPerturbations(PROBE_ID, n, device=device)
        with fp32_inference(device):
            runtime = runtime_record(device)
        code = {**_cluster_code_hashes(), **{
            name: _digest(Path(__file__).with_name(name)) for name in
            ("evaluation_smoke.py", "perturbations.py", "lfr.py", "spatial_metrics.py", "diagnostic_metrics.py")
        }}
        perturbations = generator.record()
        perturbations["row_order"] = "synthetic fixture indices 0..1024; not real HDF5 rows"
        _write_json(results / "run.json", {
            "schema_version": 1, "mode": "synthetic_evaluation_smoke", "scope": SCOPE,
            "condition": condition, "fold": data.fold, "repeat": repeat, "source": source,
            "config": experiment_config(), "code_sha256": code, "runtime": runtime,
            "fixture": {"id": PROBE_ID, "seed": PROBE_SEED, "shape": PROBE_SHAPE,
                        "rows": n, "spectrum": "Gaussian FP32, row-centered, ddof=1 SNV"},
            "perturbations": perturbations,
        })
        accumulators = {k: LFRAccumulator(PROBE_ID, n, k=k) for k in CLUSTER_COUNTS}
        values = np.empty((n, _dimension(condition)), dtype=np.float32)
        labels = {k: np.empty(n, dtype=np.int64) for k in CLUSTER_COUNTS}
        block_sizes = []
        centers_before = {k: cluster.centroids for k, cluster in clusters.items()}
        for block in generator.batches((batch,)):
            prediction = accumulate_lfr_block(block, representation, clusters, accumulators)
            rows = block.clean.hdf5_rows
            values[rows] = prediction.values
            for k in CLUSTER_COUNTS:
                labels[k][rows] = prediction.clean[k].labels
            block_sizes.append(len(rows))
            print(f"Synthetic LFR: {sum(block_sizes)}/{n} rows, all 15 draws and all K", flush=True)
        if block_sizes != [1024, 1]:
            raise ValueError("Synthetic evaluation generation coverage mismatch")
        arrays = {"snv": batch.snv, "fixture_rows": batch.hdf5_rows,
                  "pixel_row_col": batch.pixel_row_col, "clean_features": values}
        scores = {}
        for k in CLUSTER_COUNTS:
            pixels = PixelLabels(PROBE_ID, batch.hdf5_rows, batch.pixel_row_col, labels[k])
            silhouette = fold_silhouette(
                (SampleFeatures(pixels, values),), expected_test_pixels={PROBE_ID: n},
                condition_id=condition, fold=data.fold, repeat=repeat, k=k, device=device,
                chunk_pixels=1024,
            )
            summary = asdict(silhouette)
            summary.pop("pixel_scores")
            if silhouette.pixel_scores is not None:
                arrays[f"silhouette_k{k}"] = silhouette.pixel_scores
            label_map = labels[k].reshape(PROBE_SHAPE)
            arrays[f"labels_k{k}"] = label_map
            scores[str(k)] = {
                "lfr": asdict(accumulators[k].finish()), "silhouette": summary,
                "spatial": asdict(local_label_agreement(label_map, np.ones(PROBE_SHAPE, dtype=bool), k=k)),
            }
        metrics = {"scope": SCOPE, "scores": scores, "silhouette_kernel_probe": _silhouette_kernel_probe(device)}
        _write_json(results / "metrics.json", metrics)
        if _read_json(results / "metrics.json") != json.loads(json.dumps(metrics, allow_nan=False)):
            raise ValueError("Synthetic metric JSON save/load mismatch")
        with (results / "probe.npz").open("xb") as destination:
            np.savez_compressed(destination, **arrays)
        with np.load(results / "probe.npz", allow_pickle=False) as saved:
            if (set(saved.files) != set(arrays) or any(
                    saved[key].dtype != value.dtype or not np.array_equal(saved[key], value)
                    for key, value in arrays.items())):
                raise ValueError("Synthetic arrays save/load mismatch")
            for k, cluster in clusters.items():
                if (not np.array_equal(cluster.centroids, centers_before[k])
                        or not np.array_equal(cluster.predict(saved["clean_features"]) + 1,
                                              saved[f"labels_k{k}"].ravel())):
                    raise ValueError("Synthetic saved features/labels or fixed centers mismatch")
        report = {
            "status": "evaluation_smoke_completed", "scope": SCOPE, "checks_passed": True,
            "synthetic_rows": n, "generation_block_sizes": block_sizes,
            "cluster_counts": list(CLUSTER_COUNTS), "source": source,
            "arrays_and_metrics_save_load_exact": True, "fixed_centers_unchanged": True,
            "saved_clean_labels_reproduced": True,
            "silhouette_kernel_probe": metrics["silhouette_kernel_probe"],
            "artifact_sha256": {name: _digest(results / name) for name in ("run.json", "metrics.json", "probe.npz")},
            "wall_seconds": time.perf_counter() - started,
            "timing_scope": "source load through synthetic evaluation and save/reload; excludes initial manifest validation",
            "output_bytes_before_completion": sum(path.stat().st_size for path in results.iterdir() if path.is_file()),
        }
        if device.type == "cuda":
            report.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device))
        _write_json(results / "completion.json", report)
        completed = True
    finally:
        if not completed:
            error = sys.exc_info()[1]
            try:
                _write_json(results / "failure.json", {
                    "status": "failed_or_interrupted", "scope": SCOPE,
                    "error_type": type(error).__name__, "reason": str(error),
                    "rule": "Partial outputs are not complete; rerun creates a separate smoke",
                })
            except OSError as logging_error:
                print(f"Could not save failure record: {logging_error}", file=sys.stderr)
    return results
