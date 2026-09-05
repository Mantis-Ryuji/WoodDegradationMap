"""Connect verified representations to fixed-center clustering and clean label maps."""

from __future__ import annotations

import sys
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from chemomae.models.chemo_mae import ChemoMAE

from .baselines import B0Baseline, NormalizedRepresentation, PCABaseline, RepresentationError
from .clustering import (
    FittedClusters, LabelMap, Representation, TrainFeatures, _diagnose, _dimension,
    collect_train_features, fit_clusters,
)
from .config import CLUSTER_COUNTS, experiment_config, kmeans_seed, run_seed
from .data import FoldData, SpectrumBatch, SpectrumInputError
from .input_validation import InputInventory
from .manifests import _digest, _read_json, _write_json
from .neural import build_model, extract_full_visible, fp32_inference
from .training import _code_hashes, runtime_record


def _simple_id(value: str) -> str:
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                        for c in value):
        raise ValueError("Expected a simple run ID containing letters, digits, _ or -")
    return value


@dataclass(frozen=True)
class NeuralRepresentation:
    model: ChemoMAE
    device: torch.device

    def transform(self, spectra: np.ndarray) -> NormalizedRepresentation:
        if (spectra.dtype != np.float32 or spectra.ndim != 2 or spectra.shape[1] != 256
                or len(spectra) == 0):
            raise ValueError("Neural extraction requires nonempty FP32 N x 256 spectra")
        # Preserve the caller's clean SNV and use the verified all-visible encoder path.
        return extract_full_visible(
            self.model, torch.from_numpy(np.ascontiguousarray(spectra)).to(self.device),
        )


def load_representation(
    experiment: Path, data: FoldData, condition: str, repeat: int, *, device: torch.device,
    neural_smoke_id: str | None = None, pca_repeat: int | None = None,
) -> tuple[Representation, dict[str, object]]:
    """Load an explicitly identified source; never fit PCA or select a checkpoint."""
    _dimension(condition)
    kmeans_seed(data.fold, repeat, CLUSTER_COUNTS[0])
    if condition != "B1" and pca_repeat is not None:
        raise ValueError("pca_repeat is only valid for B1")
    if condition in ("B0", "B1") and neural_smoke_id is not None:
        raise ValueError("A neural smoke ID is only valid for neural conditions")
    artifacts = _read_json(experiment / "manifests/complete.json")["artifact_sha256"]
    if condition == "B0":
        return B0Baseline(), {"kind": "B0", "fit_required": False}
    if condition == "B1":
        source_repeat = repeat if pca_repeat is None else pca_repeat
        kmeans_seed(data.fold, source_repeat, CLUSTER_COUNTS[0])
        suffix = f"baselines/fold_{data.fold}/repeat_{source_repeat}"
        weights = experiment / "checkpoints" / suffix / "pca.npz"
        report_path = experiment / "results" / suffix / "fit.json"
        report = _read_json(report_path)
        weight_hash = _digest(weights)
        if (report["status"] != "fitted_and_roundtrip_checked"
                or report["manifest_artifact_sha256"] != artifacts
                or report["pca_checkpoint_sha256"] != weight_hash):
            raise ValueError("PCA source manifest/weights/completion mismatch")
        pca = PCABaseline.load(weights, fold=data.fold, repeat=repeat)
        if (pca.record.repeat != source_repeat or pca.record.sample_ids != data.train_sample_ids
                or pca.record.train_pixel_count != data.train_pixel_count):
            raise ValueError("PCA source differs from the shared train selection")
        return pca, {"kind": "B1", "source_repeat": source_repeat,
                     "weights_sha256": weight_hash, "fit_record_sha256": _digest(report_path)}

    branch = "neural" if neural_smoke_id is None else f"neural_smoke/{_simple_id(neural_smoke_id)}"
    suffix = f"{branch}/{condition}/fold_{data.fold}/repeat_{repeat}"
    results = experiment / "results" / suffix
    filename = "last_model.pt" if neural_smoke_id is None else "smoke_model.pt"
    weights = experiment / "checkpoints" / suffix / filename
    run = _read_json(results / "run.json")
    completion = _read_json(results / "completion.json")
    config = experiment_config()
    expected = {
        "schema_version": 1, "condition": condition, "fold": data.fold, "repeat": repeat,
        "mode": "training" if neural_smoke_id is None else "smoke",
        "smoke_id": neural_smoke_id, "config": config,
        "train_sample_ids": list(data.train_sample_ids), "train_pixels": data.train_pixel_count,
        "manifest_artifact_sha256": artifacts, "code_sha256": _code_hashes(),
        "batch_size": config["training"]["batch_size"],
        "seeds": {purpose: run_seed(purpose, data.fold, repeat)
                  for purpose in ("model_init", "pixel_order", "mask", "train_aug")},
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("Neural source run/config/manifest/code mismatch")
    current_runtime = runtime_record(device)
    if any(run["runtime"][key] != current_runtime[key] for key in ("torch", "chemomae")):
        raise ValueError("Neural source library version mismatch")
    steps = data.train_pixel_count // config["training"]["batch_size"]
    epochs = config["training"]["epochs"] if neural_smoke_id is None else 2
    batches = steps if neural_smoke_id is None else run["smoke_batches_per_epoch"]
    if (type(batches) is not int or not 1 <= batches <= steps
            or run["steps_per_full_epoch"] != steps
            or run["planned_production_updates"] != steps * config["training"]["epochs"]
            or (neural_smoke_id is None and run["smoke_batches_per_epoch"] is not None)
            or completion["status"] != ("training_completed" if neural_smoke_id is None
                                         else "smoke_fit_completed")
            or completion["completed_epochs"] != epochs
            or completion["attempted_updates"] != epochs * batches):
        raise ValueError("Neural source has not completed its declared training budget")
    if not (0 <= completion["nonzero_lr_updates"] <= completion["optimizer_updates"]
            <= completion["attempted_updates"]
            and completion["amp_skips"] == completion["attempted_updates"]
            - completion["optimizer_updates"]):
        raise ValueError("Invalid neural source update counters")
    weight_hash = _digest(weights)
    if completion["weights_sha256"] != weight_hash:
        raise ValueError("Neural source weights hash mismatch")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                   or not torch.isfinite(value).all() for value in state.values())):
        raise ValueError("Expected finite raw FP32 model weights")
    model = build_model(condition, data.fold, repeat)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return NeuralRepresentation(model, device), {
        "kind": "neural", "mode": run["mode"], "smoke_id": neural_smoke_id,
        "weights_sha256": weight_hash, "run_sha256": _digest(results / "run.json"),
        "completion_sha256": _digest(results / "completion.json"),
        "training_runtime": run["runtime"], "completed_epochs": epochs,
    }


def _transform_batch(
    representation: Representation, batch: SpectrumBatch, condition: str,
) -> NormalizedRepresentation:
    try:
        result = representation.transform(batch.snv)
        if len(result.values) != len(batch.snv):
            raise ValueError("Transform changed the number of pixels")
        _diagnose(result.values, _dimension(condition))
        return result
    except RepresentationError as error:
        diagnostic = error.diagnostics
        bad = sorted(set(diagnostic.nonfinite_rows + diagnostic.nonfinite_norm_rows
                         + diagnostic.zero_norm_rows + diagnostic.epsilon_clamped_rows))
        raise SpectrumInputError(batch.sample_id, batch.hdf5_rows[bad], str(error)) from error


def _fit_and_reload(
    features: TrainFeatures, checkpoints: Path, device: torch.device,
) -> dict[int, FittedClusters]:
    clusters = {}
    for k in CLUSTER_COUNTS:
        print(f"K={k}: fitting {len(features.values)} train representations", flush=True)
        fitted = fit_clusters(features, k, device=device)
        path = checkpoints / f"centers_k{k}.npz"
        fitted.save(path)
        restored = FittedClusters.load(
            path, condition_id=features.condition_id, fold=features.fold,
            repeat=features.repeat, k=k, device=device,
        )
        probe = features.values[:8]
        if (not np.array_equal(fitted.centroids, restored.centroids)
                or not np.array_equal(fitted.predict(probe), restored.predict(probe))):
            raise ValueError(f"K={k}: center save/load changed values or probe labels")
        clusters[k] = restored
    return clusters


def _predict_maps(
    data: FoldData, inventory: InputInventory, representation: Representation,
    clusters: dict[int, FittedClusters], results: Path, condition: str, chunk_pixels: int,
) -> list[dict[str, object]]:
    """Transform each held-out chunk once, keeping maps for only one sample at a time."""
    samples = {sample.sample_id: sample for sample in inventory.samples}
    reports: list[dict[str, object]] = []
    current_id: str | None = None
    maps: dict[int, LabelMap] = {}
    row_count = 0
    norm_error = 0.0

    def finish_sample() -> None:
        if current_id is None:
            return
        sample = samples[current_id]
        if row_count != sample.saved_pixel_count:
            raise ValueError(f"{current_id}: test row count differs from the input inventory")
        arrays = {f"labels_k{k}": output.finish() for k, output in maps.items()}
        path = results / "maps" / f"{current_id}.npz"
        with path.open("xb") as destination:
            np.savez_compressed(destination, **arrays)
        reports.append({
            "sample_id": current_id, "pixels": row_count, "shape": [sample.height, sample.width],
            "unit_norm_absolute_error_max": norm_error,
            "map_file": f"maps/{current_id}.npz", "map_sha256": _digest(path),
            "occupancy": {str(k): np.bincount(arrays[f"labels_k{k}"].ravel(),
                                              minlength=k + 1)[1:].tolist() for k in clusters},
        })
        print(f"{current_id}: saved {row_count} clean test labels for all K", flush=True)

    for batch in data.batches("test", chunk_pixels=chunk_pixels):
        if batch.sample_id != current_id:
            finish_sample()
            if batch.sample_id in {row["sample_id"] for row in reports}:
                raise ValueError("Test samples must appear in contiguous chunks")
            current_id = batch.sample_id
            if current_id not in data.test_sample_ids:
                raise ValueError("Loader returned a non-test sample")
            with h5py.File(samples[current_id].path, "r") as handle:
                valid = handle["valid_spectrum_mask"][:]
            maps = {k: LabelMap(valid, k) for k in clusters}
            row_count, norm_error = 0, 0.0
        transformed = _transform_batch(representation, batch, condition)
        diagnostic = _diagnose(transformed.values, _dimension(condition))
        norm_error = max(norm_error, diagnostic.unit_norm_absolute_error_max)
        for k, cluster in clusters.items():
            maps[k].add(batch.pixel_row_col, cluster.predict(transformed.values))
        row_count += len(batch.snv)
    finish_sample()
    if (tuple(row["sample_id"] for row in reports) != data.test_sample_ids
            or sum(row["pixels"] for row in reports) != data.test_pixel_count):
        raise ValueError("Test samples/pixels differ from the saved fold")
    return reports


def _probe_features(
    data: FoldData, representation: Representation, condition: str, repeat: int,
) -> tuple[TrainFeatures, list[dict[str, object]]]:
    # Engineering probe only: first 64 shared train rows in canonical order.
    # No test spectra or full train matrix are loaded for this CUDA check.
    values, rows = [], []
    remaining = 64
    with closing(data.batches("train", chunk_pixels=64)) as batches:
        for batch in batches:
            take = min(remaining, len(batch.snv))
            probe = SpectrumBatch(batch.sample_id, batch.hdf5_rows[:take],
                                  batch.pixel_row_col[:take], batch.snv[:take])
            values.append(_transform_batch(representation, probe, condition).values)
            rows.append({"sample_id": probe.sample_id, "hdf5_rows": probe.hdf5_rows.tolist()})
            remaining -= take
            if remaining == 0:
                break
    if not values or sum(len(value) for value in values) < max(CLUSTER_COUNTS):
        raise ValueError("Insufficient shared train rows for the clustering probe")
    ids = tuple(dict.fromkeys(row["sample_id"] for row in rows))
    return TrainFeatures(condition, data.fold, repeat, ids, np.concatenate(values)), rows


def run_clustering(
    experiment: Path, data: FoldData, inventory: InputInventory, condition: str, repeat: int,
    *, device: torch.device, smoke: bool = False, neural_smoke_id: str | None = None,
    pca_repeat: int | None = None, chunk_pixels: int = 1024,
) -> Path:
    """Create one exclusive run; partial outputs are never resumed or overwritten."""
    started = time.perf_counter()
    if type(chunk_pixels) is not int or chunk_pixels <= 0:
        raise ValueError("chunk_pixels must be a positive integer")
    if neural_smoke_id is not None and not smoke:
        raise ValueError("Smoke weights cannot generate production maps")
    _dimension(condition)
    kmeans_seed(data.fold, repeat, CLUSTER_COUNTS[0])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    branch = f"clustering_smoke/{stamp}" if smoke else "clustering"
    suffix = f"{branch}/{condition}/fold_{data.fold}/repeat_{repeat}"
    results, checkpoints = experiment / "results" / suffix, experiment / "checkpoints" / suffix
    if results.exists() or checkpoints.exists():
        raise FileExistsError("Clustering output exists; check it instead of overwriting/refitting")
    representation, source = load_representation(
        experiment, data, condition, repeat, device=device, neural_smoke_id=neural_smoke_id,
        pca_repeat=pca_repeat,
    )
    with fp32_inference(device):
        runtime = runtime_record(device)
    record = {
        "schema_version": 1, "mode": "smoke" if smoke else "clean_test_maps",
        "condition": condition, "fold": data.fold, "repeat": repeat,
        "cluster_counts": list(CLUSTER_COUNTS), "source": source,
        "source_pca_repeat": pca_repeat, "source_neural_smoke_id": neural_smoke_id,
        "train_sample_ids": list(data.train_sample_ids), "train_pixels": data.train_pixel_count,
        "test_sample_ids": list(data.test_sample_ids), "test_pixels": data.test_pixel_count,
        "manifest_artifact_sha256": _read_json(experiment / "manifests/complete.json")["artifact_sha256"],
        "config": experiment_config(), "runtime": runtime, "chunk_pixels": chunk_pixels,
        "code_sha256": {**_code_hashes(), **{name: _digest(Path(__file__).with_name(name))
                                            for name in ("clustering.py", "cluster_pipeline.py")}},
        "label_convention": "background=0; clusters=1..K; no cross-K/fold label alignment",
        "scope": "train-only bounded probe; no CV metrics" if smoke else "clean test maps; no metrics",
    }
    results.mkdir(parents=True, exist_ok=False)
    checkpoints.mkdir(parents=True, exist_ok=False)
    _write_json(results / "run.json", record)
    completed = False
    print(f"Results: {results}", flush=True)
    try:
        if smoke:
            features, probe_rows = _probe_features(data, representation, condition, repeat)
        else:
            print(f"Extracting {data.train_pixel_count} shared train rows once", flush=True)
            features = collect_train_features(data, representation, condition_id=condition,
                                               repeat=repeat, chunk_pixels=chunk_pixels)
            probe_rows = []
        clusters = _fit_and_reload(features, checkpoints, device)
        fit_records = {str(k): asdict(cluster.record) for k, cluster in clusters.items()}
        del features
        _write_json(results / "fits.json", fit_records)
        samples = []
        if not smoke:
            (results / "maps").mkdir()
            samples = _predict_maps(data, inventory, representation, clusters, results,
                                    condition, chunk_pixels)
        report = {
            "status": "clustering_smoke_completed" if smoke else "clean_test_maps_completed",
            "scope": record["scope"], "checks_passed": True,
            "centers_and_probe_labels_save_load_exact": True,
            "probe_rows": probe_rows, "samples": samples,
            "run_sha256": _digest(results / "run.json"),
            "fits_sha256": _digest(results / "fits.json"),
            "centers_sha256": {str(k): _digest(checkpoints / f"centers_k{k}.npz") for k in clusters},
            "wall_seconds": time.perf_counter() - started,
            "timing_scope": "source load through output save; excludes initial manifest validation",
        }
        if device.type == "cuda":
            report.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device))
        _write_json(results / "completion.json", report)
        completed = True
    finally:
        if not completed:
            error = sys.exc_info()[1]
            failure = {"status": "failed_or_interrupted", "error_type": type(error).__name__,
                       "wall_seconds": time.perf_counter() - started,
                       "rule": "Partial outputs are not complete and are not automatically resumed"}
            if isinstance(error, SpectrumInputError):
                failure.update(sample_id=error.sample_id, hdf5_rows=list(error.hdf5_rows),
                               reason=error.reason)
            elif isinstance(error, RepresentationError):
                failure["diagnostics"] = asdict(error.diagnostics)
            try:
                _write_json(results / "failure.json", failure)
            except OSError as logging_error:
                print(f"Could not save failure record: {logging_error}", file=sys.stderr)
    return results


def check_clustering(
    experiment: Path, data: FoldData, inventory: InputInventory, condition: str, repeat: int,
    *, device: torch.device,
) -> dict[str, object]:
    """Check completed production artifacts without fitting or extracting spectra."""
    _dimension(condition)
    kmeans_seed(data.fold, repeat, CLUSTER_COUNTS[0])
    suffix = f"clustering/{condition}/fold_{data.fold}/repeat_{repeat}"
    results, checkpoints = experiment / "results" / suffix, experiment / "checkpoints" / suffix
    run = _read_json(results / "run.json")
    completion = _read_json(results / "completion.json")
    artifacts = _read_json(experiment / "manifests/complete.json")["artifact_sha256"]
    expected = {"schema_version": 1, "mode": "clean_test_maps", "condition": condition,
                "fold": data.fold, "repeat": repeat, "config": experiment_config(),
                "cluster_counts": list(CLUSTER_COUNTS), "manifest_artifact_sha256": artifacts,
                "train_sample_ids": list(data.train_sample_ids), "train_pixels": data.train_pixel_count,
                "test_sample_ids": list(data.test_sample_ids), "test_pixels": data.test_pixel_count}
    if (any(run.get(key) != value for key, value in expected.items())
            or run["source_neural_smoke_id"] is not None
            or completion["status"] != "clean_test_maps_completed"
            or not completion["checks_passed"]
            or completion["run_sha256"] != _digest(results / "run.json")
            or completion["fits_sha256"] != _digest(results / "fits.json")):
        raise ValueError("Clustering completion/run/manifest mismatch")
    representation, source = load_representation(
        experiment, data, condition, repeat, device=device, pca_repeat=run["source_pca_repeat"],
    )
    del representation
    if source != run["source"]:
        raise ValueError("Clustering representation source mismatch")
    fits = _read_json(results / "fits.json")
    for k in CLUSTER_COUNTS:
        path = checkpoints / f"centers_k{k}.npz"
        if completion["centers_sha256"][str(k)] != _digest(path):
            raise ValueError("Saved center hash mismatch")
        cluster = FittedClusters.load(path, condition_id=condition, fold=data.fold,
                                      repeat=repeat, k=k, device=device)
        if (cluster.record.train_sample_ids != data.train_sample_ids
                or cluster.record.train_pixels != data.train_pixel_count
                or fits[str(k)]["seed"] != cluster.record.seed):
            raise ValueError("Saved center provenance mismatch")
    reports = completion["samples"]
    if [row["sample_id"] for row in reports] != list(data.test_sample_ids):
        raise ValueError("Saved map sample coverage mismatch")
    samples = {sample.sample_id: sample for sample in inventory.samples}
    for report in reports:
        sample = samples[report["sample_id"]]
        path = results / "maps" / f"{sample.sample_id}.npz"
        if report["map_sha256"] != _digest(path) or report["pixels"] != sample.saved_pixel_count:
            raise ValueError("Saved map hash/pixel count mismatch")
        with h5py.File(sample.path, "r") as handle:
            valid = handle["valid_spectrum_mask"][:] == 1
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {f"labels_k{k}" for k in CLUSTER_COUNTS}:
                raise ValueError("Saved map K coverage mismatch")
            for k in CLUSTER_COUNTS:
                labels = saved[f"labels_k{k}"]
                if (labels.dtype != np.uint8 or labels.shape != (sample.height, sample.width)
                        or np.any(labels[~valid] != 0) or np.any(labels[valid] == 0)
                        or np.any(labels > k) or int(valid.sum()) != sample.saved_pixel_count
                        or np.bincount(labels.ravel(), minlength=k + 1)[1:].tolist()
                        != report["occupancy"][str(k)]):
                    raise ValueError("Saved map labels/coverage/occupancy mismatch")
    return {"status": "validated_existing_clustering", "sample_count": len(reports),
            "test_pixels": data.test_pixel_count, "cluster_counts": list(CLUSTER_COUNTS)}
