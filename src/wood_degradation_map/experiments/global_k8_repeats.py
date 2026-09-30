"""Three isolated K=8 KMeans fits on the saved global representations."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans

from .cluster_pipeline import _transform_batch
from .clustering import LabelMap, _diagnose, _dimension, cluster_contract
from .config import REPEATS, experiment_config
from .global_clustering import (
    K, AllPixelData, GlobalClusterRecord, GlobalClusters, clustering_contract,
    collect_global_features, load_global_representation, read_label_map, source_record,
)
from .global_manifest import GLOBAL_CONDITIONS, GlobalData
from .global_seed_plan import global_kmeans_plan, global_kmeans_seed
from .input_validation import InputInventory
from .manifests import _digest, _read_json, _require, _write_json
from .neural import fp32_inference
from .training import runtime_record


def repeat_paths(experiment: Path, condition: str, repeat: int) -> tuple[Path, Path]:
    _require(condition in GLOBAL_CONDITIONS and type(repeat) is int and repeat in REPEATS,
             "Expected a planned condition and KMeans repeat 1, 2, or 3")
    suffix = f"global_k8_3seed_v1/clustering/{condition}/repeat_{repeat}"
    return experiment / "results" / suffix, experiment / "checkpoints" / suffix


def repeat_contract(
    experiment: Path, data: GlobalData, inventory: InputInventory, condition: str, repeat: int,
) -> dict[str, object]:
    global_kmeans_seed(repeat)
    contract = clustering_contract(experiment, data, inventory, condition)
    contract.update(schema_version=2, scope="global_kmeans_repeat", repeat=repeat,
                    seed=global_kmeans_seed(repeat), seed_plan=global_kmeans_plan(),
                    representation_source="frozen_global_fit")
    directory = Path(__file__).parent
    contract["code_sha256"].update({name: _digest(directory / name) for name in (
        "global_k8_repeats.py", "global_seed_plan.py")})
    return contract


def fit_repeat_clusters(
    features: np.ndarray, condition: str, sample_ids: tuple[str, ...], repeat: int,
    *, device: torch.device,
) -> GlobalClusters:
    _require(condition in GLOBAL_CONDITIONS and type(repeat) is int and repeat in REPEATS,
             "Invalid repeat fit identity")
    diagnostic = _diagnose(features, _dimension(condition))
    _require(len(features) >= K and bool(sample_ids), "Insufficient global fit pixels")
    config = experiment_config()
    _require(cluster_contract()["chemomae"] == config["chemomae"]["version"],
             "ChemoMAE version differs")
    settings = config["clustering"]
    seed = global_kmeans_seed(repeat)
    module = CosineKMeans(K, tol=settings["tol"], max_iter=settings["max_iter"],
                         device=device, random_state=seed).to(dtype=torch.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(features))
    started = time.perf_counter()
    with fp32_inference(device):
        module.fit(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        _diagnose(module.centroids.detach().cpu().numpy(), _dimension(condition))
        labels, distances = module.predict(tensor, return_dist=True)
        inertia = float(distances.gather(1, labels[:, None]).mean())
        occupancy = tuple(torch.bincount(labels, minlength=K).cpu().tolist())
    _require(math.isfinite(module.inertia_) and math.isfinite(inertia), "Nonfinite KMeans objective")
    record = GlobalClusterRecord(
        condition, _dimension(condition), K, seed, sample_ids, len(features), occupancy,
        float(module.inertia_), inertia, settings["max_iter"], settings["tol"], None,
        f"not_exposed_by_ChemoMAE_{cluster_contract()['chemomae']}", elapsed, str(device),
        diagnostic.unit_norm_absolute_error_max,
    )
    return GlobalClusters(module, record)


def save_repeat_clusters(fitted: GlobalClusters, path: Path, repeat: int) -> None:
    _require(fitted.record.seed == global_kmeans_seed(repeat), "Wrong KMeans repeat seed")
    with path.open("xb") as handle:
        np.savez_compressed(handle, centroids=fitted.centroids, metadata=np.array(json.dumps({
            "schema_version": 2, "scope": "global_kmeans_repeat", "repeat": repeat,
            "runtime": cluster_contract(), "fit": asdict(fitted.record),
        })))


def load_repeat_clusters(
    path: Path, condition: str, repeat: int, *, device: torch.device,
) -> GlobalClusters:
    _require(condition in GLOBAL_CONDITIONS and type(repeat) is int and repeat in REPEATS,
             "Invalid center identity")
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == {"centroids", "metadata"}, "Unexpected global centers")
        metadata = json.loads(str(saved["metadata"].item()))
        centers = saved["centroids"].copy()
    _require(metadata.get("schema_version") == 2
             and metadata.get("scope") == "global_kmeans_repeat"
             and metadata.get("repeat") == repeat
             and metadata.get("runtime") == cluster_contract(), "Repeat center contract differs")
    fit = metadata["fit"]
    record = GlobalClusterRecord(**{**fit, "sample_ids": tuple(fit["sample_ids"]),
                                   "fit_occupancy": tuple(fit["fit_occupancy"])})
    settings = experiment_config()["clustering"]
    _require(record.condition == condition and record.dimension == _dimension(condition)
             and record.k == K and record.seed == global_kmeans_seed(repeat)
             and record.max_iter == settings["max_iter"] and record.tol == settings["tol"]
             and centers.shape == (K, record.dimension) and record.fit_pixels >= K
             and len(record.fit_occupancy) == K and sum(record.fit_occupancy) == record.fit_pixels
             and math.isfinite(record.reference_inertia)
             and math.isfinite(record.final_center_inertia), "Invalid repeat centers")
    _diagnose(centers, record.dimension)
    module = CosineKMeans(K, tol=record.tol, max_iter=record.max_iter,
                         device=device, random_state=record.seed).to(dtype=torch.float32)
    module.centroids.resize_(centers.shape)
    module.centroids.copy_(torch.from_numpy(centers).to(device))
    module.latent_dim, module.inertia_, module._fitted = (
        record.dimension, record.reference_inertia, True)
    return GlobalClusters(module, record)


def run_repeat_clustering(
    experiment: Path, data: GlobalData, inventory: InputInventory, condition: str, repeat: int,
    *, device: torch.device, chunk_pixels: int = 1024,
) -> dict[str, object]:
    _require(type(chunk_pixels) is int and chunk_pixels > 0, "Invalid chunk size")
    results, checkpoints = repeat_paths(experiment, condition, repeat)
    _require(not results.exists() and not checkpoints.exists(),
             f"{condition} repeat {repeat}: output exists; use check or --resume")
    contract = repeat_contract(experiment, data, inventory, condition, repeat)
    started = time.perf_counter()
    representation, source = load_global_representation(experiment, data, condition, device=device)
    print(f"{condition} repeat {repeat}: extracting fixed global representation", flush=True)
    features = collect_global_features(data, representation, condition, chunk_pixels=chunk_pixels)
    samples = {sample.sample_id: sample for sample in inventory.samples}
    _require(set(samples) == set(data.train_sample_ids), "Global prediction samples differ")
    with TemporaryDirectory(prefix=".global-kmeans-repeat-", dir=experiment) as temporary:
        staging = Path(temporary)
        result_stage, weight_stage = staging / "results", staging / "checkpoints"
        (result_stage / "maps").mkdir(parents=True)
        weight_stage.mkdir()
        print(f"{condition} repeat {repeat}: fitting K={K}", flush=True)
        fitted = fit_repeat_clusters(features, condition, data.train_sample_ids, repeat, device=device)
        center_path = weight_stage / "centers_k8.npz"
        save_repeat_clusters(fitted, center_path, repeat)
        restored = load_repeat_clusters(center_path, condition, repeat, device=device)
        _require(np.array_equal(fitted.centroids, restored.centroids)
                 and np.array_equal(fitted.predict(features[:32]), restored.predict(features[:32])),
                 "Repeat center roundtrip differs")
        del features, fitted
        reports = []
        for sample_id in data.train_sample_ids:
            sample = samples[sample_id]
            with h5py.File(sample.path, "r") as handle:
                output = LabelMap(handle["valid_spectrum_mask"][:], K)
            count, norm_error = 0, 0.0
            for batch in AllPixelData((sample,)).all_batches(chunk_pixels=chunk_pixels):
                transformed = _transform_batch(representation, batch, condition)
                norm_error = max(norm_error, transformed.diagnostics.unit_norm_absolute_error_max)
                output.add(batch.pixel_row_col, restored.predict(transformed.values))
                count += len(batch.snv)
            labels = output.finish()
            _require(count == sample.saved_pixel_count, f"{sample_id}: missing prediction rows")
            path = result_stage / "maps" / f"{sample_id}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, labels_k8=labels)
            reports.append({"sample_id": sample_id, "pixels": count,
                            "map_sha256": _digest(path), "shape": list(labels.shape),
                            "occupancy": np.bincount(labels.ravel(), minlength=K + 1)[1:].tolist(),
                            "unit_norm_error_max": norm_error})
            print(f"{condition} repeat {repeat} {sample_id}: saved {count} labels", flush=True)
        _require(source_record(experiment, data, condition) == source
                 and repeat_contract(experiment, data, inventory, condition, repeat) == contract,
                 "Repeat clustering source changed during run")
        _write_json(result_stage / "run.json", {
            "contract": contract, "source": source, "execution": runtime_record(device),
            "chunk_pixels": chunk_pixels, "fit": asdict(restored.record),
        })
        report = {"status": "global_kmeans_repeat_completed", "scope": "global_fit",
                  "checks_passed": True, "repeat": repeat, "K": K,
                  "run_sha256": _digest(result_stage / "run.json"), "samples": reports,
                  "centers_sha256": _digest(center_path),
                  "centers_and_probe_labels_save_load_exact": True,
                  "wall_seconds": time.perf_counter() - started}
        _write_json(result_stage / "completion.json", report)
        results.parent.mkdir(parents=True, exist_ok=True)
        checkpoints.parent.mkdir(parents=True, exist_ok=True)
        weight_stage.rename(checkpoints)
        result_stage.rename(results)
    return report


def check_repeat_clustering(
    experiment: Path, data: GlobalData, inventory: InputInventory, condition: str, repeat: int,
) -> dict[str, object]:
    """Audit the fixed source, repeat seed, centers and all saved label maps."""
    results, checkpoints = repeat_paths(experiment, condition, repeat)
    run, completion = _read_json(results / "run.json"), _read_json(results / "completion.json")
    _require(run.get("contract") == repeat_contract(experiment, data, inventory, condition, repeat)
             and run.get("source") == source_record(experiment, data, condition),
             "Repeat clustering contract/source changed")
    _require(completion.get("status") == "global_kmeans_repeat_completed"
             and completion.get("scope") == "global_fit" and completion.get("repeat") == repeat
             and completion.get("K") == K and completion.get("checks_passed") is True
             and completion.get("centers_and_probe_labels_save_load_exact") is True
             and completion.get("run_sha256") == _digest(results / "run.json")
             and completion.get("centers_sha256") == _digest(checkpoints / "centers_k8.npz"),
             "Repeat clustering completion/artifact differs")
    cluster = load_repeat_clusters(checkpoints / "centers_k8.npz", condition, repeat,
                                   device=torch.device("cpu"))
    _require(cluster.record.sample_ids == data.train_sample_ids
             and cluster.record.fit_pixels == data.train_pixel_count
             and json.loads(json.dumps(asdict(cluster.record))) == run["fit"],
             "Repeat center fit provenance differs")
    reports = completion["samples"]
    _require([r["sample_id"] for r in reports] == list(data.train_sample_ids),
             "Repeat sample coverage differs")
    samples = {sample.sample_id: sample for sample in inventory.samples}
    for report in reports:
        sample = samples[report["sample_id"]]
        path = results / "maps" / f"{sample.sample_id}.npz"
        _require(report["map_sha256"] == _digest(path), "Repeat map hash changed")
        labels = read_label_map(path, sample)
        _require(report["pixels"] == sample.saved_pixel_count
                 and report["shape"] == list(labels.shape)
                 and report["occupancy"] == np.bincount(
                     labels.ravel(), minlength=K + 1)[1:].tolist(),
                 "Repeat map summary differs")
    return {"status": "validated_global_kmeans_repeat", "checks_passed": True,
            "condition": condition, "repeat": repeat, "K": K,
            "samples": len(reports), "fit_pixels": data.train_pixel_count,
            "prediction_pixels": sum(r["pixels"] for r in reports)}
