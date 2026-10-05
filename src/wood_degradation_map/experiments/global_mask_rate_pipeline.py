"""Frozen M11-75 K=8 fits and diagnostics, isolated from the main global results."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans

from .cluster_pipeline import NeuralRepresentation, _transform_batch
from .clustering import LabelMap, _diagnose, cluster_contract
from .config import REPEATS, experiment_config
from .global_clustering import K, AllPixelData, collect_global_features, read_label_map
from .global_manifest import GlobalData
from .global_mask_rate_training import (
    CONDITION, SNAPSHOT, build_model, training_paths, training_source,
)
from .global_seed_plan import global_kmeans_plan, global_kmeans_seed
from .global_silhouette import PooledCosineSilhouette
from .input_validation import InputInventory
from .manifests import _digest, _read_json, _require, _write_json
from .neural import fp32_inference
from .spatial_metrics import local_label_agreement
from .training import runtime_record

METRICS = ("adjusted_lla_3", "adjusted_lla_5", "adjusted_lla_9", "silhouette",
           "used_clusters", "maximum_occupancy")


def cluster_paths(experiment: Path, repeat: int) -> tuple[Path, Path]:
    global_kmeans_seed(repeat)
    suffix = f"{SNAPSHOT}/clustering/{CONDITION}/repeat_{repeat}"
    return experiment / "results" / suffix, experiment / "checkpoints" / suffix


def evaluation_path(experiment: Path) -> Path:
    return experiment / f"results/{SNAPSHOT}/conditions/{CONDITION}"


def seal(output: Path, status: str) -> dict[str, object]:
    """Commit a complete staged result, including every map/table in its audit."""
    hashes = {path.relative_to(output).as_posix(): _digest(path)
              for path in sorted(output.rglob("*")) if path.is_file()}
    _require("completion.json" not in hashes, "Output was already sealed")
    report = {"status": status, "checks_passed": True, "artifact_sha256": hashes}
    _write_json(output / "completion.json", report)
    return report


def check_sealed(output: Path, status: str, contract: dict[str, object]) -> dict[str, object]:
    completion = _read_json(output / "completion.json")
    _require(completion.get("status") == status and completion.get("checks_passed") is True
             and _read_json(output / "run.json").get("contract") == contract,
             f"Supplement contract/completion differs: {output}")
    hashes = completion.get("artifact_sha256")
    names = {path.relative_to(output).as_posix() for path in output.rglob("*")
             if path.is_file() and path.name != "completion.json"}
    _require(isinstance(hashes, dict) and set(hashes) == names
             and all(_digest(output / name) == value for name, value in hashes.items()),
             f"Supplement artifact changed: {output}")
    return completion


def pipeline_contract(
    experiment: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    _require(sorted(sample.sample_id for sample in inventory.samples) == list(data.train_sample_ids),
             "Supplement inventory differs from the global fit manifest")
    files = (Path(__file__).name, "global_clustering.py", "global_seed_plan.py",
             "global_silhouette.py", "spatial_metrics.py", "cluster_pipeline.py", "clustering.py")
    return {
        "schema_version": 1, "scope": SNAPSHOT, "condition": CONDITION,
        "K": K, "seed_plan": global_kmeans_plan(), "source": training_source(experiment, data),
        "manifest_sha256": _digest(experiment / "manifests/complete.json"),
        "sample_ids": list(data.train_sample_ids), "fit_pixels": data.train_pixel_count,
        "prediction_pixels": sum(sample.saved_pixel_count for sample in inventory.samples),
        "settings": experiment_config()["clustering"], "runtime": cluster_contract(),
        "code_sha256": {name: _digest(Path(__file__).with_name(name)) for name in files},
        "representation": "one saved epoch-800 global fit, fixed across all KMeans repeats",
        "held_out": False,
    }


def load_representation(experiment: Path, data: GlobalData, device: torch.device) -> NeuralRepresentation:
    training_source(experiment, data)
    state = torch.load(training_paths(experiment)[1] / "last_model.pt",
                       map_location="cpu", weights_only=True)
    _require(isinstance(state, dict) and bool(state) and all(
        isinstance(value, torch.Tensor) and value.dtype == torch.float32
        and torch.isfinite(value).all() for value in state.values()),
        "Expected finite FP32 M11-75 raw weights")
    model = build_model()
    model.load_state_dict(state, strict=True)
    return NeuralRepresentation(model.to(device).eval(), device)


def _new_kmeans(repeat: int, device: torch.device) -> CosineKMeans:
    settings = experiment_config()["clustering"]
    _require(cluster_contract()["chemomae"] == experiment_config()["chemomae"]["version"],
             "ChemoMAE version differs")
    return CosineKMeans(K, tol=settings["tol"], max_iter=settings["max_iter"],
                        device=device, random_state=global_kmeans_seed(repeat)).to(torch.float32)


def load_centers(path: Path, repeat: int, device: torch.device) -> CosineKMeans:
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == {"centroids", "seed", "condition"}
                 and saved["seed"].item() == global_kmeans_seed(repeat)
                 and saved["condition"].item() == CONDITION, "M11-75 center identity differs")
        centers = saved["centroids"].copy()
    _require(centers.shape == (K, 16), "Expected eight 16-dimensional centers")
    _diagnose(centers, 16)
    model = _new_kmeans(repeat, device)
    # Avoid the library loader's second normalization, as in the main global fits.
    model.centroids.resize_(centers.shape)
    model.centroids.copy_(torch.from_numpy(centers).to(device))
    model.latent_dim, model._fitted = 16, True
    return model


def _predict(model: CosineKMeans, features: np.ndarray) -> np.ndarray:
    _diagnose(features, 16)
    with fp32_inference(model.device):
        return model.predict(torch.from_numpy(np.ascontiguousarray(features))).cpu().numpy()


def check_cluster(
    experiment: Path, data: GlobalData, inventory: InputInventory, repeat: int,
) -> dict[str, object]:
    results, weights = cluster_paths(experiment, repeat)
    contract = {**pipeline_contract(experiment, data, inventory), "repeat": repeat}
    completion = check_sealed(results, "mask_rate_clustering_completed", contract)
    run = _read_json(results / "run.json")
    _require(run["centers_sha256"] == _digest(weights / "centers_k8.npz"), "Centers changed")
    load_centers(weights / "centers_k8.npz", repeat, torch.device("cpu"))
    _require(run["fit"]["seed"] == global_kmeans_seed(repeat)
             and sum(run["fit"]["occupancy"]) == data.train_pixel_count,
             "Fit occupancy/seed differs")
    for sample in inventory.samples:
        read_label_map(results / "maps" / f"{sample.sample_id}.npz", sample)
    return completion


def cluster(
    experiment: Path, data: GlobalData, inventory: InputInventory,
    *, device: torch.device, chunk_pixels: int = 1024, resume: bool = False,
) -> None:
    """Extract the selected fit pixels once and reuse them in all three fits."""
    _require(type(chunk_pixels) is int and chunk_pixels > 0, "Invalid chunk size")
    pending = []
    for repeat in REPEATS:
        results, weights = cluster_paths(experiment, repeat)
        if results.exists() or weights.exists():
            _require(resume, "Cluster output exists; use --resume to verify and skip it")
            check_cluster(experiment, data, inventory, repeat)
            print(f"{CONDITION} repeat {repeat}: verified complete; skipped", flush=True)
        else:
            pending.append(repeat)
    if not pending:
        return
    contract = pipeline_contract(experiment, data, inventory)
    representation = load_representation(experiment, data, device)
    features = collect_global_features(data, representation, CONDITION, chunk_pixels=chunk_pixels)
    _diagnose(features, 16)
    samples = {sample.sample_id: sample for sample in inventory.samples}
    for repeat in pending:
        results, weights = cluster_paths(experiment, repeat)
        with TemporaryDirectory(prefix=".mask-rate-cluster-", dir=experiment) as temporary:
            stage, weight_stage = Path(temporary) / "results", Path(temporary) / "weights"
            (stage / "maps").mkdir(parents=True)
            weight_stage.mkdir()
            model = _new_kmeans(repeat, device)
            started = time.perf_counter()
            with fp32_inference(device):
                tensor = torch.from_numpy(np.ascontiguousarray(features))
                model.fit(tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                fit_seconds = time.perf_counter() - started
                labels, distances = model.predict(tensor, return_dist=True)
                inertia = float(distances.gather(1, labels[:, None]).mean())
                occupancy = torch.bincount(labels, minlength=K).cpu().tolist()
            _require(np.isfinite(model.inertia_) and np.isfinite(inertia), "Nonfinite objective")
            centers = model.centroids.detach().cpu().numpy().copy()
            _diagnose(centers, 16)
            center_path = weight_stage / "centers_k8.npz"
            np.savez_compressed(center_path, centroids=centers, seed=global_kmeans_seed(repeat),
                                condition=CONDITION)
            restored = load_centers(center_path, repeat, device)
            _require(np.array_equal(centers, restored.centroids.detach().cpu().numpy())
                     and np.array_equal(_predict(model, features[:32]),
                                        _predict(restored, features[:32])), "Center roundtrip differs")
            fit = {"seed": global_kmeans_seed(repeat), "occupancy": occupancy,
                   "reference_inertia": float(model.inertia_), "final_center_inertia": inertia,
                   "fit_seconds": fit_seconds}
            del model, tensor, labels, distances
            for sample_id in data.train_sample_ids:
                sample = samples[sample_id]
                with h5py.File(sample.path, "r") as handle:
                    label_map = LabelMap(handle["valid_spectrum_mask"][:], K)
                for batch in AllPixelData((sample,)).all_batches(chunk_pixels=chunk_pixels):
                    values = _transform_batch(representation, batch, CONDITION).values
                    label_map.add(batch.pixel_row_col, _predict(restored, values))
                np.savez_compressed(stage / "maps" / f"{sample_id}.npz",
                                    labels_k8=label_map.finish())
                print(f"{CONDITION} repeat {repeat}: {sample_id}", flush=True)
            _require(pipeline_contract(experiment, data, inventory) == contract,
                     "Clustering source changed during computation")
            _write_json(stage / "run.json", {
                "contract": {**contract, "repeat": repeat}, "execution": runtime_record(device),
                "fit": fit, "chunk_pixels": chunk_pixels,
                "centers_sha256": _digest(center_path), "center_roundtrip_exact": True,
            })
            seal(stage, "mask_rate_clustering_completed")
            results.parent.mkdir(parents=True, exist_ok=True)
            weights.parent.mkdir(parents=True, exist_ok=True)
            weight_stage.rename(weights)
            stage.rename(results)
        check_cluster(experiment, data, inventory, repeat)


def evaluation_contract(
    experiment: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    for repeat in REPEATS:
        check_cluster(experiment, data, inventory, repeat)
    return {**pipeline_contract(experiment, data, inventory),
            "maps": {str(repeat): _digest(cluster_paths(experiment, repeat)[0] / "completion.json")
                     for repeat in REPEATS}, "metrics": list(METRICS)}


def evaluate(
    experiment: Path, data: GlobalData, inventory: InputInventory,
    *, device: torch.device, chunk_pixels: int = 1024,
) -> dict[str, object]:
    """Use the main full-fit LLA and pooled cosine silhouette definitions."""
    _require(type(chunk_pixels) is int and chunk_pixels > 0, "Invalid chunk size")
    output = evaluation_path(experiment)
    _require(not output.exists(), "Evaluation exists; use --resume to verify and skip")
    contract = evaluation_contract(experiment, data, inventory)
    representation = load_representation(experiment, data, device)
    samples = {sample.sample_id: sample for sample in inventory.samples}
    pooled = {repeat: PooledCosineSilhouette(K, 16) for repeat in REPEATS}
    rows: dict[tuple[str, int], dict[str, object]] = {}
    snv_means = np.full((len(samples), K, 256), np.nan, dtype=np.float64)
    for si, sample_id in enumerate(data.train_sample_ids):
        sample = samples[sample_id]
        maps = {repeat: read_label_map(cluster_paths(experiment, repeat)[0] / "maps"
                                       / f"{sample_id}.npz", sample) for repeat in REPEATS}
        for repeat, labels in maps.items():
            spatial = local_label_agreement(labels, labels > 0, k=K)
            row = {"sample_id": sample_id, "repeat": repeat,
                   "valid_pixels": sample.saved_pixel_count, "occupancy": list(spatial.occupancy),
                   "used_clusters": spatial.used_clusters,
                   "maximum_occupancy": spatial.maximum_occupancy, "undefined_reasons": {}}
            for window in spatial.windows:
                metric = f"adjusted_lla_{window.window}"
                row[metric] = window.adjusted_lla
                if window.adjusted_lla is None:
                    row["undefined_reasons"][metric] = ";".join(window.adjusted_undefined_reasons)
            rows[sample_id, repeat] = row
        sums, counts = np.zeros((K, 256), dtype=np.float64), np.zeros(K, dtype=np.int64)
        count = 0
        for batch in AllPixelData((sample,)).all_batches(chunk_pixels=chunk_pixels):
            values = _transform_batch(representation, batch, CONDITION).values
            coordinates = tuple(batch.pixel_row_col.T)
            for repeat in REPEATS:
                pooled[repeat].add(values, maps[repeat][coordinates])
            labels = maps[1][coordinates].astype(np.int64) - 1
            counts += np.bincount(labels, minlength=K)
            for label in np.unique(labels):
                sums[label] += batch.snv[labels == label].sum(axis=0, dtype=np.float64)
            count += len(values)
        _require(count == sample.saved_pixel_count, "Incomplete first-pass feature coverage")
        present = counts > 0
        snv_means[si, present] = sums[present] / counts[present, None]
        print(f"{CONDITION} {sample_id}: LLA, occupancy and silhouette pass 1", flush=True)
    reasons = {repeat: pooled[repeat].finalize() for repeat in REPEATS}
    for sample_id in data.train_sample_ids:
        sample = samples[sample_id]
        maps = {repeat: read_label_map(cluster_paths(experiment, repeat)[0] / "maps"
                                       / f"{sample_id}.npz", sample) for repeat in REPEATS}
        totals = {repeat: 0.0 for repeat in REPEATS}
        count = 0
        for batch in AllPixelData((sample,)).all_batches(chunk_pixels=chunk_pixels):
            values = _transform_batch(representation, batch, CONDITION).values
            coordinates = tuple(batch.pixel_row_col.T)
            for repeat in REPEATS:
                if reasons[repeat] is None:
                    totals[repeat] += float(pooled[repeat].score(
                        values, maps[repeat][coordinates]).sum(dtype=np.float64))
            count += len(values)
        _require(count == sample.saved_pixel_count, "Incomplete second-pass feature coverage")
        for repeat in REPEATS:
            row = rows[sample_id, repeat]
            row["silhouette"] = totals[repeat] / count if reasons[repeat] is None else None
            if reasons[repeat] is not None:
                row["undefined_reasons"]["silhouette"] = reasons[repeat]
        print(f"{CONDITION} {sample_id}: silhouette pass 2", flush=True)
    _require(evaluation_contract(experiment, data, inventory) == contract,
             "Evaluation source changed during computation")
    with TemporaryDirectory(prefix=".mask-rate-evaluation-", dir=experiment) as temporary:
        stage = Path(temporary) / "evaluation"
        stage.mkdir()
        _write_json(stage / "run.json", {
            "contract": contract, "chunk_pixels": chunk_pixels, "execution": runtime_record(device),
            "silhouette_method": "pooled exact cosine via cluster vector sums; FP64 accumulation",
            "pooled_cluster_counts": {str(r): pooled[r].counts.tolist() for r in REPEATS},
        })
        _write_json(stage / "samples.json", {"samples": [rows[sample_id, repeat]
            for sample_id in data.train_sample_ids for repeat in REPEATS]})
        np.savez_compressed(stage / "snv_means.npz", sample_ids=np.array(data.train_sample_ids),
                            means=snv_means)
        completion = seal(stage, "mask_rate_evaluation_completed")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output)
    return completion


def check_evaluation(
    experiment: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    output = evaluation_path(experiment)
    completion = check_sealed(output, "mask_rate_evaluation_completed",
                              evaluation_contract(experiment, data, inventory))
    rows = _read_json(output / "samples.json")["samples"]
    _require([(row["sample_id"], row["repeat"]) for row in rows]
             == [(sample_id, repeat) for sample_id in data.train_sample_ids for repeat in REPEATS],
             "Evaluation sample/repeat coverage differs")
    counts = _read_json(output / "run.json")["pooled_cluster_counts"]
    _require(all(len(counts[str(r)]) == K and sum(counts[str(r)])
                 == sum(s.saved_pixel_count for s in inventory.samples) for r in REPEATS),
             "Pooled occupancy coverage differs")
    with np.load(output / "snv_means.npz", allow_pickle=False) as saved:
        _require(saved["sample_ids"].tolist() == list(data.train_sample_ids)
                 and saved["means"].shape == (len(data.train_sample_ids), K, 256),
                 "SNV summary coverage differs")
    return completion
