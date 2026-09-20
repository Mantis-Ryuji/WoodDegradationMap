"""Fixed global representations, one K=8 fit, and complete all-sample predictions."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans

from .baselines import B0Baseline
from .cluster_pipeline import NeuralRepresentation, _transform_batch
from .clustering import FittedClusters, LabelMap, Representation, _diagnose, _dimension
from .clustering import cluster_contract
from .config import experiment_config
from .data import FoldData, SpectrumBatch, _Selection
from .global_baselines import GlobalPCA, global_baselines
from .global_manifest import GLOBAL_CONDITIONS, GlobalData, global_config, global_seed
from .global_training import build_global_model, global_code_hashes
from .input_validation import InputInventory, SampleInput
from .manifests import _digest, _read_json, _require, _write_json
from .neural import fp32_inference
from .records import matches_run
from .training import runtime_record

K = 8


class AllPixelData(FoldData):
    """Use the existing validated chunk reader without assigning a CV fold/split."""

    def __init__(self, samples: tuple[SampleInput, ...]) -> None:
        self._train = tuple(_Selection(s, None, None) for s in samples)
        self._test = ()

    def all_batches(self, *, chunk_pixels: int) -> Iterator[SpectrumBatch]:
        yield from self.batches("train", chunk_pixels=chunk_pixels)


def condition_paths(experiment: Path, condition: str) -> tuple[Path, Path]:
    _require(condition in GLOBAL_CONDITIONS, "Not a planned global condition")
    suffix = f"clustering/{condition}/repeat_1"
    return experiment / "results" / suffix, experiment / "checkpoints" / suffix


def source_record(experiment: Path, data: GlobalData, condition: str) -> dict[str, object]:
    """Validate completed source identity and hashes without loading optimizer state."""
    _require(condition in GLOBAL_CONDITIONS, "Not a planned global condition")
    if condition in ("B0", "B1"):
        report = global_baselines(experiment, data, fit=False)
        return {"kind": condition, "completion_sha256": _digest(
            experiment / "results/baselines/completion.json"),
            "artifact_sha256": report["artifact_sha256"]}
    result = experiment / f"results/neural/{condition}/repeat_1"
    weights = experiment / f"checkpoints/neural/{condition}/repeat_1/last_model.pt"
    run, completion = _read_json(result / "run.json"), _read_json(result / "completion.json")
    config = global_config()
    steps = data.train_pixel_count // config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    expected = {
        "schema_version": 2, "scope": "global_fit", "mode": "training",
        "condition": condition, "repeat": 1, "smoke_id": None,
        "smoke_batches_per_epoch": None, "config": config,
        "train_sample_ids": list(data.train_sample_ids), "train_pixels": data.train_pixel_count,
        "batch_size": config["training"]["batch_size"], "steps_per_full_epoch": steps,
        "planned_production_updates": steps * epochs, "code_sha256": global_code_hashes(),
        "manifest_artifact_sha256": _read_json(
            experiment / "manifests/complete.json")["artifact_sha256"],
        "seeds": {purpose: global_seed(purpose)
                  for purpose in ("model_init", "pixel_order", "mask", "train_aug")},
    }
    _require(matches_run(run, expected), f"{condition}: global neural source contract differs")
    runtime = runtime_record(torch.device("cpu"))
    _require(all(run["contract"]["runtime"][key] == runtime[key]
                 for key in ("torch", "chemomae")), "Source library versions differ")
    _require(completion.get("status") == "training_completed"
             and completion.get("completed_epochs") == epochs
             and completion.get("attempted_updates") == steps * epochs,
             f"{condition}: source has not completed epoch {epochs}")
    _require(0 < completion["nonzero_lr_updates"] <= completion["optimizer_updates"]
             <= completion["attempted_updates"]
             and completion["amp_skips"] == completion["attempted_updates"]
             - completion["optimizer_updates"], "Invalid source update counters")
    _require(Path(completion["weights_file"]).resolve() == weights.resolve()
             and completion["weights_sha256"] == _digest(weights), "Source weights changed")
    boundary = _read_json(result / "checkpoint.json")
    _require(all(boundary.get(key) == completion[key] for key in (
        "completed_epochs", "attempted_updates", "optimizer_updates")),
        "Source checkpoint/completion counters differ")
    return {"kind": condition, "run_sha256": _digest(result / "run.json"),
            "completion_sha256": _digest(result / "completion.json"),
            "weights_sha256": completion["weights_sha256"], "completed_epochs": epochs}


def load_global_representation(
    experiment: Path, data: GlobalData, condition: str, *, device: torch.device,
) -> tuple[Representation, dict[str, object]]:
    source = source_record(experiment, data, condition)
    if condition == "B0":
        return B0Baseline.load(experiment / "results/baselines/b0.json"), source
    if condition == "B1":
        return GlobalPCA.load(experiment / "checkpoints/baselines/pca.npz"), source
    weights = experiment / f"checkpoints/neural/{condition}/repeat_1/last_model.pt"
    state = torch.load(weights, map_location="cpu", weights_only=True)
    _require(isinstance(state, dict) and bool(state) and all(
        isinstance(v, torch.Tensor) and v.dtype == torch.float32 and torch.isfinite(v).all()
        for v in state.values()), "Expected finite raw FP32 global weights")
    model = build_global_model(condition)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return NeuralRepresentation(model, device), source


def collect_global_features(
    data: GlobalData, representation: Representation, condition: str, *, chunk_pixels: int,
) -> np.ndarray:
    features = np.empty((data.train_pixel_count, _dimension(condition)), dtype=np.float32)
    offset = 0
    for batch in data.batches("train", chunk_pixels=chunk_pixels):
        values = _transform_batch(representation, batch, condition).values
        stop = offset + len(values)
        _require(stop <= len(features), "Too many global fit rows")
        features[offset:stop] = values
        offset = stop
    _require(offset == len(features), "Missing global fit rows")
    return features


@dataclass(frozen=True)
class GlobalClusterRecord:
    condition: str
    dimension: int
    k: int
    seed: int
    sample_ids: tuple[str, ...]
    fit_pixels: int
    fit_occupancy: tuple[int, ...]
    reference_inertia: float
    final_center_inertia: float
    max_iter: int
    tol: float
    iterations: int | None
    stop_reason: str
    fit_seconds: float
    fit_device: str
    unit_norm_error_max: float


class GlobalClusters(FittedClusters):
    """Reuse fixed-center prediction with a distinct, fold-free saved contract."""

    def __init__(self, module: CosineKMeans, record: GlobalClusterRecord) -> None:
        self._module = module
        self.record = record

    def save(self, path: Path) -> None:
        with path.open("xb") as handle:
            np.savez_compressed(handle, centroids=self.centroids, metadata=np.array(json.dumps({
                "schema_version": 1, "scope": "global_clustering", "runtime": cluster_contract(),
                "fit": asdict(self.record),
            })))

    @classmethod
    def load(cls, path: Path, *, condition: str, device: torch.device) -> GlobalClusters:
        with np.load(path, allow_pickle=False) as saved:
            _require(set(saved.files) == {"centroids", "metadata"}, "Unexpected global centers")
            metadata = json.loads(str(saved["metadata"].item()))
            centers = saved["centroids"].copy()
        _require(metadata.get("schema_version") == 1
                 and metadata.get("scope") == "global_clustering"
                 and metadata.get("runtime") == cluster_contract(), "Global center contract differs")
        fit = metadata["fit"]
        record = GlobalClusterRecord(**{**fit, "sample_ids": tuple(fit["sample_ids"]),
                                       "fit_occupancy": tuple(fit["fit_occupancy"])})
        settings = experiment_config()["clustering"]
        _require(condition in GLOBAL_CONDITIONS and record.condition == condition
                 and record.dimension == _dimension(condition) and record.k == K
                 and record.seed == global_seed("kmeans")
                 and record.max_iter == settings["max_iter"] and record.tol == settings["tol"],
                 "Global center identity/settings differ")
        _require(centers.shape == (K, record.dimension) and record.fit_pixels >= K
                 and bool(record.sample_ids) and len(record.fit_occupancy) == K
                 and all(type(n) is int and n >= 0 for n in record.fit_occupancy)
                 and sum(record.fit_occupancy) == record.fit_pixels
                 and math.isfinite(record.reference_inertia)
                 and math.isfinite(record.final_center_inertia), "Invalid global center provenance")
        _diagnose(centers, record.dimension)
        module = CosineKMeans(K, tol=record.tol, max_iter=record.max_iter,
                             device=device, random_state=record.seed).to(dtype=torch.float32)
        module.centroids.resize_(centers.shape)
        module.centroids.copy_(torch.from_numpy(centers).to(device))
        module.latent_dim, module.inertia_, module._fitted = (
            record.dimension, record.reference_inertia, True)
        return cls(module, record)


def fit_global_clusters(
    features: np.ndarray, condition: str, sample_ids: tuple[str, ...], *, device: torch.device,
) -> GlobalClusters:
    _require(condition in GLOBAL_CONDITIONS, "Not a planned global condition")
    config = experiment_config()
    _require(cluster_contract()["chemomae"] == config["chemomae"]["version"],
             "ChemoMAE version differs")
    diagnostic = _diagnose(features, _dimension(condition))
    _require(len(features) >= K and bool(sample_ids), "Insufficient global fit pixels")
    settings = config["clustering"]
    module = CosineKMeans(K, tol=settings["tol"], max_iter=settings["max_iter"],
                         device=device, random_state=global_seed("kmeans")).to(dtype=torch.float32)
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
        condition, _dimension(condition), K, global_seed("kmeans"), sample_ids, len(features),
        occupancy, float(module.inertia_), inertia, settings["max_iter"], settings["tol"], None,
        f"not_exposed_by_ChemoMAE_{cluster_contract()['chemomae']}", elapsed, str(device),
        diagnostic.unit_norm_absolute_error_max,
    )
    return GlobalClusters(module, record)


def clustering_contract(
    experiment: Path, data: GlobalData, inventory: InputInventory, condition: str,
) -> dict[str, object]:
    code = ("global_clustering.py", "cluster_pipeline.py", "clustering.py", "global_baselines.py",
            "global_manifest.py", "global_training.py", "baselines.py", "data.py", "neural.py",
            "config.py", "manifests.py", "input_validation.py", "records.py", "training.py")
    return {
        "schema_version": 1, "scope": "global_clustering", "condition": condition,
        "repeat": 1, "K": K, "seed": global_seed("kmeans"),
        "settings": experiment_config()["clustering"], "runtime": cluster_contract(),
        "sample_ids": list(data.train_sample_ids), "fit_pixels": data.train_pixel_count,
        "prediction_pixels": sum(s.saved_pixel_count for s in inventory.samples),
        "manifest_sha256": _digest(experiment / "manifests/complete.json"),
        "code_sha256": {name: _digest(Path(__file__).with_name(name)) for name in code},
    }


def read_label_map(path: Path, sample: SampleInput) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == {"labels_k8"}, "Unexpected global label fields")
        labels = saved["labels_k8"].copy()
    with h5py.File(sample.path, "r") as handle:
        mask = handle["valid_spectrum_mask"][:]
    _require(mask.shape == (sample.height, sample.width) and np.isin(mask, (0, 1)).all(),
             f"{sample.sample_id}: invalid mask")
    valid = mask == 1
    _require(labels.dtype == np.uint8 and labels.shape == valid.shape
             and np.all(labels[~valid] == 0) and np.all((labels[valid] >= 1) & (labels[valid] <= K))
             and int(valid.sum()) == sample.saved_pixel_count,
             f"{sample.sample_id}: invalid global label coverage")
    return labels


def run_global_clustering(
    experiment: Path, data: GlobalData, inventory: InputInventory, condition: str,
    *, device: torch.device, chunk_pixels: int = 1024,
) -> dict[str, object]:
    _require(type(chunk_pixels) is int and chunk_pixels > 0, "Invalid chunk size")
    results, checkpoints = condition_paths(experiment, condition)
    _require(not results.exists() and not checkpoints.exists(),
             f"{condition}: global clustering output exists; use check or --resume")
    contract = clustering_contract(experiment, data, inventory, condition)
    started = time.perf_counter()
    representation, source = load_global_representation(experiment, data, condition, device=device)
    print(f"{condition}: extracting {data.train_pixel_count} global fit representations", flush=True)
    features = collect_global_features(data, representation, condition, chunk_pixels=chunk_pixels)
    samples = {sample.sample_id: sample for sample in inventory.samples}
    _require(set(samples) == set(data.train_sample_ids), "Global prediction samples differ")
    # Publish only a complete condition. Failed runs leave their final paths available to retry.
    with TemporaryDirectory(prefix=".global-clustering-", dir=experiment) as temporary:
        staging = Path(temporary)
        result_stage, weight_stage = staging / "results", staging / "checkpoints"
        (result_stage / "maps").mkdir(parents=True)
        weight_stage.mkdir()
        print(f"{condition}: fitting K={K} once", flush=True)
        fitted = fit_global_clusters(features, condition, data.train_sample_ids, device=device)
        center_path = weight_stage / "centers_k8.npz"
        fitted.save(center_path)
        restored = GlobalClusters.load(center_path, condition=condition, device=device)
        _require(np.array_equal(fitted.centroids, restored.centroids)
                 and np.array_equal(fitted.predict(features[:32]), restored.predict(features[:32])),
                 "Global center roundtrip differs")
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
            print(f"{condition} {sample_id}: saved {count} global labels", flush=True)
        _require(source_record(experiment, data, condition) == source, "Source changed during run")
        _require(clustering_contract(experiment, data, inventory, condition) == contract,
                 "Global clustering inputs/code changed during run")
        _write_json(result_stage / "run.json", {
            "contract": contract, "source": source, "execution": runtime_record(device),
            "chunk_pixels": chunk_pixels, "fit": asdict(restored.record),
        })
        report = {"status": "global_maps_completed", "scope": "global_fit", "checks_passed": True,
                  "run_sha256": _digest(result_stage / "run.json"), "samples": reports,
                  "centers_sha256": _digest(center_path), "K": K,
                  "centers_and_probe_labels_save_load_exact": True,
                  "wall_seconds": time.perf_counter() - started}
        _write_json(result_stage / "completion.json", report)
        results.parent.mkdir(parents=True, exist_ok=True)
        checkpoints.parent.mkdir(parents=True, exist_ok=True)
        weight_stage.rename(checkpoints)
        result_stage.rename(results)
    return report


def check_global_clustering(
    experiment: Path, data: GlobalData, inventory: InputInventory, condition: str,
) -> dict[str, object]:
    """CPU audit of source, saved centers, every map hash and valid-mask coverage."""
    results, checkpoints = condition_paths(experiment, condition)
    run, completion = _read_json(results / "run.json"), _read_json(results / "completion.json")
    _require(run.get("contract") == clustering_contract(experiment, data, inventory, condition)
             and run.get("source") == source_record(experiment, data, condition),
             "Global clustering contract/source changed")
    _require(completion.get("status") == "global_maps_completed"
             and completion.get("scope") == "global_fit" and completion.get("K") == K
             and completion.get("checks_passed") is True
             and completion.get("centers_and_probe_labels_save_load_exact") is True
             and completion.get("run_sha256") == _digest(results / "run.json")
             and completion.get("centers_sha256") == _digest(checkpoints / "centers_k8.npz"),
             "Global clustering completion/artifact differs")
    cluster = GlobalClusters.load(checkpoints / "centers_k8.npz", condition=condition,
                                  device=torch.device("cpu"))
    _require(cluster.record.sample_ids == data.train_sample_ids
             and cluster.record.fit_pixels == data.train_pixel_count
             and json.loads(json.dumps(asdict(cluster.record))) == run["fit"],
             "Global center fit provenance differs")
    reports = completion["samples"]
    _require([r["sample_id"] for r in reports] == list(data.train_sample_ids),
             "Global sample coverage differs")
    samples = {sample.sample_id: sample for sample in inventory.samples}
    for report in reports:
        sample = samples[report["sample_id"]]
        path = results / "maps" / f"{sample.sample_id}.npz"
        _require(report["map_sha256"] == _digest(path), "Global map hash changed")
        labels = read_label_map(path, sample)
        _require(report["pixels"] == sample.saved_pixel_count
                 and report["shape"] == list(labels.shape)
                 and report["occupancy"] == np.bincount(
                     labels.ravel(), minlength=K + 1)[1:].tolist(), "Global map summary differs")
    return {"status": "validated_global_clustering", "checks_passed": True,
            "condition": condition, "K": K, "samples": len(reports),
            "fit_pixels": data.train_pixel_count,
            "prediction_pixels": sum(r["pixels"] for r in reports)}
