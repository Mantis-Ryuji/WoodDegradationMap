"""Inputs, clustering, evaluation and reporting for the M11 spatial experiment."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import h5py
import numpy as np
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans

from .cluster_pipeline import _transform_batch
from .clustering import LabelMap, _diagnose, _dimension, cluster_contract
from .config import _seed, experiment_config
from .global_clustering import (
    K, AllPixelData, GlobalClusterRecord, GlobalClusters, check_global_clustering,
    collect_global_features, condition_paths, read_label_map, source_record,
)
from .global_k8_repeats import check_repeat_clustering, load_repeat_clusters, repeat_paths
from .global_manifest import GlobalData
from .global_seed_plan import global_kmeans_seed
from .input_validation import InputInventory, SampleInput
from .manifests import _digest, _read_json, _require, _write_json
from .neural import fp32_inference
from .spatial_reproducibility_metrics import WINDOWS, compare_maps, map_diagnostics, validate_map
from .spatial_reproducibility_reporting import (
    SAMPLE_FILES, SUMMARY_FILES, captions, color_alignment_contract, compose_display_mappings,
    display_mappings, label_overlap_counts, load_global_display, render_sample, render_summary,
    summarize_samples,
)
from .spatial_reproducibility_training import load_representation, seed_plan, training_source
from .training import runtime_record

SCOPE = "global_spatial_reproducibility_v2"
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


def clustering_seed(kmeans: int) -> int:
    """Return a fixed clustering stream shared by the three encoders."""
    _require(type(kmeans) is int and kmeans in (1, 2, 3), "Invalid clustering repeat")
    if kmeans == 2:
        return _seed("kmeans", "global_spatial_reproducibility", "seed2_v2", 2, 8)
    return global_kmeans_seed(kmeans)


def clustering_seed_plan() -> dict[str, object]:
    """Describe the active training, sampling and clustering streams."""
    plan = seed_plan()
    for row in plan["records"]:
        if row["purpose"] == "kmeans":
            row["seed"] = clustering_seed(row["kmeans_repeat"])
    _require(len({row["seed"] for row in plan["records"]}) == len(plan["records"]),
             "Spatial clustering seed collision")
    plan["scope"] = SCOPE
    return plan


def cluster_paths(experiment: Path, encoder: int, kmeans: int) -> tuple[Path, Path]:
    _require(type(encoder) is int and type(kmeans) is int
             and (encoder, kmeans) in MAP_IDS, "Invalid map identity")
    suffix = f"clustering/M11/encoder_{encoder}/kmeans_{kmeans}"
    return experiment / "results" / suffix, experiment / "checkpoints" / suffix


def map_path(
    experiment: Path, source: Path, sample_id: str, encoder: int, kmeans: int,
    *, sources: SpatialSources,
) -> Path:
    _require(type(encoder) is int and type(kmeans) is int
             and (encoder, kmeans) in MAP_IDS, "Invalid map identity")
    _require(bool(sample_id) and Path(sample_id).name == sample_id
             and sample_id not in (".", ".."), "Invalid sample ID")
    sources.validate_roots(experiment, source)
    if kmeans == 2:
        result = cluster_paths(experiment, encoder, kmeans)[0]
    elif encoder == 1:
        result = (condition_paths(source, "M11")[0] if kmeans == 1
                  else repeat_paths(source, "M11", kmeans)[0])
    else:
        result = cluster_paths(sources.artifacts, encoder, kmeans)[0]
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
    """Validate the global maps used for comparison and display alignment."""
    _require(tuple(sorted(sample.sample_id for sample in inventory.samples)) == data.train_sample_ids,
             "Prediction samples differ from the fixed global manifest")
    check_global_clustering(source, data, inventory, "M11")
    entries = [("canonical_initial", *condition_paths(source, "M11"))]
    for kmeans in (3,):
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


def fit_clusters(
    features: np.ndarray, sample_ids: tuple[str, ...], *, device: torch.device,
) -> GlobalClusters:
    """Use the original single-initialization recipe with the fixed clustering seed."""
    diagnostic = _diagnose(features, _dimension("M11"))
    _require(len(features) >= K and bool(sample_ids), "Insufficient global fit pixels")
    config = experiment_config()
    _require(cluster_contract()["chemomae"] == config["chemomae"]["version"],
             "ChemoMAE version differs")
    settings = config["clustering"]
    module = CosineKMeans(K, tol=settings["tol"], max_iter=settings["max_iter"],
                         device=device, random_state=clustering_seed(2)).to(dtype=torch.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(features))
    started = time.perf_counter()
    with fp32_inference(device):
        module.fit(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        _diagnose(module.centroids.detach().cpu().numpy(), _dimension("M11"))
        labels, distances = module.predict(tensor, return_dist=True)
        inertia = float(distances.gather(1, labels[:, None]).mean())
        occupancy = tuple(torch.bincount(labels, minlength=K).cpu().tolist())
    _require(math.isfinite(module.inertia_) and math.isfinite(inertia), "Nonfinite KMeans objective")
    record = GlobalClusterRecord(
        "M11", _dimension("M11"), K, clustering_seed(2), sample_ids, len(features), occupancy,
        float(module.inertia_), inertia, settings["max_iter"], settings["tol"], None,
        f"not_exposed_by_ChemoMAE_{cluster_contract()['chemomae']}", elapsed, str(device),
        diagnostic.unit_norm_absolute_error_max,
    )
    return GlobalClusters(module, record)


def save_clusters(fitted: GlobalClusters, path: Path, encoder: int) -> None:
    cluster_paths(path.parent, encoder, 2)
    _require(fitted.record.seed == clustering_seed(2), "Wrong clustering seed")
    with path.open("xb") as handle:
        np.savez_compressed(handle, centroids=fitted.centroids, metadata=np.array(json.dumps({
            "schema_version": 1, "scope": SCOPE, "encoder_repeat": encoder, "kmeans_repeat": 2,
            "runtime": cluster_contract(), "fit": asdict(fitted.record),
        })))


def load_clusters(path: Path, encoder: int, *, device: torch.device) -> GlobalClusters:
    cluster_paths(path.parent, encoder, 2)
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == {"centroids", "metadata"}, "Unexpected clustering centers")
        metadata = json.loads(str(saved["metadata"].item()))
        centers = saved["centroids"].copy()
    _require(metadata.get("schema_version") == 1 and metadata.get("scope") == SCOPE
             and metadata.get("encoder_repeat") == encoder and metadata.get("kmeans_repeat") == 2
             and metadata.get("runtime") == cluster_contract(), "Clustering center contract differs")
    fit = metadata["fit"]
    record = GlobalClusterRecord(**{**fit, "sample_ids": tuple(fit["sample_ids"]),
                                   "fit_occupancy": tuple(fit["fit_occupancy"])})
    settings = experiment_config()["clustering"]
    _require(record.condition == "M11" and record.dimension == _dimension("M11")
             and record.k == K and record.seed == clustering_seed(2)
             and record.max_iter == settings["max_iter"] and record.tol == settings["tol"]
             and centers.shape == (K, record.dimension) and bool(record.sample_ids)
             and record.fit_pixels >= K and len(record.fit_occupancy) == K
             and all(type(count) is int and count >= 0 for count in record.fit_occupancy)
             and sum(record.fit_occupancy) == record.fit_pixels
             and math.isfinite(record.reference_inertia) and math.isfinite(record.final_center_inertia),
             "Invalid clustering centers")
    _diagnose(centers, record.dimension)
    module = CosineKMeans(K, tol=record.tol, max_iter=record.max_iter,
                         device=device, random_state=record.seed).to(dtype=torch.float32)
    module.centroids.resize_(centers.shape)
    module.centroids.copy_(torch.from_numpy(centers).to(device))
    module.latent_dim, module.inertia_, module._fitted = record.dimension, record.reference_inertia, True
    return GlobalClusters(module, record)


@dataclass(frozen=True)
class SpatialSources:
    """Bind completed encoders and reusable maps to the active experiment."""

    artifacts: Path
    scope: ClassVar[str] = SCOPE

    def validate_roots(self, experiment: Path, source: Path) -> None:
        separate_roots(experiment, source)
        separate_roots(experiment, self.artifacts)
        separate_roots(self.artifacts, source)

    def _cached_cluster_contract(
        self, source: Path, data: GlobalData, encoder: int, kmeans: int,
    ) -> dict[str, object]:
        _require(encoder in (2, 3) and kmeans in (1, 3), "Invalid cached map identity")
        result, _ = cluster_paths(self.artifacts, encoder, kmeans)
        saved = _read_json(result / "run.json")["contract"]
        orchestration = saved["code_sha256"][Path(__file__).name]
        _require(isinstance(orchestration, str) and len(orchestration) == 64
                 and all(char in "0123456789abcdef" for char in orchestration),
                 "Invalid cached pipeline hash")
        # Completed inputs retain their producer identity. Numerical dependencies
        # must match the code used to extract and cluster current features.
        code = _code_hashes(("global_k8_repeats.py", "global_clustering.py", "cluster_pipeline.py",
                             "clustering.py", "neural.py", "spatial_reproducibility_training.py"))
        code[Path(__file__).name] = orchestration
        return {
            "schema_version": 1, "scope": "global_spatial_reproducibility_v1",
            "condition": "M11", "K": K, "encoder_repeat": encoder, "kmeans_repeat": kmeans,
            "protocol_sha256": _digest(self.artifacts / "protocol/completion.json"),
            "source": training_source(self.artifacts, source, data, encoder),
            "sample_ids": list(data.train_sample_ids), "fit_pixels": data.train_pixel_count,
            "settings": experiment_config()["clustering"], "runtime": cluster_contract(),
            "seed_plan": seed_plan(), "code_sha256": code,
        }

    def _check_cached_cluster(
        self, source: Path, data: GlobalData, inventory: InputInventory, encoder: int, kmeans: int,
    ) -> dict[str, object]:
        result, weights = cluster_paths(self.artifacts, encoder, kmeans)
        completion = check_sealed(result, "spatial_clustering_completed",
                                  self._cached_cluster_contract(source, data, encoder, kmeans))
        run = _read_json(result / "run.json")
        path = weights / "centers_k8.npz"
        _require(run["centers_sha256"] == _digest(path) and run.get("center_roundtrip_exact") is True,
                 "Cached centers changed")
        fitted = load_repeat_clusters(path, "M11", kmeans, device=torch.device("cpu"))
        _require(json.loads(json.dumps(asdict(fitted.record))) == run["fit"]
                 and fitted.record.sample_ids == data.train_sample_ids
                 and fitted.record.fit_pixels == data.train_pixel_count
                 and all(type(count) is int and count >= 0 for count in fitted.record.fit_occupancy),
                 "Cached fit provenance differs")
        _require({path.name for path in (result / "maps").iterdir()}
                 == {f"{sample.sample_id}.npz" for sample in inventory.samples}, "Cached maps differ")
        for sample in inventory.samples:
            read_label_map(result / "maps" / f"{sample.sample_id}.npz", sample)
        return completion

    def _snapshot(
        self, source: Path, data: GlobalData, inventory: InputInventory,
    ) -> dict[str, object]:
        reference = _source_snapshot(source, data, inventory)
        encoders = {str(e): training_source(self.artifacts, source, data, e) for e in (1, 2, 3)}
        maps = []
        for encoder in (2, 3):
            for kmeans in (1, 3):
                self._check_cached_cluster(source, data, inventory, encoder, kmeans)
                result, weights = cluster_paths(self.artifacts, encoder, kmeans)
                maps.append({"encoder_repeat": encoder, "kmeans_repeat": kmeans,
                             "results": str(result.resolve()),
                             "completion_sha256": _digest(result / "completion.json"),
                             "centers_sha256": _digest(weights / "centers_k8.npz")})
        return {"global": reference, "artifacts": str(self.artifacts.resolve()),
                "encoders": encoders, "cached_maps": maps}

    def _protocol_contract(
        self, experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
    ) -> dict[str, object]:
        self.validate_roots(experiment, source)
        return {
            "schema_version": 1, "scope": SCOPE, "condition": "M11", "K": K,
            "definition_version": "valid-edge-midpoint-chebyshev-v1",
            "windows": list(WINDOWS), "radii_pixels": [1, 2, 4],
            "boundary_matching": "many-to-one proximity; both directions; no orientation matching",
            "map_ids": [list(identity) for identity in MAP_IDS], "pairs": pair_plan(),
            "sample_ids": list(data.train_sample_ids), "fit_pixels": data.train_pixel_count,
            "prediction_pixels": sum(sample.saved_pixel_count for sample in inventory.samples),
            "seed_plan": clustering_seed_plan(), "inputs": self._snapshot(source, data, inventory),
            "fit_map_ids": [[e, 2] for e in (1, 2, 3)],
            "reused_map_ids": [[e, k] for e, k in MAP_IDS if k != 2],
            "code_sha256": _digest(Path(__file__)),
            "primary_aggregation": "all expected pairs defined; equal sample weight; no imputation",
            "held_out": False, "scientific_pass_threshold": None,
        }

    def prepare(
        self, experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
    ) -> dict[str, object]:
        self.validate_roots(experiment, source)
        output = experiment / "protocol"
        _require(not output.exists(), "Protocol exists; use --resume to verify it")
        contract = self._protocol_contract(experiment, source, data, inventory)
        audit = []
        if contract["inputs"]["global"]["regenerated_initial_available"]:
            for sample in inventory.samples:
                initial = read_label_map(map_path(experiment, source, sample.sample_id, 1, 1,
                                                  sources=self), sample)
                displayed = read_label_map(repeat_paths(source, "M11", 1)[0] / "maps"
                                          / f"{sample.sample_id}.npz", sample)
                audit.append({"sample_id": sample.sample_id,
                              "exact_label_array_equal": bool(np.array_equal(initial, displayed)),
                              "raw_label_changed_pixels": int(np.count_nonzero(initial != displayed)),
                              "comparison": compare_maps(initial, displayed, initial > 0)})
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
            _require(self._protocol_contract(experiment, source, data, inventory) == contract,
                     "Inputs changed during preparation")
            completion = seal(stage, "spatial_protocol_prepared")
            stage.rename(output)
        return completion

    def check_prepared(
        self, experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
    ) -> dict[str, object]:
        return check_sealed(experiment / "protocol", "spatial_protocol_prepared",
                            self._protocol_contract(experiment, source, data, inventory))

    def _cluster_contract(
        self, experiment: Path, source: Path, data: GlobalData, encoder: int,
    ) -> dict[str, object]:
        self.validate_roots(experiment, source)
        cluster_paths(experiment, encoder, 2)
        return {"schema_version": 1, "scope": SCOPE, "encoder_repeat": encoder, "kmeans_repeat": 2,
                "protocol_sha256": _digest(experiment / "protocol/completion.json"),
                "source": training_source(self.artifacts, source, data, encoder),
                "sample_ids": list(data.train_sample_ids), "fit_pixels": data.train_pixel_count,
                "seed_plan": clustering_seed_plan(), "settings": experiment_config()["clustering"],
                "runtime": cluster_contract(), "code_sha256": _code_hashes((
                    Path(__file__).name, "global_clustering.py", "cluster_pipeline.py", "clustering.py",
                    "neural.py", "spatial_reproducibility_training.py"))}

    def check_cluster(
        self, experiment: Path, source: Path, data: GlobalData, inventory: InputInventory, encoder: int,
    ) -> dict[str, object]:
        result, weights = cluster_paths(experiment, encoder, 2)
        completion = check_sealed(result, "spatial_clustering_completed",
                                           self._cluster_contract(experiment, source, data, encoder))
        run = _read_json(result / "run.json")
        path = weights / "centers_k8.npz"
        _require(run["centers_sha256"] == _digest(path) and run.get("center_roundtrip_exact") is True,
                 "Clustering centers changed")
        fitted = load_clusters(path, encoder, device=torch.device("cpu"))
        _require(json.loads(json.dumps(asdict(fitted.record))) == run["fit"]
                 and fitted.record.sample_ids == data.train_sample_ids
                 and fitted.record.fit_pixels == data.train_pixel_count, "Clustering fit provenance differs")
        _require({path.name for path in (result / "maps").iterdir()}
                 == {f"{sample.sample_id}.npz" for sample in inventory.samples}, "Clustering maps differ")
        for sample in inventory.samples:
            read_label_map(result / "maps" / f"{sample.sample_id}.npz", sample)
        return completion

    def cluster(
        self, experiment: Path, source: Path, data: GlobalData, inventory: InputInventory, encoder: int,
        *, device: torch.device, chunk_pixels: int = 1024, resume: bool = False,
    ) -> None:
        self.validate_roots(experiment, source)
        _require(type(chunk_pixels) is int and chunk_pixels > 0, "Invalid chunk size")
        _require(device.type == "cuda", "Production clustering requires a single CUDA device")
        self.check_prepared(experiment, source, data, inventory)
        result, weights = cluster_paths(experiment, encoder, 2)
        if result.exists() or weights.exists():
            _require(resume, "Clustering output exists; use --resume to verify and skip")
            self.check_cluster(experiment, source, data, inventory, encoder)
            return
        contract = self._cluster_contract(experiment, source, data, encoder)
        representation, reference = load_representation(self.artifacts, source, data, encoder, device=device)
        _require(reference == contract["source"], "Clustering encoder source differs")
        features = collect_global_features(data, representation, "M11", chunk_pixels=chunk_pixels)
        samples = {sample.sample_id: sample for sample in inventory.samples}
        started = time.perf_counter()
        with TemporaryDirectory(prefix=".spatial-cluster-", dir=experiment) as temporary:
            stage, weight_stage = Path(temporary) / "results", Path(temporary) / "weights"
            (stage / "maps").mkdir(parents=True)
            weight_stage.mkdir()
            fitted = fit_clusters(features, data.train_sample_ids, device=device)
            center_path = weight_stage / "centers_k8.npz"
            save_clusters(fitted, center_path, encoder)
            restored = load_clusters(center_path, encoder, device=device)
            _require(np.array_equal(fitted.centroids, restored.centroids)
                     and np.array_equal(fitted.predict(features[:32]), restored.predict(features[:32])),
                     "Clustering center roundtrip changed predictions")
            del fitted
            for sample_id in data.train_sample_ids:
                sample = samples[sample_id]
                with h5py.File(sample.path, "r") as handle:
                    output = LabelMap(handle["valid_spectrum_mask"][:], K)
                count = 0
                for batch in AllPixelData((sample,)).all_batches(chunk_pixels=chunk_pixels):
                    transformed = _transform_batch(representation, batch, "M11")
                    output.add(batch.pixel_row_col, restored.predict(transformed.values))
                    count += len(batch.snv)
                _require(count == sample.saved_pixel_count, f"{sample_id}: incomplete clustering map")
                np.savez_compressed(stage / "maps" / f"{sample_id}.npz", labels_k8=output.finish())
                print(f"encoder {encoder}, Cosine-KMeans 2: {sample_id}", flush=True)
            self.check_prepared(experiment, source, data, inventory)
            _require(self._cluster_contract(experiment, source, data, encoder) == contract,
                     "Clustering source changed during computation")
            _write_json(stage / "run.json", {"contract": contract, "execution": runtime_record(device),
                        "fit": asdict(restored.record), "chunk_pixels": chunk_pixels,
                        "centers_sha256": _digest(center_path), "center_roundtrip_exact": True,
                        "wall_seconds": time.perf_counter() - started})
            seal(stage, "spatial_clustering_completed")
            result.parent.mkdir(parents=True, exist_ok=True)
            weights.parent.mkdir(parents=True, exist_ok=True)
            weight_stage.rename(weights)
            try:
                stage.rename(result)
            except OSError:
                weights.rename(weight_stage)
                raise
        self.check_cluster(experiment, source, data, inventory, encoder)


def _read_maps(
    experiment: Path, source: Path, sample: SampleInput,
    *, sources: SpatialSources,
) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray]:
    with h5py.File(sample.path, "r") as handle:
        mask = handle["valid_spectrum_mask"][:]
    _require(mask.shape == (sample.height, sample.width) and np.isin(mask, (0, 1)).all()
             and int(np.count_nonzero(mask)) == sample.saved_pixel_count, "Source mask differs")
    mask = mask.astype(bool)
    maps = {}
    for encoder, kmeans in MAP_IDS:
        path = map_path(experiment, source, sample.sample_id, encoder, kmeans, sources=sources)
        with np.load(path, allow_pickle=False) as saved:
            _require(set(saved.files) == {"labels_k8"}, "Unexpected label fields")
            labels = saved["labels_k8"].copy()
        _require(labels.dtype == np.uint8, "Expected saved uint8 labels")
        validate_map(labels, mask)
        maps[encoder, kmeans] = labels
    return maps, mask


def _evaluation_contract(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
    *, include_overlap: bool, sources: SpatialSources,
) -> dict:
    sources.check_prepared(experiment, source, data, inventory)
    maps = []
    for encoder, kmeans in MAP_IDS:
        if kmeans == 2:
            sources.check_cluster(experiment, source, data, inventory, encoder)
        result = map_path(experiment, source, data.train_sample_ids[0], encoder, kmeans,
                          sources=sources).parent.parent
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
    *, include_overlap: bool = False, sources: SpatialSources,
) -> dict[str, object]:
    """Evaluate every full sample; missing maps fail instead of becoming empty boundaries."""
    experiment, source = separate_roots(experiment, source)
    output = experiment / "results/evaluation"
    _require(not output.exists(), "Evaluation exists; use --resume to verify and skip")
    contract = _evaluation_contract(experiment, source, data, inventory,
                                    include_overlap=include_overlap, sources=sources)
    started = time.perf_counter()
    evaluations, diagnostics, map_records = [], [], []
    samples = {sample.sample_id: sample for sample in inventory.samples}
    with TemporaryDirectory(prefix=".spatial-evaluation-", dir=experiment) as temporary:
        stage = Path(temporary) / "evaluation"
        (stage / "samples").mkdir(parents=True)
        for sample_id in data.train_sample_ids:
            maps, mask = _read_maps(experiment, source, samples[sample_id], sources=sources)
            map_rows = []
            for (encoder, kmeans), labels in maps.items():
                row = {"sample_id": sample_id, "encoder_repeat": encoder,
                       "kmeans_repeat": kmeans, **map_diagnostics(labels, mask)}
                map_rows.append(row)
                diagnostics.append({key: json.dumps(value) if isinstance(value, list) else value
                                    for key, value in row.items()})
                path = map_path(experiment, source, sample_id, encoder, kmeans, sources=sources)
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
                                      include_overlap=include_overlap, sources=sources) == contract,
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
    *, sources: SpatialSources,
) -> dict[str, object]:
    output = experiment / "results/evaluation"
    include_overlap = _read_json(output / "run.json")["contract"]["include_overlap"]
    _require(type(include_overlap) is bool, "Invalid overlap option")
    contract = _evaluation_contract(experiment, source, data, inventory,
                                    include_overlap=include_overlap, sources=sources)
    completion = check_sealed(output, "spatial_evaluation_completed", contract)
    records = _read_json(output / "map_sources.json")["maps"]
    _require([(row["sample_id"], row["encoder_repeat"], row["kmeans_repeat"]) for row in records]
             == [(sample, e, k) for sample in data.train_sample_ids for e, k in MAP_IDS],
             "Evaluation map coverage differs")
    for row in records:
        path = map_path(experiment, source, row["sample_id"], row["encoder_repeat"],
                        row["kmeans_repeat"], sources=sources)
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


def _report_contract(
    experiment: Path, source: Path, sample_ids: tuple[str, ...], *, dpi: int,
    sources: SpatialSources,
) -> dict[str, object]:
    display = load_global_display(source, sample_ids)
    return {"schema_version": 3, "scope": SCOPE, "dpi": dpi,
            "evaluation_sha256": _digest(experiment / "results/evaluation/completion.json"),
            "matplotlib": version("matplotlib"), "numpy": np.__version__,
            "scipy": version("scipy"), "color_alignment": color_alignment_contract(),
            "global_display": display.provenance,
            "figure_style": {"titles": False, "spatial_axes": False,
                             "sample_id": "below image group", "clustering": "Cosine-KMeans",
                             "panel_explanations": "captions.json"},
            "code_sha256": _code_hashes((Path(__file__).name,
                                          "spatial_reproducibility_reporting.py",
                                          "spatial_reproducibility_metrics.py"))}


def write_report(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory, *, dpi: int = 160,
    sources: SpatialSources,
) -> dict[str, object]:
    experiment, source = separate_roots(experiment, source)
    _require(type(dpi) is int and dpi > 0, "DPI must be a positive integer")
    output = experiment / "results/report"
    _require(not output.exists(), "Report exists; use --resume to verify and skip")
    check_evaluation(experiment, source, data, inventory, sources=sources)
    contract = _report_contract(experiment, source, data.train_sample_ids, dpi=dpi, sources=sources)
    global_display = load_global_display(source, data.train_sample_ids)
    evaluation_root = experiment / "results/evaluation"
    samples = {sample.sample_id: sample for sample in inventory.samples}
    # One permutation per run across ALL samples preserves the meaning of colors
    # between specimen panels. Counts use source labels; never relabel saved maps.
    counts = {identity: np.zeros((8, 8), dtype=np.int64) for identity in MAP_IDS}
    for sample_id in data.train_sample_ids:
        maps, mask = _read_maps(experiment, source, samples[sample_id], sources=sources)
        for identity in MAP_IDS:
            counts[identity] += label_overlap_counts(maps[1, 1], maps[identity], mask)
    initial_lookups, assignments = display_mappings(counts)
    display_lookups = compose_display_mappings(initial_lookups, global_display.lookup)
    with TemporaryDirectory(prefix=".spatial-report-", dir=experiment) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        _write_json(stage / "color_matching.json", {
            "schema_version": 2, "contract": color_alignment_contract(),
            "sample_ids": list(data.train_sample_ids), "initial_mappings": assignments,
            "global_display": global_display.provenance,
            "mappings": [{"encoder_repeat": e, "kmeans_repeat": k,
                          "target_to_display_including_background": display_lookups[e, k].tolist()}
                         for e, k in MAP_IDS],
        })
        summary = _read_json(evaluation_root / "summary.json")
        _write_json(stage / "captions.json", captions(summary))
        for sample_id in data.train_sample_ids:
            maps, mask = _read_maps(experiment, source, samples[sample_id], sources=sources)
            evaluation = _read_json(evaluation_root / "samples" / f"{sample_id}.json")
            render_sample(sample_id, maps, mask, evaluation, stage / "samples" / sample_id,
                          dpi=dpi, display_lookups=display_lookups, palette=global_display.palette)
            print(f"{sample_id}: spatial figures saved", flush=True)
        render_summary(summary, stage / "summary", dpi=dpi)
        check_evaluation(experiment, source, data, inventory, sources=sources)
        _require(_report_contract(experiment, source, data.train_sample_ids,
                                  dpi=dpi, sources=sources) == contract,
                 "Report source changed")
        _write_json(stage / "run.json", {"contract": contract, "sample_ids": list(data.train_sample_ids),
                                        "execution": runtime_record(torch.device("cpu")),
                                        "label_colors": "pooled overlap to initial raw IDs, then "
                                                        "saved global-fit display IDs and palette"})
        completion = seal(stage, "spatial_report_completed")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output)
    return completion


def check_report(
    experiment: Path, source: Path, data: GlobalData, inventory: InputInventory,
    *, sources: SpatialSources,
) -> dict[str, object]:
    check_evaluation(experiment, source, data, inventory, sources=sources)
    output = experiment / "results/report"
    run = _read_json(output / "run.json")
    completion = check_sealed(output, "spatial_report_completed",
                              _report_contract(experiment, source, data.train_sample_ids,
                                               dpi=run["contract"]["dpi"], sources=sources))
    _require(run["sample_ids"] == list(data.train_sample_ids), "Report sample coverage differs")
    matching = _read_json(output / "color_matching.json")
    global_display = load_global_display(source, data.train_sample_ids)
    _require(matching.get("schema_version") == 2
             and matching.get("contract") == color_alignment_contract()
             and matching.get("sample_ids") == list(data.train_sample_ids)
             and matching.get("global_display") == global_display.provenance,
             "Report color-alignment provenance differs")
    mappings = matching["mappings"]
    _require([(row["encoder_repeat"], row["kmeans_repeat"]) for row in mappings] == list(MAP_IDS),
             "Report color mappings must cover all nine maps")
    initial_mappings = matching["initial_mappings"]
    _require([(row["encoder_repeat"], row["kmeans_repeat"]) for row in initial_mappings]
             == list(MAP_IDS), "Initial-label mappings must cover all nine maps")
    initial_lookups, expected_initial = display_mappings({
        (row["encoder_repeat"], row["kmeans_repeat"]): np.asarray(row["contingency"])
        for row in initial_mappings})
    _require(initial_mappings == expected_initial, "Initial-label mappings differ from overlap counts")
    composed = compose_display_mappings(initial_lookups, global_display.lookup)
    expected_mappings = [{"encoder_repeat": e, "kmeans_repeat": k,
                          "target_to_display_including_background": composed[e, k].tolist()}
                         for e, k in MAP_IDS]
    _require(mappings == expected_mappings, "Display mappings differ from global-fit composition")
    _require(_read_json(output / "captions.json")
             == captions(_read_json(experiment / "results/evaluation/summary.json")),
             "Publication captions differ from the evaluated sample counts")
    _require({path.name for path in (output / "samples").iterdir()}
             == set(data.train_sample_ids), "Report sample directories differ")
    for sample_id in data.train_sample_ids:
        _require({path.name for path in (output / "samples" / sample_id).iterdir()} == set(SAMPLE_FILES),
                 f"{sample_id}: report figure/frequency coverage differs")
    _require({path.name for path in (output / "summary").iterdir()} == set(SUMMARY_FILES),
             "Report summary figure coverage differs")
    return {"status": completion["status"], "samples": len(data.train_sample_ids),
            "maps_per_sample": 9, "pairs_per_sample": 36, "scientific_pass_threshold": None}
