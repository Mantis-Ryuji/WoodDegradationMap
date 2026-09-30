"""Evaluate three frozen-representation global KMeans fits at K=8, without LFR."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch

from .cluster_pipeline import _transform_batch
from .config import REPEATS
from .diagnostic_metrics import _ari_pair
from .global_clustering import (
    K, AllPixelData, load_global_representation, read_label_map,
)
from .global_k8_repeats import check_repeat_clustering, repeat_paths
from .global_manifest import GLOBAL_CONDITIONS, GlobalData
from .global_seed_plan import global_kmeans_plan
from .global_silhouette import PooledCosineSilhouette
from .input_validation import InputInventory
from .manifests import _digest, _read_json, _require, _write_json
from .spatial_metrics import local_label_agreement

RESULT_DIRECTORY = Path("results/global_k8_3seed_v1")


def condition_output(experiment: Path, condition: str) -> Path:
    _require(condition in GLOBAL_CONDITIONS, "Unknown global condition")
    return experiment / RESULT_DIRECTORY / "conditions" / condition


def map_path(experiment: Path, condition: str, repeat: int, sample_id: str) -> Path:
    results, _ = repeat_paths(experiment, condition, repeat)
    return results / "maps" / f"{sample_id}.npz"


def evaluation_contract(
    experiment: Path, data: GlobalData, inventory: InputInventory, condition: str,
) -> dict[str, object]:
    _require(condition in GLOBAL_CONDITIONS, "Unknown global condition")
    sources = {}
    for repeat in REPEATS:
        check_repeat_clustering(experiment, data, inventory, condition, repeat)
        results, _ = repeat_paths(experiment, condition, repeat)
        sources[str(repeat)] = _digest(results / "completion.json")
    directory = Path(__file__).parent
    return {
        "schema_version": 1, "scope": "global_k8_3seed_evaluation",
        "condition": condition, "K": K, "repeats": list(REPEATS),
        "seed_plan": global_kmeans_plan(),
        "representation": "one saved global fit per condition, fixed across KMeans repeats",
        "pixel_scope": "all saved valid pixels of every global sample",
        "held_out": False,
        "sources": sources,
        "manifest_sha256": _digest(experiment / "manifests/complete.json"),
        "code_sha256": {name: _digest(directory / name) for name in (
            "global_k8_evaluation.py", "global_silhouette.py", "spatial_metrics.py",
            "diagnostic_metrics.py", "cluster_pipeline.py")},
    }


def evaluate_condition(
    experiment: Path, data: GlobalData, inventory: InputInventory, condition: str,
    *, device: torch.device, chunk_pixels: int = 1024,
) -> dict[str, object]:
    """Extract each fixed feature chunk twice, sharing it across all three labelings."""
    _require(type(chunk_pixels) is int and chunk_pixels > 0, "Invalid chunk size")
    output = condition_output(experiment, condition)
    _require(not output.exists(), f"{condition}: evaluation output exists; use check or --resume")
    contract = evaluation_contract(experiment, data, inventory, condition)
    representation, representation_source = load_global_representation(
        experiment, data, condition, device=device)
    samples = {sample.sample_id: sample for sample in inventory.samples}
    _require(set(samples) == set(data.train_sample_ids), "Global sample set differs")
    dimensions = 256 if condition == "B0" else 16
    pooled = {repeat: PooledCosineSilhouette(K, dimensions) for repeat in REPEATS}
    rows: dict[tuple[str, int], dict[str, object]] = {}
    ari_rows: list[dict[str, object]] = []
    for sample_id in data.train_sample_ids:
        sample = samples[sample_id]
        with h5py.File(sample.path, "r") as handle:
            valid = handle["valid_spectrum_mask"][:].astype(bool)
        maps = {repeat: read_label_map(map_path(experiment, condition, repeat, sample_id), sample)
                for repeat in REPEATS}
        if int(valid.sum()) != sample.saved_pixel_count:
            raise ValueError(f"{sample_id}: valid pixel count changed")
        flat = {repeat: maps[repeat][valid] for repeat in REPEATS}
        pairs = [_ari_pair(flat[left], flat[right], K, (left, right))
                 for left, right in ((1, 2), (1, 3), (2, 3))]
        ari_rows.append({"sample_id": sample_id, "valid_pixels": sample.saved_pixel_count,
                         "pairs": [asdict(pair) for pair in pairs],
                         "mean": (float(np.mean([pair.value for pair in pairs], dtype=np.float64))
                                  if sample.saved_pixel_count >= 2 else None),
                         "undefined_reason": (None if sample.saved_pixel_count >= 2
                                              else "fewer_than_two_valid_pixels")})
        for repeat in REPEATS:
            spatial = local_label_agreement(maps[repeat], valid, k=K)
            row: dict[str, object] = {
                "sample_id": sample_id, "repeat": repeat,
                "valid_pixels": sample.saved_pixel_count,
                "used_clusters": spatial.used_clusters,
                "k_usage_rate": spatial.used_clusters / K,
                "maximum_occupancy": spatial.maximum_occupancy,
                "occupancy": list(spatial.occupancy),
                "undefined_reasons": {},
            }
            for window in spatial.windows:
                metric = f"adjusted_lla_{window.window}"
                row[metric] = window.adjusted_lla
                if window.adjusted_lla is None:
                    row["undefined_reasons"][metric] = ";".join(window.adjusted_undefined_reasons)
            rows[sample_id, repeat] = row
        count = 0
        for batch in AllPixelData((sample,)).all_batches(chunk_pixels=chunk_pixels):
            values = _transform_batch(representation, batch, condition).values
            coordinates = tuple(batch.pixel_row_col.T)
            for repeat in REPEATS:
                pooled[repeat].add(values, maps[repeat][coordinates])
            count += len(values)
        _require(count == sample.saved_pixel_count, f"{sample_id}: incomplete feature coverage")
        print(f"{condition} {sample_id}: LLA, occupancy and pooled silhouette pass 1", flush=True)
    reasons = {repeat: pooled[repeat].finalize() for repeat in REPEATS}
    _require(all(int(pooled[repeat].counts.sum()) == sum(s.saved_pixel_count for s in inventory.samples)
                 for repeat in REPEATS), "Pooled feature count differs")
    for sample_id in data.train_sample_ids:
        sample = samples[sample_id]
        maps = {repeat: read_label_map(map_path(experiment, condition, repeat, sample_id), sample)
                for repeat in REPEATS}
        totals = {repeat: 0.0 for repeat in REPEATS}
        count = 0
        for batch in AllPixelData((sample,)).all_batches(chunk_pixels=chunk_pixels):
            values = _transform_batch(representation, batch, condition).values
            coordinates = tuple(batch.pixel_row_col.T)
            for repeat in REPEATS:
                if reasons[repeat] is None:
                    scores = pooled[repeat].score(values, maps[repeat][coordinates])
                    totals[repeat] += float(scores.sum(dtype=np.float64))
            count += len(values)
        _require(count == sample.saved_pixel_count, f"{sample_id}: incomplete silhouette coverage")
        for repeat in REPEATS:
            row = rows[sample_id, repeat]
            row["silhouette"] = (totals[repeat] / count if reasons[repeat] is None else None)
            if reasons[repeat] is not None:
                row["undefined_reasons"]["silhouette"] = reasons[repeat]
        print(f"{condition} {sample_id}: pooled silhouette pass 2", flush=True)
    _require(evaluation_contract(experiment, data, inventory, condition) == contract,
             "Global evaluation source changed during computation")
    with TemporaryDirectory(prefix=".global-k8-evaluation-", dir=experiment) as temporary:
        stage = Path(temporary) / condition
        stage.mkdir()
        _write_json(stage / "run.json", {
            "contract": contract, "representation_source": representation_source,
            "execution_device": str(device), "chunk_pixels": chunk_pixels,
            "silhouette_method": "pooled exact cosine via cluster vector sums; FP64 accumulation",
            "silhouette_undefined_reason": reasons,
            "pooled_cluster_counts": {str(repeat): pooled[repeat].counts.tolist()
                                      for repeat in REPEATS},
        })
        _write_json(stage / "samples.json", {"samples": [rows[sample_id, repeat]
                                                          for sample_id in data.train_sample_ids
                                                          for repeat in REPEATS]})
        _write_json(stage / "ari_pairs.json", {"samples": ari_rows})
        hashes = {name: _digest(stage / name)
                  for name in ("run.json", "samples.json", "ari_pairs.json")}
        report = {"status": "global_k8_evaluation_completed", "condition": condition,
                  "checks_passed": True, "sample_count": len(data.train_sample_ids),
                  "repeat_count": len(REPEATS), "artifact_sha256": hashes}
        _write_json(stage / "completion.json", report)
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output)
    return report


def check_condition_evaluation(
    experiment: Path, data: GlobalData, inventory: InputInventory, condition: str,
) -> dict[str, object]:
    """Bind saved scores to all three current maps and exact sample coverage."""
    output = condition_output(experiment, condition)
    run, completion = _read_json(output / "run.json"), _read_json(output / "completion.json")
    _require(run.get("contract") == evaluation_contract(experiment, data, inventory, condition),
             "Global K8 evaluation source changed")
    _require(completion.get("status") == "global_k8_evaluation_completed"
             and completion.get("condition") == condition
             and completion.get("checks_passed") is True
             and completion.get("sample_count") == len(data.train_sample_ids)
             and completion.get("repeat_count") == len(REPEATS),
             "Global K8 evaluation is incomplete")
    hashes = completion.get("artifact_sha256")
    _require(isinstance(hashes, dict)
             and set(hashes) == {"run.json", "samples.json", "ari_pairs.json"}
             and all(_digest(output / name) == value for name, value in hashes.items()),
             "Global K8 evaluation artifact changed")
    rows = _read_json(output / "samples.json")["samples"]
    aris = _read_json(output / "ari_pairs.json")["samples"]
    _require([(row["sample_id"], row["repeat"]) for row in rows]
             == [(sample_id, repeat) for sample_id in data.train_sample_ids for repeat in REPEATS]
             and [row["sample_id"] for row in aris] == list(data.train_sample_ids),
             "Global K8 evaluation sample/repeat coverage differs")
    return {"status": "validated_global_k8_evaluation", "condition": condition,
            "checks_passed": True, "samples": len(aris), "output": str(output)}
