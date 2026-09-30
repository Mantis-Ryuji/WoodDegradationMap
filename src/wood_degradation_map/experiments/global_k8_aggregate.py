"""Equal-sample, three-repeat summaries for the global K=8 diagnostics."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .config import REPEATS
from .global_k8_evaluation import RESULT_DIRECTORY, check_condition_evaluation, condition_output
from .global_manifest import GLOBAL_CONDITIONS, GlobalData
from .global_seed_plan import global_kmeans_plan
from .input_validation import InputInventory
from .manifests import _digest, _read_json, _require, _write_json

METRICS = ("adjusted_lla_3", "adjusted_lla_5", "adjusted_lla_9", "silhouette",
           "used_clusters", "k_usage_rate", "maximum_occupancy")
REPORT_METRICS = (*METRICS[:4], "ari", *METRICS[4:])


def summarize_repeats(
    samples_by_repeat: Mapping[int, Mapping[str, Mapping[str, object]]],
    metric: str,
) -> dict[str, object]:
    """Use the same defined samples in every repeat for both reported SDs."""
    if metric not in METRICS or set(samples_by_repeat) != set(REPEATS):
        raise ValueError("Expected one planned metric and all three repeats")
    ids = set(samples_by_repeat[1])
    if not ids or any(set(samples_by_repeat[repeat]) != ids for repeat in REPEATS):
        raise ValueError("Global repeats must cover identical nonempty sample IDs")
    ordered = sorted(ids)
    availability: dict[int, dict[str, object]] = {}
    common = []
    for repeat in REPEATS:
        defined, undefined = [], {}
        for sample_id in ordered:
            row = samples_by_repeat[repeat][sample_id]
            value = row.get(metric)
            if value is None:
                reason = row.get("undefined_reasons", {}).get(metric)
                if not isinstance(reason, str) or not reason:
                    raise ValueError(f"{sample_id}: undefined {metric} lacks a reason")
                undefined[sample_id] = reason
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not np.isfinite(value)):
                raise ValueError(f"{sample_id}: invalid {metric} value")
            else:
                defined.append(sample_id)
        availability[repeat] = {"defined_samples": len(defined), "undefined": undefined}
    for sample_id in ordered:
        if all(samples_by_repeat[repeat][sample_id].get(metric) is not None for repeat in REPEATS):
            common.append(sample_id)
    result: dict[str, object] = {
        "metric": metric, "total_samples": len(ordered), "common_samples": len(common),
        "common_sample_ids": common, "excluded_sample_ids": sorted(ids - set(common)),
        "availability": availability,
        "mean": None, "sample_sd": None, "repeat_sd": None,
        "repeat_macro_means": {repeat: None for repeat in REPEATS},
        "sample_means": {},
        "mean_undefined_reason": "no_common_samples" if not common else None,
        "sample_sd_undefined_reason": "fewer_than_two_common_samples" if len(common) < 2 else None,
        "repeat_sd_undefined_reason": "no_common_samples" if not common else None,
    }
    if common:
        values = np.array([[samples_by_repeat[repeat][sample_id][metric] for repeat in REPEATS]
                           for sample_id in common], dtype=np.float64)
        sample_means = values.mean(axis=1)
        repeat_means = values.mean(axis=0)
        result.update({
            "mean": float(repeat_means.mean()),
            "sample_sd": float(sample_means.std(ddof=1)) if len(common) > 1 else None,
            "repeat_sd": float(repeat_means.std(ddof=1)),
            "repeat_macro_means": dict(zip(REPEATS, map(float, repeat_means), strict=True)),
            "sample_means": dict(zip(common, map(float, sample_means), strict=True)),
        })
    return result


def summarize_ari(sample_means: Mapping[str, float | None]) -> dict[str, object]:
    """ARI's three paired comparisons are not three independent run scores."""
    if not sample_means:
        raise ValueError("No global ARI samples")
    common = sorted(sample_id for sample_id, value in sample_means.items() if value is not None)
    values = np.array([sample_means[sample_id] for sample_id in common], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite ARI")
    return {
        "metric": "ari", "total_samples": len(sample_means), "common_samples": len(common),
        "common_sample_ids": common,
        "excluded_sample_ids": sorted(set(sample_means) - set(common)),
        "mean": float(values.mean()) if len(values) else None,
        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "repeat_sd": None,
        "mean_undefined_reason": "no_defined_samples" if not len(values) else None,
        "sample_sd_undefined_reason": "fewer_than_two_defined_samples" if len(values) < 2 else None,
    }


def report_contract(
    experiment: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    """Require all six conditions and all three KMeans runs before aggregation."""
    sources = {}
    for condition in GLOBAL_CONDITIONS:
        check_condition_evaluation(experiment, data, inventory, condition)
        sources[condition] = _digest(condition_output(experiment, condition) / "completion.json")
    return {
        "schema_version": 1, "scope": "global_k8_3seed_summary",
        "K": 8, "conditions": list(GLOBAL_CONDITIONS), "repeats": list(REPEATS),
        "seed_plan": global_kmeans_plan(), "sources": sources,
        "code_sha256": _digest(Path(__file__)),
        "metrics": list(REPORT_METRICS),
    }


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    experiment: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    """Save per-sample results and macro summaries without aligning cluster IDs."""
    output = experiment / RESULT_DIRECTORY / "report"
    _require(not output.exists(), "Global K8 report exists; use check")
    contract = report_contract(experiment, data, inventory)
    summary_rows, sample_rows, occupancy_rows, pooled_rows, ari_pair_rows = [], [], [], [], []
    detailed: dict[str, object] = {}
    for condition in GLOBAL_CONDITIONS:
        directory = condition_output(experiment, condition)
        evaluation_run = _read_json(directory / "run.json")
        samples = _read_json(directory / "samples.json")["samples"]
        aris = _read_json(directory / "ari_pairs.json")["samples"]
        expected_pixels = sum(sample.saved_pixel_count for sample in inventory.samples)
        for repeat in REPEATS:
            counts = evaluation_run["pooled_cluster_counts"][str(repeat)]
            _require(len(counts) == 8 and sum(counts) == expected_pixels,
                     "Global pooled occupancy differs from inventory")
            for cluster, pixels in enumerate(counts, start=1):
                pooled_rows.append({"condition": condition, "repeat": repeat,
                                    "cluster": cluster, "pixels": pixels,
                                    "fraction": pixels / expected_pixels})
        by_repeat: dict[int, dict[str, dict[str, object]]] = {repeat: {} for repeat in REPEATS}
        for row in samples:
            repeat, sample_id = row["repeat"], row["sample_id"]
            _require(repeat in REPEATS and sample_id not in by_repeat[repeat],
                     "Duplicate or invalid global sample/repeat")
            by_repeat[repeat][sample_id] = row
            sample_rows.append({"condition": condition, "sample_id": sample_id,
                                "repeat": repeat, "valid_pixels": row["valid_pixels"],
                                **{metric: row.get(metric) for metric in METRICS},
                                "undefined_reasons": json.dumps(
                                    row["undefined_reasons"], ensure_ascii=False)})
            _require(len(row["occupancy"]) == 8, "Expected eight occupancy fractions")
            for cluster, fraction in enumerate(row["occupancy"], start=1):
                occupancy_rows.append({"condition": condition, "sample_id": sample_id,
                                       "repeat": repeat, "cluster": cluster,
                                       "fraction": fraction})
        metrics = {metric: summarize_repeats(by_repeat, metric) for metric in METRICS}
        ari_by_sample = {}
        for row in aris:
            sample_id = row["sample_id"]
            _require(sample_id not in ari_by_sample, "Duplicate global ARI sample")
            ari_by_sample[sample_id] = row["mean"]
            _require(len(row["pairs"]) == 3, "Expected all three KMeans ARI pairs")
            for pair in row["pairs"]:
                ari_pair_rows.append({"condition": condition, "sample_id": sample_id,
                                      "repeat_a": pair["repeats"][0],
                                      "repeat_b": pair["repeats"][1],
                                      "ari": pair["value"],
                                      "undefined_reason": pair["undefined_reason"],
                                      "used_clusters_a": pair["used_clusters"][0],
                                      "used_clusters_b": pair["used_clusters"][1],
                                      "degeneracy_flags": json.dumps(pair["degeneracy_flags"])})
        _require(set(ari_by_sample) == set(by_repeat[1]), "ARI/sample coverage differs")
        metrics["ari"] = summarize_ari(ari_by_sample)
        detailed[condition] = metrics
        for metric in REPORT_METRICS:
            summary = metrics[metric]
            repeat_means = summary.get("repeat_macro_means", {})
            summary_rows.append({
                "condition": condition, "metric": metric,
                "total_samples": summary["total_samples"],
                "common_samples": summary["common_samples"],
                "mean": summary["mean"], "sample_sd": summary["sample_sd"],
                "repeat_sd": summary["repeat_sd"],
                "repeat_1_mean": repeat_means.get(1),
                "repeat_2_mean": repeat_means.get(2),
                "repeat_3_mean": repeat_means.get(3),
                "excluded_sample_ids": json.dumps(summary["excluded_sample_ids"]),
                "mean_undefined_reason": summary["mean_undefined_reason"],
            })
    _require(report_contract(experiment, data, inventory) == contract,
             "Global K8 evaluation changed during aggregation")
    with TemporaryDirectory(prefix=".global-k8-report-", dir=experiment) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        _write_json(stage / "run.json", {"contract": contract,
            "definitions": {
                "LLA": "occupancy-adjusted local label agreement; windows 3, 5, 9",
                "silhouette": "pooled all-valid-pixel cosine distances, then equal-sample mean",
                "ARI": "three KMeans repeat pairs per sample, then equal-sample mean",
                "K_usage": "used clusters / 8; used_clusters also retained",
                "occupancy": "original cluster fractions per sample/repeat; maximum fraction summarized",
                "SD": "sample SD of three-repeat means; repeat SD of equal-sample macro means",
                "repeat_variation": "KMeans initialization only; saved representation is fixed",
                "cluster_ids": "original, not aligned across repeats",
                "interpretation": "descriptive full-fit diagnostics; no held-out test claim",
                "LFR": "not computed",
            }})
        _write_json(stage / "summary.json", {"conditions": detailed})
        _write_csv(stage / "summary.csv", [
            "condition", "metric", "total_samples", "common_samples", "mean", "sample_sd",
            "repeat_sd", "repeat_1_mean", "repeat_2_mean", "repeat_3_mean",
            "excluded_sample_ids", "mean_undefined_reason"], summary_rows)
        _write_csv(stage / "sample_metrics.csv", [
            "condition", "sample_id", "repeat", "valid_pixels", *METRICS,
            "undefined_reasons"], sample_rows)
        _write_csv(stage / "occupancy.csv", [
            "condition", "sample_id", "repeat", "cluster", "fraction"], occupancy_rows)
        _write_csv(stage / "pooled_occupancy.csv", [
            "condition", "repeat", "cluster", "pixels", "fraction"], pooled_rows)
        _write_csv(stage / "ari_pairs.csv", [
            "condition", "sample_id", "repeat_a", "repeat_b", "ari",
            "undefined_reason", "used_clusters_a", "used_clusters_b",
            "degeneracy_flags"], ari_pair_rows)
        names = ("run.json", "summary.json", "summary.csv", "sample_metrics.csv",
                 "occupancy.csv", "pooled_occupancy.csv", "ari_pairs.csv")
        hashes = {name: _digest(stage / name) for name in names}
        completion = {"status": "global_k8_3seed_report_completed", "checks_passed": True,
                      "conditions": list(GLOBAL_CONDITIONS), "samples": len(data.train_sample_ids),
                      "artifact_sha256": hashes}
        _write_json(stage / "completion.json", completion)
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output)
    return completion


def check_report(
    experiment: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    output = experiment / RESULT_DIRECTORY / "report"
    run, completion = _read_json(output / "run.json"), _read_json(output / "completion.json")
    _require(run.get("contract") == report_contract(experiment, data, inventory)
             and completion.get("status") == "global_k8_3seed_report_completed"
             and completion.get("checks_passed") is True
             and completion.get("conditions") == list(GLOBAL_CONDITIONS)
             and completion.get("samples") == len(data.train_sample_ids),
             "Global K8 three-seed report contract differs")
    hashes = completion.get("artifact_sha256")
    _require(isinstance(hashes, dict)
             and set(hashes) == {"run.json", "summary.json", "summary.csv",
                                 "sample_metrics.csv", "occupancy.csv",
                                 "pooled_occupancy.csv", "ari_pairs.csv"}
             and all(_digest(output / name) == value for name, value in hashes.items()),
             "Global K8 three-seed report artifact changed")
    return {"status": "validated_global_k8_3seed_report", "checks_passed": True,
            "conditions": list(GLOBAL_CONDITIONS), "samples": len(data.train_sample_ids),
            "output": str(output)}
