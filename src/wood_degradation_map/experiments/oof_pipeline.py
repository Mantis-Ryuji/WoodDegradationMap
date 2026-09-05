"""Persist complete OOF summaries, aligned repeat ARI and prespecified contrasts."""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .aggregation import (
    REPEATED_METRICS, RepeatedSummary, ScoreRecord, SummaryRow, _number, _summarize,
    _unavailable, _validated_records, aggregate_ari, aggregate_scores, paired_difference,
)
from .config import CLUSTER_COUNTS, CONDITIONS, FOLDS, REPEATS, experiment_config
from .data import FoldData
from .diagnostic_metrics import repeat_ari
from .evaluation_pipeline import _code_hashes as evaluation_code_hashes, check_evaluations
from .input_validation import InputInventory
from .lfr import PixelLabels
from .manifests import CVManifest, _digest, _read_json, _write_json

PLANNED_PAIRS = (
    ("M11", "B0"), ("M11", "B1"), ("M11", "M00"), ("M00", "B0"), ("M00", "B1"),
    ("M00", "A0"), ("M10", "M00"), ("M01", "M00"), ("M11", "M10"), ("M11", "M01"),
)
FACTORIAL_CONDITIONS = ("M11", "M10", "M01", "M00")
PRIMARY_METRICS = ("lla_3", "lla_5", "lla_9", "lfr_noise", "lfr_shift", "lfr_both")


def interaction_contrast(
    records: Sequence[ScoreRecord], *, expected_test_folds: Mapping[str, int], metric: str, k: int,
) -> RepeatedSummary:
    """Compute (M11-M10)-(M01-M00) before the four-condition/three-repeat intersection."""
    if metric not in PRIMARY_METRICS:
        raise ValueError("The planned interaction applies to the six primary metrics")
    indexed, availability = _validated_records(
        records, expected_test_folds, FACTORIAL_CONDITIONS, metric, k)
    rows = []
    for sample in sorted(expected_test_folds):
        for repeat in REPEATS:
            sources = tuple(indexed[condition, sample, repeat] for condition in FACTORIAL_CONDITIONS)
            unavailable = tuple(_unavailable(row) for row in sources if row.status != "defined")
            value = None
            if not unavailable:
                a, b, c, d = (_number(row.value) for row in sources)
                value = float((a - b) - (c - d))
                if not np.isfinite(value):
                    raise ValueError("Nonfinite interaction contrast")
            rows.append(SummaryRow(sample, expected_test_folds[sample], repeat, value, unavailable))
    return _summarize(tuple(rows), availability, expected_test_folds,
                      "(M11 - M10) - (M01 - M00)", metric, k)


def _conditions(conditions: tuple[str, ...]) -> tuple[str, ...]:
    known = tuple(condition.condition_id for condition in CONDITIONS)
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= set(known):
        raise ValueError("Expected unique, nonempty planned conditions")
    return tuple(condition for condition in known if condition in conditions)


def _output(experiment: Path, snapshot: str) -> Path:
    if not isinstance(snapshot, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", snapshot):
        raise ValueError("Snapshot must contain 1..100 ASCII letters, digits, underscores or hyphens")
    return experiment / "results/oof" / snapshot


def _codes() -> dict[str, str]:
    return {**evaluation_code_hashes(), "oof_pipeline.py": _digest(Path(__file__))}


def _sources(
    experiment: Path, inventory: InputInventory, manifest: CVManifest, conditions: tuple[str, ...],
) -> tuple[dict[str, int], list[dict[str, object]]]:
    ledger = []
    expected: dict[str, int] = {}
    for fold in FOLDS:
        data = FoldData(inventory, manifest, fold)
        # Includes all source hashes/identities plus exact realized-input equality.
        reports = check_evaluations(experiment, data, inventory, conditions, REPEATS)
        for sample in data.test_sample_ids:
            if sample in expected:
                raise ValueError("A sample occurs in multiple OOF test folds")
            expected[sample] = fold
        for report in reports:
            condition, repeat = report["condition"], report["repeat"]
            relative = f"results/evaluation/{condition}/fold_{fold}/repeat_{repeat}/completion.json"
            ledger.append({"condition": condition, "fold": fold, "repeat": repeat,
                           "completion": relative, "sha256": _digest(experiment / relative),
                           "shared_inputs_sha256": report["shared_inputs_sha256"]})
    if set(expected) != {sample.sample_id for sample in inventory.samples}:
        raise ValueError("OOF test folds do not cover the inventory exactly once")
    return dict(sorted(expected.items())), ledger


def _comparison_plan(conditions: tuple[str, ...]) -> dict[str, object]:
    selected = set(conditions)
    return {
        "paired": [{"condition": left, "reference": right,
                    "status": "included" if {left, right} <= selected else "conditions_not_selected"}
                   for left, right in PLANNED_PAIRS],
        "interaction": "included" if set(FACTORIAL_CONDITIONS) <= selected else "conditions_not_selected",
        "paired_metrics": list(REPEATED_METRICS), "interaction_metrics": list(PRIMARY_METRICS),
    }


def _files(conditions: tuple[str, ...]) -> set[str]:
    files = {"run.json", "scores.json"}
    for condition in conditions:
        for k in CLUSTER_COUNTS:
            files.update((f"summaries/{condition}/k{k}.json", f"ari/{condition}/k{k}.json"))
    for left, right in PLANNED_PAIRS:
        if left in conditions and right in conditions:
            files.update(f"comparisons/{left}_minus_{right}/k{k}.json" for k in CLUSTER_COUNTS)
    if set(FACTORIAL_CONDITIONS) <= set(conditions):
        files.update(f"interaction/k{k}.json" for k in CLUSTER_COUNTS)
    return files


def run_oof(
    experiment: Path, inventory: InputInventory, manifest: CVManifest,
    conditions: tuple[str, ...], *, snapshot: str | None = None,
) -> Path:
    """Aggregate every saved sample/K from all five folds and three repeats.

    Require complete evaluations for every selected condition. Missing/failed/
    interrupted sources stop publication; they are never silently omitted or
    converted to undefined scores. No spectra, fit, GPU or inference is needed.
    Labels for only one sample/condition (three repeats, all K) are kept at once.
    """
    conditions = _conditions(conditions)
    snapshot = snapshot if snapshot is not None else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = _output(experiment, snapshot)
    if output.exists():
        raise FileExistsError(f"OOF snapshot already exists: {output}")
    expected, ledger = _sources(experiment, inventory, manifest, conditions)
    records = []
    for item in ledger:
        source = experiment / item["completion"]
        for value in _read_json(source.with_name("scores.json"))["records"]:
            row = ScoreRecord(**value)
            if (row.condition_id != item["condition"] or row.repeat != item["repeat"]
                    or row.fold != item["fold"] or expected.get(row.sample_id) != row.fold
                    or row.metric not in REPEATED_METRICS or row.k not in CLUSTER_COUNTS
                    or row.status not in ("defined", "undefined")):
                raise ValueError("OOF score differs from its complete source run or saved split")
            records.append(row)
    started = time.perf_counter()
    output.mkdir(parents=True)
    completed = False
    try:
        _write_json(output / "run.json", {
            "schema_version": 1, "mode": "complete_selected_conditions_oof", "snapshot": snapshot,
            "conditions": list(conditions), "folds": list(FOLDS), "repeats": list(REPEATS),
            "cluster_counts": list(CLUSTER_COUNTS), "expected_test_folds": expected,
            "config": experiment_config(), "code_sha256": _codes(),
            "manifest_artifact_sha256": _read_json(experiment / "manifests/complete.json")["artifact_sha256"],
            "sources": ledger, "comparison_plan": _comparison_plan(conditions),
        })
        _write_json(output / "scores.json", {"records": [asdict(row) for row in records]})
        grouped: dict[tuple[str, int, str], list[ScoreRecord]] = {}
        for row in records:
            grouped.setdefault((row.condition_id, row.k, row.metric), []).append(row)
        samples = {sample.sample_id: sample for sample in inventory.samples}
        for condition in conditions:
            (output / "summaries" / condition).mkdir(parents=True)
            (output / "ari" / condition).mkdir(parents=True)
            aris = {k: [] for k in CLUSTER_COUNTS}
            for sample_id, fold in expected.items():
                sample = samples[sample_id]
                with h5py.File(sample.path, "r") as handle:
                    coordinates = handle["pixel_row_col"][:]
                maps = {}
                for repeat in REPEATS:
                    path = experiment / f"results/clustering/{condition}/fold_{fold}/repeat_{repeat}/maps/{sample_id}.npz"
                    with np.load(path, allow_pickle=False) as saved:
                        maps[repeat] = {k: saved[f"labels_k{k}"][coordinates[:, 0], coordinates[:, 1]]
                                        for k in CLUSTER_COUNTS}
                for k in CLUSTER_COUNTS:
                    predictions = {repeat: PixelLabels(sample_id, np.arange(sample.saved_pixel_count),
                                                       coordinates, maps[repeat][k]) for repeat in REPEATS}
                    aris[k].append(repeat_ari(predictions, expected_pixel_count=sample.saved_pixel_count,
                                               condition_id=condition, fold=fold, k=k))
            for k in CLUSTER_COUNTS:
                metrics = {metric: asdict(aggregate_scores(
                    grouped.get((condition, k, metric), []), expected_test_folds=expected,
                    condition_id=condition, metric=metric, k=k,
                )) for metric in REPEATED_METRICS}
                _write_json(output / f"summaries/{condition}/k{k}.json",
                            {"condition_id": condition, "k": k, "metrics": metrics})
                _write_json(output / f"ari/{condition}/k{k}.json", asdict(aggregate_ari(
                    aris[k], expected_test_folds=expected, condition_id=condition, k=k)))
            print(f"{condition}: all OOF samples, three-repeat ARI and all K saved", flush=True)
        for left, right in PLANNED_PAIRS:
            if left not in conditions or right not in conditions:
                continue
            directory = output / f"comparisons/{left}_minus_{right}"
            directory.mkdir(parents=True)
            for k in CLUSTER_COUNTS:
                metrics = {metric: asdict(paired_difference(
                    grouped[left, k, metric] + grouped[right, k, metric], expected_test_folds=expected,
                    condition_id=left, reference_condition=right, metric=metric, k=k,
                )) for metric in REPEATED_METRICS}
                _write_json(directory / f"k{k}.json", {"condition": left, "reference": right,
                                                       "k": k, "metrics": metrics})
        if set(FACTORIAL_CONDITIONS) <= set(conditions):
            (output / "interaction").mkdir()
            for k in CLUSTER_COUNTS:
                metrics = {metric: asdict(interaction_contrast(
                    [row for condition in FACTORIAL_CONDITIONS for row in grouped[condition, k, metric]],
                    expected_test_folds=expected, metric=metric, k=k,
                )) for metric in PRIMARY_METRICS}
                _write_json(output / f"interaction/k{k}.json", {"k": k, "metrics": metrics})
        hashes = {name: _digest(output / name) for name in sorted(_files(conditions))}
        _write_json(output / "completion.json", {
            "status": "oof_aggregation_completed", "checks_passed": True,
            "scope": "all samples/folds/repeats/K of explicitly selected conditions",
            "artifact_sha256": hashes, "conditions": list(conditions), "sample_count": len(expected),
            "source_run_count": len(ledger), "score_record_count": len(records),
            "wall_seconds": time.perf_counter() - started,
            "timing_scope": "aggregation and saving; excludes source validation and score loading",
        })
        completed = True
    finally:
        if not completed:
            error = sys.exc_info()[1]
            try:
                _write_json(output / "failure.json", {"status": "failed_or_interrupted",
                            "error_type": type(error).__name__, "reason": str(error)})
            except OSError as logging_error:
                print(f"Could not save OOF failure record: {logging_error}", file=sys.stderr)
    return output


def check_oof(
    experiment: Path, inventory: InputInventory, manifest: CVManifest, snapshot: str,
) -> dict[str, object]:
    """Verify snapshot and current source binding without recalculating ARI or summaries."""
    output = _output(experiment, snapshot)
    run, done = _read_json(output / "run.json"), _read_json(output / "completion.json")
    conditions = _conditions(tuple(run["conditions"]))
    expected, ledger = _sources(experiment, inventory, manifest, conditions)
    required = {
        "schema_version": 1, "mode": "complete_selected_conditions_oof", "snapshot": snapshot,
        "conditions": list(conditions), "folds": list(FOLDS), "repeats": list(REPEATS),
        "cluster_counts": list(CLUSTER_COUNTS), "expected_test_folds": expected,
        "config": experiment_config(), "code_sha256": _codes(), "sources": ledger,
        "manifest_artifact_sha256": _read_json(experiment / "manifests/complete.json")["artifact_sha256"],
        "comparison_plan": _comparison_plan(conditions),
    }
    count = len(expected) * len(conditions) * len(REPEATS) * len(CLUSTER_COUNTS) * len(REPEATED_METRICS)
    if (any(run.get(key) != value for key, value in required.items())
            or done.get("status") != "oof_aggregation_completed" or done.get("checks_passed") is not True
            or done.get("conditions") != list(conditions) or done.get("sample_count") != len(expected)
            or done.get("source_run_count") != len(ledger) or done.get("score_record_count") != count
            or (output / "failure.json").exists() or set(done["artifact_sha256"]) != _files(conditions)):
        raise ValueError("OOF completion/coverage/source mismatch")
    for name, digest in done["artifact_sha256"].items():
        if digest != _digest(output / name):
            raise ValueError(f"OOF artifact hash mismatch: {name}")
    return {"status": "validated_existing_oof", "snapshot": snapshot, "conditions": list(conditions),
            "sample_count": len(expected), "source_run_count": len(ledger), "score_record_count": count}
