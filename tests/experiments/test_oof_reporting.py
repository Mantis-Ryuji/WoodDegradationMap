"""Synthetic saved artifacts exercise reporting without spectra, maps or fitting."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from wood_degradation_map.experiments import oof_reporting as report
from wood_degradation_map.experiments.aggregation import ScoreRecord, aggregate_scores, paired_difference
from wood_degradation_map.experiments.config import ADOPTED_SAMPLE_IDS, CLUSTER_COUNTS, FOLDS, REPEATS
from wood_degradation_map.experiments.manifests import _digest, _read_json, _write_json


def _summary(condition: str, metric: str, k: int, expected: dict[str, int]) -> dict:
    values = {"B0": 0.0, "B1": 0.1, "A0": 0.2, "M00": 0.2,
              "M10": 0.3, "M01": 0.4, "M11": 0.8}
    scale = {"lfr_noise": 0.25, "lfr_shift": 0.5}.get(metric, 1.0)
    records = [ScoreRecord(sample, fold, condition, k, metric, repeat, "defined",
                           (values[condition] + repeat / 100) * scale)
               for sample, fold in expected.items() for repeat in REPEATS]
    return asdict(aggregate_scores(records, expected_test_folds=expected, condition_id=condition,
                                   metric=metric, k=k))


def test_corrected_interaction_uses_all_four_conditions_and_three_repeats() -> None:
    expected = {"a": 1, "b": 2}
    parts = {condition: _summary(condition, "adjusted_lla_3", 8, expected)
             for condition in report.FACTORIAL_CONDITIONS}
    result = report.corrected_interaction(parts, expected)
    assert result["metric"] == "adjusted_lla_3"
    assert result["mean"] == pytest.approx(0.3)
    assert result["common_samples"] == 2
    # One unavailable repeat excludes the sample from the complete contrast,
    # while the other repeat rows and source reason must remain available.
    source = parts["M01"]
    row = next(row for row in source["rows"] if row["sample_id"] == "b" and row["repeat"] == 2)
    unavailable = {"sample_id": "b", "fold": 2, "condition_id": "M01", "repeat": 2,
                   "status": "undefined", "reason": "single_cluster"}
    row.update(value=None, unavailable_sources=[unavailable])
    source["availability"][1].update(defined_samples=1, unavailable=[unavailable])
    result = report.corrected_interaction(parts, expected)
    assert result["common_sample_ids"] == ("a",)
    assert result["sample_sd"] is None
    assert result["rows"][3]["value"] == pytest.approx(0.3)
    assert result["rows"][4]["unavailable_sources"][0]["reason"] == "single_cluster"


@pytest.fixture
def saved_report(tmp_path: Path) -> tuple[Path, str]:
    experiment = tmp_path / "experiment"
    snapshot = "fixture"
    prefix = experiment / "results/oof" / snapshot
    prefix.mkdir(parents=True)
    rows = [{"sample_id": sample, "test_fold": index % 5 + 1,
             "saved_pixel_count": 6 if index % 2 == 0 else 12, "height": 4, "width": 4}
            for index, sample in enumerate(ADOPTED_SAMPLE_IDS)]
    expected = {row["sample_id"]: row["test_fold"] for row in rows}
    (experiment / "manifests").mkdir()
    (experiment / "config").mkdir()
    pd.DataFrame(rows).to_parquet(experiment / "manifests/folds.parquet", index=False)
    _write_json(experiment / "manifests/inputs.json", {"preprocessing_id": "production_v1", "samples": rows})
    _write_json(experiment / "config/experiment.json", {"split": {"adopted_sample_ids": list(expected)}})
    hashes = {name: _digest(experiment / name) for name in
              ("manifests/folds.parquet", "manifests/inputs.json", "config/experiment.json")}
    _write_json(experiment / "manifests/complete.json", {"status": "complete", "artifact_sha256": hashes})
    ledger = []
    for condition in report.MAIN_CONDITIONS:
        for fold in FOLDS:
            test_ids = sorted(sample for sample, value in expected.items() if value == fold)
            for repeat in REPEATS:
                suffix = f"{condition}/fold_{fold}/repeat_{repeat}"
                cluster = experiment / f"results/clustering/{suffix}"
                evaluation = experiment / f"results/evaluation/{suffix}"
                cluster.mkdir(parents=True)
                evaluation.mkdir(parents=True)
                identity = {"schema_version": 2, "condition": condition, "fold": fold, "repeat": repeat,
                            "test_sample_ids": test_ids, "manifest_artifact_sha256": hashes,
                            "source": {"kind": condition}}
                _write_json(cluster / "run.json", {
                    **identity, "mode": "clean_test_maps", "cluster_counts": list(CLUSTER_COUNTS),
                    "train_sample_ids": sorted(set(expected) - set(test_ids)),
                })
                samples = [{"sample_id": row["sample_id"], "pixels": row["saved_pixel_count"],
                            "occupancy": {str(k): [row["saved_pixel_count"] - repeat, repeat]
                                          + [0] * (k - 2) for k in CLUSTER_COUNTS}}
                           for row in rows if row["sample_id"] in test_ids]
                _write_json(cluster / "completion.json", {
                    "status": "clean_test_maps_completed", "checks_passed": True,
                    "run_sha256": _digest(cluster / "run.json"), "samples": samples,
                })
                _write_json(evaluation / "run.json", {
                    **identity, "mode": "full_test_evaluation",
                    "clustering_completion_sha256": _digest(cluster / "completion.json"),
                })
                _write_json(evaluation / "completion.json", {
                    "status": "full_test_evaluation_completed", "checks_passed": True,
                    "run_sha256": _digest(evaluation / "run.json"), "shared_inputs_sha256": "fixture",
                })
                ledger.append({"condition": condition, "fold": fold, "repeat": repeat,
                               "completion": f"results/evaluation/{suffix}/completion.json",
                               "sha256": _digest(evaluation / "completion.json"),
                               "shared_inputs_sha256": "fixture"})
    for condition in report.MAIN_CONDITIONS:
        (prefix / "summaries" / condition).mkdir(parents=True)
        (prefix / "ari" / condition).mkdir(parents=True)
        for k in CLUSTER_COUNTS:
            _write_json(prefix / f"summaries/{condition}/k{k}.json", {
                "condition_id": condition, "k": k,
                "metrics": {metric: _summary(condition, metric, k, expected) for metric in report.REPEATED},
            })
            # No repeat_sd is provided for ARI: reporting must keep it unavailable.
            _write_json(prefix / f"ari/{condition}/k{k}.json", {
                "condition_id": condition, "k": k, "total_samples": len(expected),
                "defined_samples": len(expected), "defined_sample_ids": sorted(expected),
                "undefined_sample_ids": [], "mean": 0.5, "sample_sd": 0.0,
                "mean_undefined_reason": None, "sample_sd_undefined_reason": None,
                "samples": [{"sample_id": sample, "fold": fold, "mean": 0.5, "undefined_reason": None,
                             "pairs": [{"repeats": pair, "value": 0.5, "undefined_reason": None,
                                        "used_clusters": [2, 2], "degeneracy_flags": ["fixture_flag"]}
                                       for pair in ([1, 2], [1, 3], [2, 3])]} for sample, fold in expected.items()],
            })
    for left, right in report.PLANNED_PAIRS:
        (prefix / "comparisons" / f"{left}_minus_{right}").mkdir(parents=True)
        for k in CLUSTER_COUNTS:
            metrics = {}
            for metric in report.REPEATED:
                records = [ScoreRecord(row["sample_id"], row["fold"], condition, k, metric,
                                       row["repeat"], "defined", row["value"])
                           for condition in (left, right)
                           for row in _summary(condition, metric, k, expected)["rows"]]
                metrics[metric] = asdict(paired_difference(
                    records, expected_test_folds=expected, condition_id=left,
                    reference_condition=right, metric=metric, k=k))
            _write_json(prefix / f"comparisons/{left}_minus_{right}/k{k}.json", {
                "condition": left, "reference": right, "k": k, "metrics": metrics,
            })
    _write_json(prefix / "run.json", {
        "schema_version": 1, "mode": "complete_selected_conditions_oof", "snapshot": snapshot,
        "conditions": list(report.MAIN_CONDITIONS), "folds": list(FOLDS), "repeats": list(REPEATS),
        "cluster_counts": list(CLUSTER_COUNTS), "expected_test_folds": expected,
        "manifest_artifact_sha256": hashes, "sources": ledger,
    })
    _write_json(prefix / "completion.json", {
        "status": "oof_aggregation_completed", "checks_passed": True,
        "conditions": list(report.MAIN_CONDITIONS), "sample_count": 49, "source_run_count": 105,
        "score_record_count": 72030,
        "artifact_sha256": {path.relative_to(prefix).as_posix(): _digest(path)
                            for path in prefix.rglob("*.json")},
    })
    return experiment, snapshot


def test_saved_data_keeps_corrected_lla_ari_and_sample_weighting(saved_report: tuple[Path, str]) -> None:
    data = report.load_report_data(*saved_report)
    assert not data.summaries.source_metric.str.match(r"lla_\d").any()
    assert set(data.summaries[data.summaries.metric == "LLA"].window) == {3, 5, 9}
    assert data.summaries[data.summaries.source_metric == "ari"].repeat_sd.isna().all()
    assert all(json.loads(value) == ["fixture_flag"] for value in data.ari_pairs.degeneracy_flags)
    occupancy = data.summaries[(data.summaries.source_metric == "max_occupancy")
                               & (data.summaries.expression == "B0") & (data.summaries.k == 8)].iloc[0]
    assert occupancy["mean"] == pytest.approx((25 * (2 / 3) + 24 * (5 / 6)) / 49)
    assert len(data.occupancy) == 49 * 7 * 3 * 7
    assert len(data.occupancy_folds) == 7 * 5 * 3 * sum(CLUSTER_COUNTS)
    paired = data.summaries[(data.summaries.kind == "paired")
                            & (data.summaries.source_metric == "lfr_both")
                            & (data.summaries.expression == "M11 - M00")]
    assert paired["mean"].to_numpy() == pytest.approx(np.full(7, 0.6))


def test_changed_occupancy_is_rejected_before_output(saved_report: tuple[Path, str], tmp_path: Path) -> None:
    experiment, snapshot = saved_report
    path = experiment / "results/clustering/M11/fold_5/repeat_3/completion.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="hash mismatch"):
        report.render_report(experiment, snapshot, output, dpi=100)
    assert not output.exists()


def test_input_paths_are_protected(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    with pytest.raises(ValueError, match="Output"):
        report.render_report(experiment, "snapshot", experiment / "results/oof/other")


def test_paired_ari_keeps_common_samples_and_no_repeat_sd() -> None:
    left = {"condition_id": "M11", "k": 8, "samples": [
        {"sample_id": "a", "fold": 1, "mean": 0.25, "undefined_reason": None},
        {"sample_id": "b", "fold": 2, "mean": 0.75, "undefined_reason": None},
    ]}
    right = {"condition_id": "B0", "k": 8, "samples": [
        {"sample_id": "a", "fold": 1, "mean": 0.5, "undefined_reason": None},
        {"sample_id": "b", "fold": 2, "mean": None, "undefined_reason": "fixture_undefined"},
    ]}
    summary, samples = report.paired_ari(left, right, {"a": 1, "b": 2})
    assert summary["mean"] == -0.25
    assert summary["sample_sd"] is None and summary["repeat_sd"] is None
    assert summary["common_samples"] == 1
    assert json.loads(summary["excluded_sample_ids"]) == ["b"]
    assert samples[1]["mean"] is None and "fixture_undefined" in samples[1]["undefined_reason"]
    assert not any(key.startswith("repeat_") for row in samples for key in row)


def test_csv_and_figure_contract(
    saved_report: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, snapshot = saved_report
    output = tmp_path / "report"
    output.mkdir()
    note = output / "user_note.txt"
    note.write_text("keep", encoding="utf-8")
    for name in report.OBSOLETE_FIGURES:
        (output / f"{name}.png").write_bytes(b"obsolete")
    (output / "completion.json").write_text("stale", encoding="utf-8")
    (output / "report.json").write_text("stale", encoding="utf-8")
    original_save = report._save
    layouts = {}

    def inspect_and_save(figure: object, directory: Path, stem: str, dpi: int) -> None:
        assert not (output / "completion.json").exists()
        assert len(figure.axes) == 6
        grid = figure.axes[0].get_subplotspec().get_gridspec()
        assert (grid.nrows, grid.ncols) == (2, 3)
        assert [axis.get_ylabel() for axis in figure.axes] == [
            report.LABELS[metric] + (" difference" if stem == "03_paired_k_sweep" else "")
            for metric in report.SUMMARY_METRICS
        ]
        assert figure.axes[3].get_ylabel() == "LFR(TGN+FS)" + (
            " difference" if stem == "03_paired_k_sweep" else ""
        )
        if stem == "03_paired_k_sweep":
            assert len(figure.axes[4].get_lines()) == 4  # Three ARI differences and zero line.
        layouts[stem] = len(figure.axes)
        original_save(figure, directory, stem, dpi)

    monkeypatch.setattr(report, "_save", inspect_and_save)
    report.render_report(experiment, snapshot, output, dpi=100)
    assert set(layouts) == set(report.FIGURE_NAMES)
    assert {path.stem for path in output.glob("*.png")} == set(report.FIGURE_NAMES)
    assert sorted(path.name[:2] for path in output.glob("*.png")) == ["01", "02", "03"]
    assert len(list(output.glob("*.csv"))) == 11
    assert not pd.read_csv(output / "interaction_all_k.csv").empty
    assert not pd.read_csv(output / "interaction_k8.csv").empty
    assert note.read_text(encoding="utf-8") == "keep"
    table = pd.read_csv(output / "metrics_k8.csv")
    assert table.priority.tolist() == sorted(table.priority)
    assert len(table) == 7 * 11
    assert table.loc[table.metric == "LLA", "label"].str.startswith("LLA (").all()
    assert _read_json(output / "completion.json")["status"] == "oof_figures_completed"
    assert "user_note.txt" not in _read_json(output / "completion.json")["artifact_sha256"]
    assert set(pd.read_csv(output / "paired_k8.csv").source_metric) == set(report.TABLE_METRICS)
    variants = {"lfr_both": ("LFR(TGN+FS)", 1.0), "lfr_noise": ("LFR(TGN)", 0.25),
                "lfr_shift": ("LFR(FS)", 0.5)}
    for name in ("metrics_all_k", "metrics_k8", "paired_all_k", "paired_k8",
                 "interaction_all_k", "interaction_k8", "sample_values", "availability"):
        csv = pd.read_csv(output / f"{name}.csv")
        for metric, (label, scale) in variants.items():
            selected = csv[csv.source_metric == metric]
            assert not selected.empty
            assert set(selected.metric) == set(selected.label) == {label}
            assert set(selected.priority) == {2}
            if name.startswith(("metrics_", "paired_", "interaction_")):
                expression, value = (
                    ("M11", 0.82) if name.startswith("metrics_") else
                    ("M11 - M00", 0.6) if name.startswith("paired_") else
                    ("(M11 - M10) - (M01 - M00)", 0.3)
                )
                assert selected.loc[selected.expression == expression, "mean"].to_numpy() == (
                    pytest.approx(value * scale)
                )
    assert not pd.read_csv(output / "occupancy_samples.csv").empty
    manifest = _read_json(output / "report.json")
    assert manifest["metric_order"] == ["LLA", "LFR(TGN+FS)", "ARI", "Cosine-Silhouette", "Cluster Occupancy"]
    assert all(label in manifest["table_metric_order"] for label, _ in variants.values())
    assert not any(item["path"].endswith((".npz", ".h5", ".pt")) for item in manifest["sources"])
