"""Synthetic OOF metadata exercises CSV export, missing values and source validation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import matplotlib
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from wood_degradation_map.experiments import mask_rate_reporting as report
from wood_degradation_map.experiments.aggregation import (
    REPEATED_METRICS, ScoreRecord, aggregate_ari, aggregate_scores,
)
from wood_degradation_map.experiments.config import (
    ADOPTED_SAMPLE_IDS, CLUSTER_COUNTS, FOLDS, REPEATS,
)
from wood_degradation_map.experiments.diagnostic_metrics import RepeatARI, _ari_pair
from wood_degradation_map.experiments.manifests import _digest, _read_json, _write_json


@pytest.fixture
def saved_mask_rate(tmp_path: Path) -> tuple[Path, str]:
    experiment, snapshot = tmp_path / "experiment", "mask_fixture"
    prefix = experiment / "results/oof" / snapshot
    prefix.mkdir(parents=True)
    rows = [{"sample_id": sample, "test_fold": index % 5 + 1,
             "saved_pixel_count": 6 if index % 2 == 0 else 12, "height": 4, "width": 4}
            for index, sample in enumerate(ADOPTED_SAMPLE_IDS)]
    expected = {row["sample_id"]: row["test_fold"] for row in rows}
    (experiment / "manifests").mkdir()
    (experiment / "config").mkdir()
    pd.DataFrame(rows).to_parquet(experiment / "manifests/folds.parquet", index=False)
    _write_json(experiment / "manifests/inputs.json", {
        "preprocessing_id": "production_v1", "samples": rows,
    })
    _write_json(experiment / "config/experiment.json", {
        "split": {"adopted_sample_ids": list(expected)},
    })
    hashes = {name: _digest(experiment / name) for name in
              ("manifests/folds.parquet", "manifests/inputs.json", "config/experiment.json")}
    _write_json(experiment / "manifests/complete.json", {
        "status": "complete", "artifact_sha256": hashes,
    })
    for index, condition in enumerate(report.MASK_CONDITIONS):
        (prefix / "summaries" / condition).mkdir(parents=True)
        (prefix / "ari" / condition).mkdir(parents=True)
        for k in CLUSTER_COUNTS:
            metrics = {}
            for metric in REPEATED_METRICS:
                records = []
                for sample, fold in expected.items():
                    undefined = condition == "M11" and k == 2 and sample == ADOPTED_SAMPLE_IDS[0]
                    undefined = undefined and "lla" in metric
                    scale = {"adjusted_lla_3": 1.5, "adjusted_lla_9": 0.5}.get(metric, 1.0)
                    for repeat in REPEATS:
                        records.append(ScoreRecord(
                            sample, fold, condition, k, metric, repeat,
                            "undefined" if undefined else "defined",
                            None if undefined else ((index + 1) / 10 + repeat / 1000) * scale,
                            "single_cluster" if undefined else None,
                        ))
                metrics[metric] = asdict(aggregate_scores(
                    records, expected_test_folds=expected,
                    condition_id=condition, metric=metric, k=k,
                ))
            _write_json(prefix / f"summaries/{condition}/k{k}.json", {
                "condition_id": condition, "k": k, "metrics": metrics,
            })
            aris = []
            for row in rows:
                sample, pixels = row["sample_id"], row["saved_pixel_count"]
                single = condition == "M11" and k == 2 and sample == ADOPTED_SAMPLE_IDS[0]
                labels = {repeat: np.ones(pixels, dtype=np.int16) if single else
                          np.array([1] * (pixels - repeat) + [2] * repeat, dtype=np.int16)
                          for repeat in REPEATS}
                pairs = tuple(_ari_pair(labels[a], labels[b], k, (a, b))
                              for a, b in combinations(REPEATS, 2))
                mean = float(np.array([pair.value for pair in pairs], dtype=np.float32).mean())
                aris.append(RepeatARI(
                    sample, condition, row["test_fold"], k, pixels, pairs, mean, None,
                ))
            _write_json(prefix / f"ari/{condition}/k{k}.json", asdict(aggregate_ari(
                aris, expected_test_folds=expected, condition_id=condition, k=k,
            )))
    _write_json(prefix / "scores.json", {"records": []})  # Not consumed by the CSV exporter.
    _write_json(prefix / "run.json", {
        "schema_version": 1, "mode": "complete_selected_conditions_oof", "snapshot": snapshot,
        "conditions": sorted(report.MASK_CONDITIONS),
        "folds": list(FOLDS), "repeats": list(REPEATS),
        "cluster_counts": list(CLUSTER_COUNTS), "expected_test_folds": expected,
        "manifest_artifact_sha256": hashes,
        "sources": [{"condition": c, "fold": f, "repeat": r}
                    for c in report.MASK_CONDITIONS for f in FOLDS for r in REPEATS],
    })
    _write_json(prefix / "completion.json", {
        "status": "oof_aggregation_completed", "checks_passed": True,
        "conditions": sorted(report.MASK_CONDITIONS), "sample_count": 49, "source_run_count": 45,
        "score_record_count": 49 * 3 * 3 * 7 * len(REPEATED_METRICS),
        "artifact_sha256": {path.relative_to(prefix).as_posix(): _digest(path)
                            for path in prefix.rglob("*.json")},
    })
    return experiment, snapshot


def _rewrite_bound(prefix: Path, name: str, value: dict) -> None:
    path = prefix / name
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    done = _read_json(prefix / "completion.json")
    done["artifact_sha256"][name] = _digest(path)
    (prefix / "completion.json").write_text(json.dumps(done), encoding="utf-8")


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames), list(reader)


def test_csv_contract_and_sample_weighting(
    saved_mask_rate: tuple[Path, str], tmp_path: Path,
) -> None:
    experiment, snapshot = saved_mask_rate
    output = report.export_mask_rate_report(experiment, snapshot, tmp_path / "report")
    columns, rows = _csv_rows(output / "metrics_all_k.csv")
    assert columns == list(report.COLUMNS) and len(rows) == 231
    columns, k8 = _csv_rows(output / "metrics_k8.csv")
    assert columns == list(report.K8_COLUMNS) and len(k8) == 11
    assert {row["mask_percent"] for row in rows} == {"25", "50", "75"}
    assert not any(row["source_metric"].startswith("lla_") for row in rows)
    windows = {row["window"] for row in rows if row["source_metric"] in report.LLA_METRICS}
    assert windows == {"3", "5", "9"}
    assert all(row["repeat_sd"] == "" for row in rows if row["source_metric"] == "ari")
    expected_fraction = (25 * (2 / 3) + 24 * (5 / 6)) / 49
    occupancy = next(row for row in k8 if row["source_metric"] == "max_occupancy")
    assert float(occupancy["mask_25_mean"]) == pytest.approx(expected_fraction)
    pooled_fraction = (25 * 4 + 24 * 10) / (25 * 6 + 24 * 12)
    assert float(occupancy["mask_25_mean"]) != pytest.approx(pooled_fraction)
    lla = next(row for row in k8 if row["source_metric"] == "adjusted_lla_5")
    for condition, percent in report.MASK_PERCENT.items():
        saved = _read_json(experiment / f"results/oof/{snapshot}/summaries/{condition}/k8.json")
        assert float(lla[f"mask_{percent}_mean"]) == saved["metrics"]["adjusted_lla_5"]["mean"]
    missing = next(row for row in rows if row["condition_id"] == "M11" and row["k"] == "2"
                   and row["source_metric"] == "adjusted_lla_5")
    assert missing["common_samples"] == "48"
    assert json.loads(missing["excluded_sample_ids"]) == [ADOPTED_SAMPLE_IDS[0]]
    assert json.loads(missing["availability"])[0]["unavailable"][0]["reason"] == "single_cluster"
    single = next(row for row in rows if row["condition_id"] == "M11" and row["k"] == "2"
                  and row["source_metric"] == "single_cluster")
    assert float(single["mean"]) == pytest.approx(1 / 49)
    k8_single = next(row for row in k8 if row["source_metric"] == "single_cluster")
    assert k8_single["mask_25_mean"] == "0.0"
    assert {path.name for path in output.iterdir()} == {*report.ARTIFACTS, "completion.json"}
    done = _read_json(output / "completion.json")
    assert done["status"] == "mask_rate_csv_completed" and done["checks_passed"] is True
    assert all(_digest(output / name) == digest for name, digest in done["artifact_sha256"].items())


def test_changed_snapshot_is_rejected_before_output(
    saved_mask_rate: tuple[Path, str], tmp_path: Path,
) -> None:
    experiment, snapshot = saved_mask_rate
    path = experiment / f"results/oof/{snapshot}/summaries/M11-25/k2.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="hash mismatch"):
        report.export_mask_rate_report(experiment, snapshot, output)
    assert not output.exists()


@pytest.mark.parametrize("damage", ("repeat", "contingency", "source_run", "manifest"))
def test_incomplete_or_inconsistent_metadata_is_rejected(
    saved_mask_rate: tuple[Path, str], tmp_path: Path, damage: str,
) -> None:
    experiment, snapshot = saved_mask_rate
    prefix = experiment / "results/oof" / snapshot
    if damage == "repeat":
        name = "summaries/M11-25/k2.json"
        value = _read_json(prefix / name)
        value["metrics"]["adjusted_lla_3"]["rows"].pop()
    elif damage == "contingency":
        name = "ari/M11-25/k2.json"
        value = _read_json(prefix / name)
        table = value["samples"][0]["pairs"][1]["contingency"]
        table[0][0] -= 1
        table[1][0] += 1
    else:
        name = "run.json"
        value = _read_json(prefix / name)
        if damage == "source_run":
            value["sources"][-1] = value["sources"][0]
        else:
            value["manifest_artifact_sha256"] = {}
    _rewrite_bound(prefix, name, value)
    output = tmp_path / "report"
    message = "coverage mismatch|repeat margins|manifest/source coverage"
    with pytest.raises(ValueError, match=message):
        report.export_mask_rate_report(experiment, snapshot, output)
    assert not output.exists()


def test_explicit_overwrite_preserves_unrelated_files(
    saved_mask_rate: tuple[Path, str], tmp_path: Path,
) -> None:
    experiment, snapshot = saved_mask_rate
    output = tmp_path / "report"
    report.export_mask_rate_report(experiment, snapshot, output)
    note = output / "user_note.txt"
    note.write_text("keep", encoding="utf-8")
    original = (output / "completion.json").read_bytes()
    with pytest.raises(FileExistsError, match="--overwrite"):
        report.export_mask_rate_report(experiment, snapshot, output)
    assert (output / "completion.json").read_bytes() == original
    report.export_mask_rate_report(experiment, snapshot, output, overwrite=True)
    assert note.read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError, match="Output"):
        report.export_mask_rate_report(experiment, snapshot, experiment / "results/oof/other")


def test_interrupted_publication_removes_stale_success_records(
    saved_mask_rate: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, snapshot = saved_mask_rate
    output = tmp_path / "report"
    report.export_mask_rate_report(experiment, snapshot, output)
    original_replace = Path.replace

    def fail_second_csv(path: Path, target: Path) -> Path:
        if path.name == "metrics_k8.csv":
            raise OSError("publication interrupted")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_csv)
    with pytest.raises(OSError, match="publication interrupted"):
        report.export_mask_rate_report(experiment, snapshot, output, overwrite=True)
    assert not (output / "completion.json").exists()
    assert not (output / "report.json").exists()


def test_four_panel_k_sweep_uses_three_mask_rates_and_saved_values(
    saved_mask_rate: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, snapshot = saved_mask_rate
    output = tmp_path / "report"
    original_save = report._save
    inspected = []

    def inspect_and_save(figure: Figure, directory: Path, stem: str, dpi: int) -> None:
        assert len(figure.axes) == 4
        grid = figure.axes[0].get_subplotspec().get_gridspec()
        assert (grid.nrows, grid.ncols) == (2, 2)
        assert figure.get_size_inches() == pytest.approx([3.5, 4.1])
        for axis, metric in zip(figure.axes, report.MAIN_FIGURE_METRICS, strict=True):
            assert axis.get_ylabel() == report.LABELS[metric]
            means = [line for line in axis.get_lines() if not line.get_label().startswith("_")]
            assert [line.get_label() for line in means] == ["M11-25", "M11(-50)", "M11-75"]
            assert len(axis.get_lines()) == 12  # Three means and nine learning-repeat curves.
            for line, condition in zip(means, report.MASK_CONDITIONS, strict=True):
                values = [_read_json(experiment / (
                    f"results/oof/{snapshot}/summaries/{condition}/k{k}.json"
                ))["metrics"][metric]["mean"] for k in CLUSTER_COUNTS]
                assert line.get_xdata() == pytest.approx(CLUSTER_COUNTS)
                assert line.get_ydata() == pytest.approx(values)
            assert np.diff(axis.get_yticks()) == pytest.approx(0.1)
            expected_limits = ((-0.029712518723681573, 0.7166190753225237)
                               if metric == "lfr_both" else (0.2, 0.9))
            assert axis.get_ylim() == expected_limits
        assert all(axis.get_ylim() == figure.axes[0].get_ylim() for axis in figure.axes[:3])
        inspected.append(stem)
        original_save(figure, directory, stem, dpi)

    monkeypatch.setattr(report, "_save", inspect_and_save)
    report.export_mask_rate_report(experiment, snapshot, output, with_figure=True, dpi=100)
    assert inspected == [Path(report.FIGURE_NAME).stem]
    assert {path.name for path in output.glob("*.png")} == {report.FIGURE_NAME}
    assert (output / report.FIGURE_NAME).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    done = _read_json(output / "completion.json")
    assert done["status"] == "mask_rate_report_completed"
    assert done["artifact_sha256"][report.FIGURE_NAME] == _digest(output / report.FIGURE_NAME)
    metadata = _read_json(output / "report.json")["figure"]
    assert metadata["source_metrics"] == list(report.MAIN_FIGURE_METRICS)
    assert metadata["condition_labels"]["M11"] == "M11(-50)"
    assert metadata["y_limits"]["adjusted_lla_3"] == [0.2, 0.9]
    assert metadata["y_limits"]["lfr_both"] == [-0.029712518723681573, 0.7166190753225237]
    assert metadata["y_axis_reference"] == "main_oof_v1/01_main_metrics_k_sweep.png"


def test_undefined_k_sweep_values_are_gaps(
    saved_mask_rate: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = report.load_mask_rate_data(*saved_mask_rate)
    row = next(row for row in data.rows if row["condition_id"] == "M11" and row["k"] == 8
               and row["source_metric"] == "adjusted_lla_5")
    for field in ("mean", "repeat_1_mean", "repeat_2_mean", "repeat_3_mean"):
        row[field] = None
    original_save = report._save

    def inspect_and_save(figure: Figure, directory: Path, stem: str, dpi: int) -> None:
        lines = figure.axes[1].get_lines()[4:8]  # M11 repeats and mean in the LLA-5 panel.
        assert all(np.isnan(line.get_ydata()[CLUSTER_COUNTS.index(8)]) for line in lines)
        original_save(figure, directory, stem, dpi)

    monkeypatch.setattr(report, "_save", inspect_and_save)
    report._render_k_sweep(data, tmp_path, 100)


def test_failed_figure_render_keeps_existing_report(
    saved_mask_rate: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, snapshot = saved_mask_rate
    output = report.export_mask_rate_report(experiment, snapshot, tmp_path / "report")
    original = {path.name: path.read_bytes() for path in output.iterdir()}

    def fail_render(data: report.MaskRateData, directory: Path, dpi: int) -> None:
        raise OSError("figure render failed")

    monkeypatch.setattr(report, "_render_k_sweep", fail_render)
    with pytest.raises(OSError, match="figure render failed"):
        report.export_mask_rate_report(experiment, snapshot, output, overwrite=True,
                                       with_figure=True, dpi=100)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original


@pytest.mark.parametrize("dpi", (99, 601))
def test_figure_dpi_validation(tmp_path: Path, dpi: int) -> None:
    with pytest.raises(ValueError, match="DPI"):
        report.export_mask_rate_report(tmp_path / "experiment", "fixture", tmp_path / "report",
                                       with_figure=True, dpi=dpi)
