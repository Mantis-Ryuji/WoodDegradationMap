"""Small saved-artifact fixtures; no fits, spectra, inference or evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from wood_degradation_map.experiments import oof_sanity as sanity
from wood_degradation_map.experiments.aggregation import ScoreRecord
from wood_degradation_map.experiments.config import ADOPTED_SAMPLE_IDS, CLUSTER_COUNTS, FOLDS, REPEATS
from wood_degradation_map.experiments.manifests import _digest, _read_json, _write_json


def test_occupancy_excludes_background_and_keeps_single_cluster_samples() -> None:
    result = sanity.occupancy_values([0, 3, 0, 1], 4, 4)
    assert result["used_clusters"] == 2
    assert result["max_occupancy"] == 0.75
    assert result["single_cluster"] is False
    assert sanity.occupancy_values([0, 4], 4, 2)["single_cluster"] is True
    with pytest.raises(ValueError, match="occupancy"):
        sanity.occupancy_values([3, 0], 4, 2)
    with pytest.raises(ValueError, match="occupancy"):
        sanity.occupancy_values([True, 3], 4, 2)


@pytest.fixture
def saved_sources(tmp_path: Path) -> Path:
    experiment = tmp_path / "cv"
    (experiment / "manifests").mkdir(parents=True)
    (experiment / "config").mkdir()
    rows = [{"sample_id": sample, "test_fold": index % 5 + 1,
             "saved_pixel_count": 6, "height": 3, "width": 4}
            for index, sample in enumerate(ADOPTED_SAMPLE_IDS)]
    pd.DataFrame(rows).to_parquet(experiment / "manifests/folds.parquet", index=False)
    _write_json(experiment / "manifests/inputs.json", {
        "preprocessing_id": "production_v1", "samples": rows,
    })
    _write_json(experiment / "config/experiment.json", {
        "split": {"adopted_sample_ids": list(ADOPTED_SAMPLE_IDS)},
    })
    hashes = {name: _digest(experiment / name) for name in (
        "manifests/inputs.json", "manifests/folds.parquet", "config/experiment.json",
    )}
    _write_json(experiment / "manifests/complete.json", {"status": "complete", "artifact_sha256": hashes})
    for condition in sanity.CONDITIONS:
        for fold in FOLDS:
            test_ids = sorted(row["sample_id"] for row in rows if row["test_fold"] == fold)
            for repeat in REPEATS:
                suffix = f"{condition}/fold_{fold}/repeat_{repeat}"
                cluster = experiment / f"results/clustering/{suffix}"
                evaluation = experiment / f"results/evaluation/{suffix}"
                (cluster / "maps").mkdir(parents=True)
                evaluation.mkdir(parents=True)
                identity = {
                    "schema_version": 2, "condition": condition, "fold": fold, "repeat": repeat,
                    "test_sample_ids": test_ids, "manifest_artifact_sha256": hashes,
                    "source": {"kind": condition},
                }
                _write_json(cluster / "run.json", {
                    **identity, "mode": "clean_test_maps", "cluster_counts": list(CLUSTER_COUNTS),
                    "train_sample_ids": sorted(set(ADOPTED_SAMPLE_IDS) - set(test_ids)),
                })
                reports, scores = [], []
                for sample in test_ids:
                    map_file = f"maps/{sample}.npz"
                    if repeat == 1:
                        labels = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 2, 2]], dtype=np.int16)
                        if condition == "B1":
                            labels = np.array([0, 2, 1], dtype=np.int16)[labels]
                        np.savez_compressed(cluster / map_file, labels_k8=labels)
                    reports.append({
                        "sample_id": sample, "pixels": 6, "shape": [3, 4], "map_file": map_file,
                        "map_sha256": _digest(cluster / map_file) if (cluster / map_file).exists() else "unused",
                        "occupancy": {str(k): ([4, 2] if condition == "B0" else [2, 4])
                                      + [0] * (k - 2) for k in CLUSTER_COUNTS},
                    })
                    value = (ADOPTED_SAMPLE_IDS.index(sample) / 100 + repeat / 10
                             + sanity.CONDITIONS.index(condition) / 20)
                    scores.extend(asdict(ScoreRecord(sample, fold, condition, k, metric,
                                                     repeat, "defined", value / (index + 1)))
                                  for k in CLUSTER_COUNTS
                                  for index, metric in enumerate(sanity.REPEATED_METRICS))
                _write_json(cluster / "completion.json", {
                    "status": "clean_test_maps_completed", "checks_passed": True, "samples": reports,
                    "run_sha256": _digest(cluster / "run.json"),
                })
                _write_json(evaluation / "run.json", {
                    **identity, "mode": "full_test_evaluation",
                    "clustering_completion_sha256": _digest(cluster / "completion.json"),
                })
                _write_json(evaluation / "scores.json", {"records": scores})
                _write_json(evaluation / "completion.json", {
                    "status": "full_test_evaluation_completed", "checks_passed": True,
                    "run_sha256": _digest(evaluation / "run.json"),
                    "scores_sha256": _digest(evaluation / "scores.json"),
                })
    return experiment


def _change_scores(experiment: Path, change: str) -> None:
    directory = experiment / "results/evaluation/B0/fold_1/repeat_2"
    saved = _read_json(directory / "scores.json")
    if change == "undefined":
        saved["records"][0].update(value=None, status="undefined", reason="single_cluster")
    elif change == "missing":
        saved["records"].pop()
    elif change == "duplicate":
        saved["records"].append(saved["records"][0])
    elif change == "wrong_fold":
        saved["records"][0]["fold"] = 2
    elif change == "failed":
        saved["records"][0].update(value=None, status="failed", reason="fixture_failure")
    (directory / "scores.json").write_text(json.dumps(saved), encoding="utf-8")
    done = _read_json(directory / "completion.json")
    done["scores_sha256"] = _digest(directory / "scores.json")
    (directory / "completion.json").write_text(json.dumps(done), encoding="utf-8")


def test_saved_sources_keep_oof_macro_repeat_sd_and_provenance(saved_sources: Path) -> None:
    before = {str(path): _digest(path) for path in saved_sources.rglob("*") if path.is_file()}
    data = sanity.load_sanity_data(saved_sources)
    assert len(data.occupancy) == 49 * 2 * 3 * 7
    assert len(data.maps) == 14
    assert len(data.summaries) == 140
    summary = data.summaries[0]
    assert summary.common_samples == 49
    assert summary.repeat_macro_means == pytest.approx((0.34, 0.44, 0.54))
    assert summary.mean == pytest.approx(0.44)
    assert summary.repeat_sd == pytest.approx(0.1)
    assert summary.sample_sd == pytest.approx(np.std(np.arange(49) / 100, ddof=1))
    assert sum(entry["relative_path"].endswith(".npz") for entry in data.sources) == 98
    for _, sample in sanity.REPRESENTATIVES:
        np.testing.assert_array_equal(data.maps["B0", sample], data.maps["B1", sample])
    for matching in data.matching:
        assert matching["target_to_display_including_background"][:3] == [0, 2, 1]
        assert matching["overlap_fraction"] == 1.0
    assert before == {str(path): _digest(path) for path in saved_sources.rglob("*") if path.is_file()}


def test_undefined_scores_use_common_samples_and_preserve_reason(saved_sources: Path) -> None:
    _change_scores(saved_sources, "undefined")
    summary = sanity.load_sanity_data(saved_sources).summaries[0]
    assert summary.common_samples == 48
    assert summary.excluded_sample_ids == (ADOPTED_SAMPLE_IDS[0],)
    assert [item.defined_samples for item in summary.availability] == [49, 48, 49]
    assert summary.availability[1].unavailable[0].reason == "single_cluster"
    assert summary.repeat_macro_means == pytest.approx((0.345, 0.445, 0.545))


@pytest.mark.parametrize("change", ("missing", "duplicate", "wrong_fold", "failed"))
def test_incomplete_or_misassigned_scores_are_not_silently_averaged(saved_sources: Path, change: str) -> None:
    _change_scores(saved_sources, change)
    with pytest.raises(ValueError, match="coverage/identity"):
        sanity.load_sanity_data(saved_sources)


def test_changed_source_hash_is_rejected_before_output_is_created(saved_sources: Path, tmp_path: Path) -> None:
    source = saved_sources / "results/evaluation/B1/fold_5/repeat_3/scores.json"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / "sanity"
    with pytest.raises(ValueError, match="hash mismatch"):
        sanity.render_sanity(saved_sources, output)
    assert not output.exists()


def test_existing_output_and_cv_paths_are_refused_before_loading(tmp_path: Path) -> None:
    experiment, output = tmp_path / "cv", tmp_path / "sanity"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sanity.render_sanity(experiment, output)
    with pytest.raises(ValueError, match="separate"):
        sanity.render_sanity(experiment, experiment / "results/sanity")
    with pytest.raises(ValueError, match="separate"):
        sanity.render_sanity(experiment, tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_render_separates_conditions_but_groups_representatives_without_logs(
    saved_sources: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures = []

    def inspect_figure(figure: object, output: Path, stem: str, dpi: int) -> None:
        assert figure._suptitle is None
        assert all(not axis.get_title() for axis in figure.axes)
        assert dpi == 100
        if stem.endswith("_k_sweep"):
            assert len(figure.axes) == 1
            assert stem == "silhouette_k_sweep"
            assert figure.axes[0].get_ylabel() == "Cosine silhouette (sample macro)"
        else:
            assert stem in {f"{condition}_representatives_k8_repeat1" for condition in sanity.CONDITIONS}
            assert len(figure.axes) == 8  # Seven samples and the shared colorbar.
            assert all(len(axis.images) == 1 for axis in figure.axes[:7])
            for axis, (_, sample) in zip(figure.axes[:7], sanity.REPRESENTATIVES, strict=True):
                assert axis.images[0].get_array().shape == (3, 4)
                assert [text.get_text() for text in axis.texts] == [sample]
                assert axis.texts[0].get_position()[1] < 0
        figures.append((output.name, stem))
        sanity.plt.close(figure)

    monkeypatch.setattr(sanity, "_save_figure", inspect_figure)
    output = sanity.render_sanity(saved_sources, tmp_path / "sanity", dpi=100)
    assert len(figures) == 3
    assert {stem for _, stem in figures if stem.endswith("_k_sweep")} == {"silhouette_k_sweep"}
    assert not list(output.rglob("*.svg"))
    assert not list(output.rglob("*.json"))
    assert not list(output.glob("occupancy*.png"))
    assert {stem for directory, stem in figures if directory == "labels"} == {
        f"{condition}_representatives_k8_repeat1" for condition in sanity.CONDITIONS
    }
    assert not (output / "boundaries").exists()
    summary = pd.read_csv(output / "metrics_summary.csv")
    assert {"sample_sd", "repeat_sd", "repeat_1_macro_mean"} <= set(summary)
    assert len(summary) == 140
    assert set(summary["metric"]) == set(sanity.REPEATED_METRICS)
    matching = pd.read_csv(output / "matching.csv")
    assert len(matching) == 40
    assert set(matching["fold"]) == set(FOLDS)
    assert len(pd.read_csv(output / "occupancy.csv")) == 2058


def test_matching_pools_all_test_samples_not_only_representatives() -> None:
    maps, folds = {}, {}
    for fold, (_, sample) in zip(FOLDS, sanity.REPRESENTATIVES, strict=False):
        other = f"other{fold}"
        folds.update({sample: fold, other: fold})
        maps["B0", sample] = np.array([[1, 2]], dtype=np.int16)
        maps["B1", sample] = np.array([[1, 2]], dtype=np.int16)
        maps["B0", other] = np.array([[1] * 5 + [2] * 5], dtype=np.int16)
        maps["B1", other] = np.array([[2] * 5 + [1] * 5], dtype=np.int16)
    display, matching = sanity.match_fold_maps(maps, folds)
    for item in matching:
        assert item["target_to_display_including_background"][:3] == [0, 2, 1]
        assert item["matched_pixels"] == 10
        assert item["valid_pixels"] == 12
        sample = next(sample for sample in item["sample_ids"] if not sample.startswith("other"))
        np.testing.assert_array_equal(display["B1", sample], [[2, 1]])
        np.testing.assert_array_equal(maps["B1", sample], [[1, 2]])


def test_figure_writer_saves_png_only(tmp_path: Path) -> None:
    figure, axis = sanity.plt.subplots(figsize=(2, 2))
    axis.plot([0, 1], [0, 1])
    sanity._save_figure(figure, tmp_path, "metric", 100)
    assert [path.name for path in tmp_path.iterdir()] == ["metric.png"]
