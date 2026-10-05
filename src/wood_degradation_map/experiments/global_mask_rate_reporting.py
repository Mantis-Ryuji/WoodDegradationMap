"""Appendix comparison tables and the prespecified M11(-50)/M11-75 2 x 7 figure."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from .config import REPEATS
from .global_clustering import K, read_label_map
from .global_k8_aggregate import summarize_repeats
from .global_k8_evaluation import check_condition_evaluation, condition_output
from .global_k8_repeats import check_repeat_clustering, repeat_paths
from .global_manifest import GlobalData
from .global_mask_rate_pipeline import (
    METRICS, check_evaluation, check_sealed, cluster_paths, evaluation_path, seal,
)
from .global_mask_rate_training import CONDITION, SNAPSHOT, training_paths
from .global_reporting import (
    DEFAULT_DIRECTORY, REPRESENTATIVE_FIGURE, _paste_cluster_colorbar, equal_sample_summary,
)
from .global_seed_plan import global_kmeans_seed
from .input_validation import InputInventory
from .manifests import _digest, _read_json, _require, _write_json
from .oof_sanity import LABEL_COLORS, REPRESENTATIVES

CONDITIONS = ("M11", CONDITION)
FIGURE = "01_representative_samples_2x7.png"
DISPLAY_NAMES = {"M11": "M11(-50)", CONDITION: "M11-75"}
CAPTION = (
    "Appendix supplementary comparison of M11(-50) (top) and M11-75 (bottom). "
    "Each representation was fitted once on the same fixed pixels of all 49 samples; "
    "K=8 Cosine-KMeans repeat 1 is displayed, without seed selection. "
    "Columns retain the seven prespecified samples from the main global figure. "
    "Colors follow the main M00 reference using equal-sample observed SNV means and "
    "maximum-sum cosine assignment, for display only; they are not chemical labels. "
    "Inspect growth-ring-related divisions, edge/interior divisions, fine intermixing, "
    "and spatial continuity. This is a descriptive full-fit sensitivity comparison, "
    "not held-out evaluation or selection of an optimal mask rate."
)


def report_path(experiment: Path) -> Path:
    return experiment / f"results/figures/{SNAPSHOT}"


def comparison_tables(
    rows_by_condition: dict[str, list[dict[str, object]]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Pair conditions and repeats on the same defined samples for each metric."""
    _require(set(rows_by_condition) == set(CONDITIONS), "Expected M11 and M11-75")
    by_condition = {}
    for condition, rows in rows_by_condition.items():
        by_repeat = {repeat: {} for repeat in REPEATS}
        for row in rows:
            repeat, sample_id = row["repeat"], row["sample_id"]
            _require(repeat in REPEATS and sample_id not in by_repeat[repeat],
                     "Duplicate or invalid sample/repeat")
            by_repeat[repeat][sample_id] = row
        by_condition[condition] = by_repeat
    ids = set(by_condition["M11"][1])
    _require(bool(ids) and all(set(rows) == ids for by_repeat in by_condition.values()
                              for rows in by_repeat.values()), "Comparison sample sets differ")
    summary_rows, details = [], {}
    for metric in METRICS:
        original = {condition: summarize_repeats(by_condition[condition], metric)
                    for condition in CONDITIONS}
        common = set.intersection(*(set(value["common_sample_ids"]) for value in original.values()))
        masked = {}
        for condition in CONDITIONS:
            masked[condition] = {repeat: {
                sample_id: {metric: row[metric] if sample_id in common else None,
                            "undefined_reasons": {metric: "outside_common_condition_repeat_samples"}}
                for sample_id, row in by_condition[condition][repeat].items()}
                for repeat in REPEATS}
        difference = f"{CONDITION} - M11"
        masked[difference] = {repeat: {
            sample_id: {metric: (by_condition[CONDITION][repeat][sample_id][metric]
                                - by_condition["M11"][repeat][sample_id][metric])
                       if sample_id in common else None,
                       "undefined_reasons": {metric: "outside_common_condition_repeat_samples"}}
            for sample_id in sorted(ids)} for repeat in REPEATS}
        summaries = {condition: summarize_repeats(values, metric)
                     for condition, values in masked.items()}
        details[metric] = {"original_availability": original, "paired_summaries": summaries}
        for condition, summary in summaries.items():
            summary_rows.append({
                "condition": condition, "metric": metric, "total_samples": len(ids),
                "common_samples": len(common), "mean": summary["mean"],
                "sample_sd": summary["sample_sd"], "repeat_sd": summary["repeat_sd"],
                **{f"repeat_{r}_mean": summary["repeat_macro_means"][r] for r in REPEATS},
                "excluded_sample_ids": json.dumps(sorted(ids - common)),
                "mean_undefined_reason": summary["mean_undefined_reason"],
            })
    return pd.DataFrame(summary_rows), details


def spectral_mapping(
    reference: np.ndarray, sample_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the original global SNV assignment; absent clusters stay explicitly undefined."""
    _require(reference.shape == (K, 256) and sample_means.ndim == 3
             and sample_means.shape[1:] == (K, 256) and len(sample_means) > 0,
             "Invalid SNV summary shape")
    norms = np.linalg.norm(reference, axis=1)
    _require(np.isfinite(reference).all() and np.isfinite(norms).all() and np.all(norms > 0),
             "Invalid M00 SNV reference")
    means = np.stack([equal_sample_summary(sample_means[:, label])[0] for label in range(K)])
    present = np.isfinite(means).all(axis=1)
    _require(present.any(), "No observed M11-75 SNV clusters")
    norms_target = np.linalg.norm(means[present], axis=1)
    _require(np.isfinite(norms_target).all() and np.all(norms_target > 0),
             "Zero or nonfinite norm of observed SNV representative")
    similarity = np.full((K, K), np.nan)
    similarity[:, present] = np.clip((reference / norms[:, None])
                                     @ (means[present] / norms_target[:, None]).T, -1, 1)
    ref, target = linear_sum_assignment(similarity[:, present], maximize=True)
    mapping = np.zeros(K + 1, dtype=np.uint8)
    mapping[np.flatnonzero(present)[target] + 1] = ref + 1
    # Empty clusters have no displayed pixels and do not contribute a similarity.
    unused = sorted(set(range(1, K + 1)) - set(mapping[1:]))
    mapping[np.flatnonzero(~present) + 1] = unused
    return mapping, similarity


def _reference_contract(
    experiment: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, str]:
    output = experiment / DEFAULT_DIRECTORY
    report = _read_json(output / "report.json")
    completion = _read_json(output / "completion.json")
    contract = report["contract"]
    _require(completion.get("status") == "global_report_completed"
             and completion.get("checks_passed") is True
             and contract["K"] == K and contract["repeat"] == 1
             and contract["seed"] == global_kmeans_seed(1)
             and contract["manifest_sha256"] == _digest(experiment / "manifests/complete.json")
             and contract["reference"] == "M00+Cosine-KMeans"
             and report["sample_ids"] == list(data.train_sample_ids)
             and report["representative_sample_ids"] == [sid for _, sid in REPRESENTATIVES],
             "Main global figure reference differs")
    for condition in ("M00", "M11"):
        check_repeat_clustering(experiment, data, inventory, condition, 1)
        _require(contract["sources"][condition] == _digest(
            repeat_paths(experiment, condition, 1)[0] / "completion.json"),
            "Main figure no longer refers to the verified first KMeans fit")
    names = ("report.json", REPRESENTATIVE_FIGURE, "spectral_summary.npz", "M11/label_maps.npz")
    hashes = {name: _digest(output / name) for name in names}
    _require(all(completion["artifact_sha256"].get(name) == value
                 for name, value in hashes.items()), "Main figure/SNV reference artifact changed")
    return hashes


def report_contract(
    experiment: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    check_condition_evaluation(experiment, data, inventory, "M11")
    check_evaluation(experiment, data, inventory)
    return {
        "scope": SNAPSHOT, "publication": "appendix supplementary comparison", "K": K,
        "conditions": list(CONDITIONS), "display_repeat": 1,
        "reference_figure": _reference_contract(experiment, data, inventory),
        "evaluations": {"M11": _digest(condition_output(experiment, "M11") / "completion.json"),
                        CONDITION: _digest(evaluation_path(experiment) / "completion.json")},
        "representatives": [sid for _, sid in REPRESENTATIVES],
        "runtime": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "matplotlib", "pillow")},
        "code_sha256": {name: _digest(Path(__file__).with_name(name)) for name in (
            Path(__file__).name, "global_reporting.py", "global_k8_aggregate.py", "oof_sanity.py")},
    }


def plot_representatives(maps: dict[str, dict[str, np.ndarray]], output: Path, dpi: int) -> None:
    """Retain the main figure's column order, aspect ratios and categorical pixels."""
    _require(type(dpi) is int and dpi > 0, "DPI must be positive")
    sample_ids = [sid for _, sid in REPRESENTATIVES]
    _require(set(maps) == set(CONDITIONS)
             and all(set(rows) == set(sample_ids) for rows in maps.values()),
             "Expected exactly two rows and the seven prespecified samples")
    for sample_id in sample_ids:
        first, second = (maps[c][sample_id] for c in CONDITIONS)
        _require(first.ndim == 2 and min(first.shape) > 0 and first.shape == second.shape
                 and np.array_equal(first > 0, second > 0)
                 and all(labels.dtype == np.uint8 and np.all(labels <= K)
                         for labels in (first, second)), "Paired map coverage differs")
    width, gutter, label_height, font_size, bar_height, left = (
        max(1, round(value * dpi / 240)) for value in (720, 24, 128, 80, 360, 420))
    height = max(round(width * labels.shape[0] / labels.shape[1])
                 for labels in maps["M11"].values())
    size = (left + 7 * width + 8 * gutter, 2 * height + 3 * gutter + label_height + bar_height)
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path())
                                 / "fonts/ttf/DejaVuSans-Bold.ttf"), size=font_size)
    palette = np.rint(np.array([to_rgb(color) for color in LABEL_COLORS]) * 255).astype(np.uint8)
    with Image.new("RGB", size, "white") as canvas:
        draw = ImageDraw.Draw(canvas)
        for row, condition in enumerate(CONDITIONS):
            y = gutter + row * (height + gutter)
            draw.text((left // 2, y + height // 2), DISPLAY_NAMES[condition],
                      fill="black", font=font, anchor="mm")
            for column, sample_id in enumerate(sample_ids):
                x = left + gutter + column * (width + gutter)
                with Image.fromarray(palette[maps[condition][sample_id]]) as source:
                    ratio = min(width / source.width, height / source.height)
                    with source.resize((max(1, round(source.width * ratio)),
                                        max(1, round(source.height * ratio))),
                                       Image.Resampling.NEAREST) as display:
                        canvas.paste(display, (x + (width - display.width) // 2,
                                               y + (height - display.height) // 2))
                if row == 1:
                    draw.text((x + width // 2, y + height + label_height // 2), sample_id,
                              fill="black", font=font, anchor="mm")
        _paste_cluster_colorbar(canvas, dpi=dpi, height=bar_height)
        with output.open("xb") as handle:
            canvas.save(handle, format="PNG", dpi=(dpi, dpi))


def write_report(
    experiment: Path, data: GlobalData, inventory: InputInventory, *, dpi: int = 240,
) -> dict[str, object]:
    output = report_path(experiment)
    _require(not output.exists(), "Appendix report exists; use check")
    contract = report_contract(experiment, data, inventory)
    rows = {condition: _read_json(directory / "samples.json")["samples"] for condition, directory
            in (("M11", condition_output(experiment, "M11")), (CONDITION, evaluation_path(experiment)))}
    summary, details = comparison_tables(rows)
    reference = experiment / DEFAULT_DIRECTORY
    with np.load(reference / "spectral_summary.npz", allow_pickle=False) as saved:
        reference_index = saved["conditions"].tolist().index("M00")
        snv_index = saved["kinds"].tolist().index("snv")
        reference_means = saved["means"][reference_index, :, snv_index].copy()
        original_mapping = saved["original_to_display"][saved["conditions"].tolist().index("M11")]
    with np.load(evaluation_path(experiment) / "snv_means.npz", allow_pickle=False) as saved:
        mapping, similarity = spectral_mapping(reference_means, saved["means"])
    sample_ids = [sid for _, sid in REPRESENTATIVES]
    samples = {sample.sample_id: sample for sample in inventory.samples}
    maps = {condition: {} for condition in CONDITIONS}
    with np.load(reference / "M11/label_maps.npz", allow_pickle=False) as saved:
        for sample_id in sample_ids:
            raw = read_label_map(repeat_paths(experiment, "M11", 1)[0]
                                 / "maps" / f"{sample_id}.npz", samples[sample_id])
            _require(np.array_equal(saved[sample_id], original_mapping[raw]),
                     "M11 row differs from the main figure's first fit")
            maps["M11"][sample_id] = saved[sample_id].copy()
            maps[CONDITION][sample_id] = mapping[read_label_map(
                cluster_paths(experiment, 1)[0] / "maps" / f"{sample_id}.npz", samples[sample_id])]
    with TemporaryDirectory(prefix=".mask-rate-report-", dir=experiment) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        summary.to_csv(stage / "summary.csv", index=False)
        _write_json(stage / "summary.json", details)
        sample_rows = [{"condition": condition, **{key: row[key] for key in (
            "sample_id", "repeat", "valid_pixels", *METRICS)},
            "occupancy": json.dumps(row["occupancy"]),
            "undefined_reasons": json.dumps(row["undefined_reasons"])}
            for condition in CONDITIONS for row in rows[condition]]
        pd.DataFrame(sample_rows).to_csv(stage / "sample_metrics.csv", index=False)
        pd.DataFrame([{"original_cluster": label, "display_cluster": int(mapping[label]),
                       "snv_cosine_similarity": similarity[mapping[label] - 1, label - 1]}
                      for label in range(1, K + 1)]).to_csv(stage / "matching.csv", index=False)
        plot_representatives(maps, stage / FIGURE, dpi)
        pd.DataFrame([{"figure": FIGURE, "placement": "appendix", "caption": CAPTION}]).to_csv(
            stage / "captions.csv", index=False)
        training_seconds = {condition: _read_json(directory / "completion.json")["training_seconds"]
                            for condition, directory in (
                                ("M11", experiment / "results/neural/M11/repeat_1"),
                                (CONDITION, training_paths(experiment)[0]))}
        _write_json(stage / "run.json", {
            "contract": contract, "dpi": dpi, "training_seconds": training_seconds,
            "caption": CAPTION,
            "definitions": {
                "LLA": "occupancy-adjusted local label agreement; windows 3, 5, 9",
                "silhouette": "pooled all-valid-pixel cosine distances, then equal-sample mean",
                "comparison": "common defined samples across both conditions and all three seeds",
                "difference": "M11-75 minus M11(-50), paired by sample and KMeans seed",
                "sample_sd": "SD of per-sample three-repeat means, ddof=1",
                "repeat_sd": "SD of three equal-sample macro means, ddof=1; KMeans variation only",
                "occupancy": "per-sample used clusters and maximum fraction, including absent IDs",
                "color_matching": "M00 observed SNV reference; display only, scores use original IDs",
            },
        })
        _require(report_contract(experiment, data, inventory) == contract,
                 "Comparison sources changed during report generation")
        completion = seal(stage, "mask_rate_report_completed")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output)
    return completion


def check_report(
    experiment: Path, data: GlobalData, inventory: InputInventory,
) -> dict[str, object]:
    return check_sealed(report_path(experiment), "mask_rate_report_completed",
                        report_contract(experiment, data, inventory))
