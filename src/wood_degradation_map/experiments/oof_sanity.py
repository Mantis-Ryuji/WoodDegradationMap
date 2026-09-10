"""Read saved B0/B1 CV artifacts and render exploratory, title-free sanity figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from pandas.api.types import is_integer_dtype
from scipy.optimize import linear_sum_assignment

from .aggregation import REPEATED_METRICS, RepeatedSummary, ScoreRecord, aggregate_scores
from .config import ADOPTED_SAMPLE_IDS, CLUSTER_COUNTS, FOLDS, REPEATS
from .manifests import _digest, _read_json

CONDITIONS = ("B0", "B1")
DISPLAY_K = 8
DISPLAY_REPEAT = 1
REPRESENTATIVES = (
    ("クリ", "KYOw02789"), ("ケヤキ", "KYOw02777"), ("スギ", "KYOw02784"),
    ("ツガ", "KYOw02787"), ("ヒノキ", "KYOw02720"), ("マツ", "KYOw02769"),
    ("モミ", "KYOw16750"),
)
LABEL_COLORS = (
    "#ffffff", "#0072b2", "#e69f00", "#009e73", "#cc79a7",
    "#56b4e9", "#d55e00", "#f0e442", "#332288",
)
CONDITION_COLORS = ("#0072b2", "#d55e00")
@dataclass
class SanityData:
    expected_test_folds: dict[str, int]
    occupancy: pd.DataFrame
    summaries: tuple[RepeatedSummary, ...]
    maps: dict[tuple[str, str], np.ndarray]
    matching: list[dict[str, object]]
    sources: list[dict[str, object]]


class SourceReader:
    """Record hashes only for consumed artifacts; never open spectra or weights."""

    def __init__(self, experiment: Path) -> None:
        self.experiment = experiment.resolve()
        self.ledger: dict[str, dict[str, object]] = {}

    def path(self, relative: str) -> Path:
        path = (self.experiment / relative).resolve()
        if not path.is_relative_to(self.experiment):
            raise ValueError(f"Source path escapes experiment directory: {relative}")
        return path

    def record(self, relative: str, expected_hash: str | None = None) -> Path:
        path = self.path(relative)
        digest = _digest(path)
        if expected_hash is not None and digest != expected_hash:
            raise ValueError(f"Source hash mismatch: {relative}")
        previous = self.ledger.get(relative)
        if previous is not None and previous["sha256"] != digest:
            raise ValueError(f"Source changed during reading: {relative}")
        self.ledger[relative] = {
            "relative_path": relative, "path": str(path), "sha256": digest,
            "bytes": path.stat().st_size,
        }
        return path

    def read_json(self, relative: str, expected_hash: str | None = None) -> dict:
        return _read_json(self.record(relative, expected_hash))

    def verify_unchanged(self) -> None:
        for relative, entry in self.ledger.items():
            if _digest(self.path(relative)) != entry["sha256"]:
                raise ValueError(f"Source changed during visualization: {relative}")


def occupancy_values(counts: list[int], pixels: int, k: int) -> dict[str, object]:
    """Summarize saved clean-test cluster counts, excluding background."""
    if (type(pixels) is not int or pixels < 1 or len(counts) != k
            or any(type(count) is not int or count < 0 for count in counts)
            or sum(counts) != pixels):
        raise ValueError("Invalid saved occupancy counts or valid-pixel denominator")
    used = sum(count > 0 for count in counts)
    return {
        "valid_pixels": pixels, "used_clusters": used,
        "max_occupancy": max(counts) / pixels, "single_cluster": used == 1,
        "cluster_counts": json.dumps(counts),
    }


def _manifest(reader: SourceReader) -> tuple[pd.DataFrame, dict[str, str]]:
    done = reader.read_json("manifests/complete.json")
    if done.get("status") != "complete":
        raise ValueError("A complete saved manifest is required")
    hashes = done["artifact_sha256"]
    # Train coordinate tables are bound by the saved completion but not consumed.
    inputs = reader.read_json("manifests/inputs.json", hashes["manifests/inputs.json"])
    config = reader.read_json("config/experiment.json", hashes["config/experiment.json"])
    frame = pd.read_parquet(reader.record("manifests/folds.parquet", hashes["manifests/folds.parquet"]))
    required = {"sample_id", "test_fold", "saved_pixel_count", "height", "width"}
    if (not required <= set(frame) or frame["sample_id"].duplicated().any()
            or set(frame["sample_id"]) != set(ADOPTED_SAMPLE_IDS)
            or set(config["split"]["adopted_sample_ids"]) != set(ADOPTED_SAMPLE_IDS)
            or inputs["preprocessing_id"] != "production_v1"):
        raise ValueError("Expected exactly the fixed production_v1 49-sample inventory")
    for name in required - {"sample_id"}:
        if not is_integer_dtype(frame[name]) or frame[name].isna().any() or (frame[name] < 1).any():
            raise ValueError(f"Invalid manifest integer column: {name}")
    if set(frame["test_fold"]) != set(FOLDS):
        raise ValueError("Expected all five saved test folds")
    samples = {item["sample_id"]: item for item in inputs["samples"]}
    if len(samples) != len(inputs["samples"]) or set(samples) != set(frame["sample_id"]):
        raise ValueError("Input inventory and test-fold sample coverage differ")
    for row in frame.itertuples(index=False):
        for name in ("saved_pixel_count", "height", "width"):
            if getattr(row, name) != samples[row.sample_id][name]:
                raise ValueError(f"Input inventory and fold manifest differ: {row.sample_id}/{name}")
    return frame, hashes


def _run(
    reader: SourceReader, directory: str, condition: str, fold: int, repeat: int,
    test_ids: list[str], hashes: dict[str, str], *, evaluation: bool,
) -> tuple[dict, dict]:
    if reader.path(f"{directory}/failure.json").exists():
        raise ValueError(f"Failed or interrupted source run: {directory}")
    done = reader.read_json(f"{directory}/completion.json")
    status = "full_test_evaluation_completed" if evaluation else "clean_test_maps_completed"
    if done.get("status") != status or done.get("checks_passed") is not True:
        raise ValueError(f"Incomplete source run: {directory}")
    run = reader.read_json(f"{directory}/run.json", done["run_sha256"])
    expected = {
        "schema_version": 2, "condition": condition, "fold": fold, "repeat": repeat,
        "test_sample_ids": test_ids, "manifest_artifact_sha256": hashes,
        "mode": "full_test_evaluation" if evaluation else "clean_test_maps",
    }
    if not evaluation:
        expected["cluster_counts"] = list(CLUSTER_COUNTS)
        expected["train_sample_ids"] = sorted(set(ADOPTED_SAMPLE_IDS) - set(test_ids))
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Run identity, split or manifest mismatch: {directory}")
    return run, done


def match_fold_maps(
    maps: dict[tuple[str, str], np.ndarray], expected_test_folds: dict[str, int],
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict[str, object]]]:
    """Align B1 to B0 once per fold, pooling ALL test samples at K=8/repeat 1.

    The mapping maximizes total overlap and is shared by every sample in the fold.
    No source labels or metric records are changed. Zero-overlap assignments have
    no empirical support; save them explicitly instead of implying correspondence.
    """
    display, matching = {}, []
    representatives = {sample for _, sample in REPRESENTATIVES}
    for fold in FOLDS:
        sample_ids = sorted(sample for sample, value in expected_test_folds.items() if value == fold)
        contingency = np.zeros((DISPLAY_K, DISPLAY_K), dtype=np.int64)
        sample_counts = {}
        for sample in sample_ids:
            reference, target = maps["B0", sample], maps["B1", sample]
            if (reference.shape != target.shape or reference.ndim != 2
                    or reference.dtype.kind not in "iu" or target.dtype.kind not in "iu"
                    or np.any(reference < 0) or np.any(reference > DISPLAY_K)
                    or np.any(target < 0) or np.any(target > DISPLAY_K)
                    or not np.array_equal(reference > 0, target > 0)):
                raise ValueError(f"Invalid B0/B1 label maps or different valid masks: {sample}")
            valid = reference > 0
            pairs = (reference[valid].astype(np.int64) - 1) * DISPLAY_K + target[valid] - 1
            counts = np.bincount(pairs, minlength=DISPLAY_K ** 2).reshape(DISPLAY_K, DISPLAY_K)
            contingency += counts
            sample_counts[sample] = counts
        total = int(contingency.sum())
        if total == 0:
            raise ValueError(f"No valid test pixels for matching fold {fold}")
        reference_ids, target_ids = linear_sum_assignment(contingency, maximize=True)
        lookup = np.zeros(DISPLAY_K + 1, dtype=np.int16)
        lookup[target_ids + 1] = reference_ids + 1
        matched = int(contingency[reference_ids, target_ids].sum())
        matching.append({
            "fold": fold, "repeat": DISPLAY_REPEAT, "k": DISPLAY_K,
            "reference": "B0", "target": "B1", "sample_ids": sample_ids,
            "scope": "all test pixels pooled within this fold; one mapping shared by all samples",
            "contingency_rows": "B0 raw IDs 1..8", "contingency_columns": "B1 raw IDs 1..8",
            "contingency": contingency.tolist(), "target_to_display_including_background": lookup.tolist(),
            "valid_pixels": total, "matched_pixels": matched, "overlap_fraction": matched / total,
            "assignments": [{
                "b1_raw_id": int(target + 1), "b0_display_id": int(reference + 1),
                "overlap_pixels": int(contingency[reference, target]),
                "zero_overlap": bool(contingency[reference, target] == 0),
            } for reference, target in zip(reference_ids, target_ids, strict=True)],
            "samples": [{
                "sample_id": sample, "valid_pixels": int(counts.sum()),
                "matched_pixels": int(counts[reference_ids, target_ids].sum()),
                "overlap_fraction": float(counts[reference_ids, target_ids].sum() / counts.sum()),
            } for sample, counts in sample_counts.items()],
        })
        for sample in sample_ids:
            if sample in representatives:
                display["B0", sample] = maps["B0", sample]
                display["B1", sample] = lookup[maps["B1", sample]]
    return display, matching


def load_sanity_data(experiment: Path) -> SanityData:
    """Require 30-run coverage; use saved scores/counts and all 98 repeat-1 maps."""
    reader = SourceReader(experiment)
    frame, hashes = _manifest(reader)
    expected = {row.sample_id: int(row.test_fold) for row in frame.itertuples(index=False)}
    samples = frame.set_index("sample_id").to_dict("index")
    occupancy, records, maps = [], [], {}
    for condition in CONDITIONS:
        for fold in FOLDS:
            test_ids = sorted(sample for sample, test in expected.items() if test == fold)
            for repeat in REPEATS:
                suffix = f"{condition}/fold_{fold}/repeat_{repeat}"
                clustering = f"results/clustering/{suffix}"
                cluster_run, cluster_done = _run(
                    reader, clustering, condition, fold, repeat, test_ids, hashes, evaluation=False,
                )
                reports = {item["sample_id"]: item for item in cluster_done["samples"]}
                if len(reports) != len(cluster_done["samples"]) or set(reports) != set(test_ids):
                    raise ValueError(f"Clean map sample coverage mismatch: {suffix}")
                for sample in test_ids:
                    report, metadata = reports[sample], samples[sample]
                    if (report["pixels"] != metadata["saved_pixel_count"]
                            or report["shape"] != [metadata["height"], metadata["width"]]
                            or report["map_file"] != f"maps/{sample}.npz"
                            or set(report["occupancy"]) != {str(k) for k in CLUSTER_COUNTS}):
                        raise ValueError(f"Clean map metadata mismatch: {suffix}/{sample}")
                    for k in CLUSTER_COUNTS:
                        occupancy.append({
                            "condition": condition, "sample_id": sample, "fold": fold,
                            "repeat": repeat, "k": k,
                            **occupancy_values(report["occupancy"][str(k)], report["pixels"], k),
                        })
                    if repeat == DISPLAY_REPEAT:
                        path = reader.record(f"{clustering}/{report['map_file']}", report["map_sha256"])
                        with np.load(path, allow_pickle=False) as saved:
                            labels = saved[f"labels_k{DISPLAY_K}"]
                        if (list(labels.shape) != report["shape"] or labels.dtype.kind not in "iu"
                                or np.any(labels < 0) or np.any(labels > DISPLAY_K)
                                or np.bincount(labels.ravel(), minlength=DISPLAY_K + 1)[1:].tolist()
                                != report["occupancy"][str(DISPLAY_K)]):
                            raise ValueError(f"Label map/count mismatch: {path}")
                        maps[condition, sample] = labels
                evaluation = f"results/evaluation/{suffix}"
                run, done = _run(
                    reader, evaluation, condition, fold, repeat, test_ids, hashes, evaluation=True,
                )
                if (run["clustering_completion_sha256"]
                        != reader.ledger[f"{clustering}/completion.json"]["sha256"]
                        or run["source"] != cluster_run["source"]):
                    raise ValueError(f"Evaluation is not bound to these clean maps: {suffix}")
                saved_records = reader.read_json(f"{evaluation}/scores.json", done["scores_sha256"])
                selected = [ScoreRecord(**row) for row in saved_records["records"]
                            if row["metric"] in REPEATED_METRICS]
                keys = {(row.sample_id, row.k, row.metric) for row in selected}
                if (len(keys) != len(selected)
                        or keys != {(sample, k, metric) for sample in test_ids for k in CLUSTER_COUNTS
                                    for metric in REPEATED_METRICS}
                        or any((row.condition_id, row.fold, row.repeat) != (condition, fold, repeat)
                               or row.status not in ("defined", "undefined") for row in selected)):
                    raise ValueError(f"Metric score coverage/identity mismatch: {suffix}")
                records.extend(selected)
    display, matching = match_fold_maps(maps, expected)
    summaries = tuple(aggregate_scores(
        [row for row in records if row.condition_id == condition and row.k == k and row.metric == metric],
        expected_test_folds=expected, condition_id=condition, metric=metric, k=k,
    ) for metric in REPEATED_METRICS for condition in CONDITIONS for k in CLUSTER_COUNTS)
    reader.verify_unchanged()
    return SanityData(expected, pd.DataFrame(occupancy), summaries, display, matching,
                      [reader.ledger[name] for name in sorted(reader.ledger)])


def _save_figure(figure: plt.Figure, output: Path, stem: str, dpi: int) -> None:
    try:
        figure.savefig(output / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def _plot_maps(data: SanityData, output: Path, dpi: int) -> None:
    directory = output / "labels"
    directory.mkdir()
    cmap = ListedColormap(LABEL_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, DISPLAY_K + 1.5), cmap.N)
    for condition in CONDITIONS:
        fig, axes = plt.subplots(1, len(REPRESENTATIVES), figsize=(15, 4.8), squeeze=False)
        for axis, (_, sample) in zip(axes[0], REPRESENTATIVES, strict=True):
            labels = data.maps[condition, sample]
            axis.imshow(labels, cmap=cmap, norm=norm, interpolation="nearest")
            axis.set_axis_off()
            axis.text(0.5, -0.025, sample, transform=axis.transAxes,
                      ha="center", va="top", fontsize=9, clip_on=False)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.20, wspace=0.04)
        color_axis = fig.add_axes((0.30, 0.09, 0.40, 0.025))
        bar = fig.colorbar(axes[0, 0].images[0], cax=color_axis,
                           orientation="horizontal", ticks=np.arange(DISPLAY_K + 1))
        bar.set_label("B0 reference cluster ID within each fold (0 = background)")
        _save_figure(fig, directory, f"{condition}_representatives_k8_repeat1", dpi)


def _plot_silhouette(data: SanityData, output: Path, dpi: int) -> None:
    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for condition, color in zip(CONDITIONS, CONDITION_COLORS, strict=True):
        summaries = [row for row in data.summaries
                     if row.expression == condition and row.metric == "silhouette"]
        means = np.array([np.nan if row.mean is None else row.mean for row in summaries])
        sd = np.array([np.nan if row.repeat_sd is None else row.repeat_sd for row in summaries])
        axis.plot(CLUSTER_COUNTS, means, color=color, marker="o", linewidth=2, label=condition)
        axis.fill_between(CLUSTER_COUNTS, means - sd, means + sd, color=color, alpha=0.15)
        for index in range(len(REPEATS)):
            values = [row.repeat_macro_means[index] for row in summaries]
            axis.plot(CLUSTER_COUNTS, [np.nan if v is None else v for v in values],
                      color=color, linestyle=("--", ":", "-.")[index], alpha=0.6, linewidth=0.8)
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(Line2D([], [], color="#555555", linestyle=style, linewidth=0.8)
                   for style in ("--", ":", "-."))
    labels.extend(f"Repeat {repeat} macro" for repeat in REPEATS)
    axis.legend(handles, labels, frameon=False, fontsize=9)
    axis.set_ylabel("Cosine silhouette (sample macro)")
    axis.set_xticks(CLUSTER_COUNTS)
    axis.set_xlabel("K")
    axis.grid(alpha=0.2)
    _save_figure(fig, output, "silhouette_k_sweep", dpi)


def _summary_frame(data: SanityData) -> pd.DataFrame:
    rows = []
    for summary in data.summaries:
        row = {
            "metric": summary.metric, "condition": summary.expression, "k": summary.k,
            "total_samples": summary.total_samples, "common_samples": summary.common_samples,
            "mean": summary.mean, "sample_sd": summary.sample_sd, "repeat_sd": summary.repeat_sd,
            "common_sample_ids": json.dumps(summary.common_sample_ids),
            "excluded_sample_ids": json.dumps(summary.excluded_sample_ids),
            "mean_undefined_reason": summary.mean_undefined_reason,
            "sample_sd_undefined_reason": summary.sample_sd_undefined_reason,
            "repeat_sd_undefined_reason": summary.repeat_sd_undefined_reason,
        }
        for repeat, value, availability in zip(
            REPEATS, summary.repeat_macro_means, summary.availability, strict=True,
        ):
            row[f"repeat_{repeat}_macro_mean"] = value
            row[f"repeat_{repeat}_defined_samples"] = availability.defined_samples
            row[f"repeat_{repeat}_unavailable"] = json.dumps([
                {"sample_id": item.sample_id, "status": item.status, "reason": item.reason}
                for item in availability.unavailable
            ])
        rows.append(row)
    return pd.DataFrame(rows)


def _matching_frame(data: SanityData) -> pd.DataFrame:
    # Each row is an assigned B1 ID, with its overlap against all B0 IDs. This
    # retains the complete contingency matrix without extra diagnostic files.
    return pd.DataFrame([{
        "fold": fold["fold"], "repeat": DISPLAY_REPEAT, "k": DISPLAY_K,
        "reference": "B0", "target": "B1", **assignment,
        "fold_valid_pixels": fold["valid_pixels"],
        "fold_overlap_fraction": fold["overlap_fraction"],
        "fold_sample_ids": json.dumps(fold["sample_ids"]),
        **{f"overlap_b0_id_{index + 1}": counts[assignment["b1_raw_id"] - 1]
           for index, counts in enumerate(fold["contingency"])},
    } for fold in data.matching for assignment in fold["assignments"]])


def render_sanity(experiment: Path, output: Path, *, dpi: int = 240) -> Path:
    """Save two condition-wise map sheets, a silhouette PNG and three CSV tables."""
    experiment, output = experiment.resolve(), output.resolve()
    if output.is_relative_to(experiment) or experiment.is_relative_to(output):
        raise ValueError("Sanity output must be separate from the CV experiment directory")
    if output.exists():
        raise FileExistsError(f"Sanity output already exists: {output}")
    if type(dpi) is not int or not 100 <= dpi <= 600:
        raise ValueError("DPI must be an integer between 100 and 600")
    data = load_sanity_data(experiment)
    output.mkdir(parents=True, exist_ok=False)
    data.occupancy.to_csv(output / "occupancy.csv", index=False)
    _matching_frame(data).to_csv(output / "matching.csv", index=False)
    _summary_frame(data).to_csv(output / "metrics_summary.csv", index=False)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        _plot_maps(data, output, dpi)
        _plot_silhouette(data, output, dpi)
    for entry in data.sources:
        if _digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Source changed during visualization: {entry['relative_path']}")
    return output
