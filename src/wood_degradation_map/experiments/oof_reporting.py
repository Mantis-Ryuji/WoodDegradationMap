"""Render the main OOF snapshot as figures and CSV tables, without fitting models.

LLA means the occupancy-corrected score (saved key ``adjusted_lla_*``).
The three prespecified windows remain separate; no window is selected by results.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from .aggregation import Availability, SummaryRow, UnavailableScore, _summarize
from .config import CLUSTER_COUNTS, CONDITIONS, FOLDS, REPEATS
from .manifests import _digest, _write_json
from .oof_pipeline import FACTORIAL_CONDITIONS, PLANNED_PAIRS
from .oof_sanity import SourceReader, _manifest, _run, occupancy_values

MAIN_CONDITIONS = tuple(item.condition_id for item in CONDITIONS if item.experiment == "main")
K0 = 8
LLA_METRICS = ("adjusted_lla_3", "adjusted_lla_5", "adjusted_lla_9")
LFR_METRICS = ("lfr_both", "lfr_noise", "lfr_shift")
REPEATED = (*LLA_METRICS, *LFR_METRICS, "silhouette")
CONTRAST_METRICS = (*LLA_METRICS, *LFR_METRICS)
OCCUPANCY_METRICS = ("used_clusters", "max_occupancy", "single_cluster")
# The six figure panels are a subset of the complete CSV metrics.
SUMMARY_METRICS = (*LLA_METRICS, "lfr_both", "ari", "silhouette")
TABLE_METRICS = (*LLA_METRICS, *LFR_METRICS, "ari", "silhouette")
DISPLAY_ORDER = (*TABLE_METRICS, *OCCUPANCY_METRICS)
FIGURE_NAMES = (
    "01_main_metrics_k_sweep", "02_k8_distributions", "03_paired_k_sweep",
)
OBSOLETE_FIGURES = (
    "01_lla_k_sweep", "02_lfr_both_k_sweep", "03_ari_k_sweep",
    "04_cosine_silhouette_k_sweep", "05_cluster_occupancy_k_sweep", "07_paired_k8",
    "06_k8_distributions", "08_paired_k_sweep", "09_interaction_k_sweep",
)
TABLE_NAMES = (
    "metrics_all_k", "paired_all_k", "interaction_all_k", "sample_values", "availability",
    "ari_pairs", "occupancy_samples", "occupancy_folds", "metrics_k8", "paired_k8", "interaction_k8",
)
COLORS = dict(zip(MAIN_CONDITIONS, (
    "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#332288",
), strict=True))
LABELS = {
    **{f"adjusted_lla_{w}": f"LLA ({w} x {w})" for w in (3, 5, 9)},
    "lfr_both": "LFR(TGN+FS)", "lfr_noise": "LFR(TGN)", "lfr_shift": "LFR(FS)",
    "ari": "ARI", "silhouette": "Cosine-Silhouette",
    "used_clusters": "Cluster Occupancy: used clusters",
    "max_occupancy": "Cluster Occupancy: maximum fraction",
    "single_cluster": "Cluster Occupancy: single-cluster fraction",
}
STYLES = ("--", ":", "-.")


@dataclass
class ReportData:
    summaries: pd.DataFrame
    samples: pd.DataFrame
    availability: pd.DataFrame
    ari_pairs: pd.DataFrame
    occupancy: pd.DataFrame
    occupancy_folds: pd.DataFrame
    reader: SourceReader


def _identity(metric: str, expression: str, k: int, kind: str) -> dict:
    priority = (1 if metric in LLA_METRICS else 2 if metric in LFR_METRICS else
                3 if metric == "ari" else 4 if metric == "silhouette" else 5)
    return {
        "priority": priority, "metric": "LLA" if metric in LLA_METRICS else
        "Cluster Occupancy" if metric in OCCUPANCY_METRICS else LABELS[metric],
        "window": int(metric.rsplit("_", 1)[1]) if metric in LLA_METRICS else None,
        "source_metric": metric, "label": LABELS[metric], "expression": expression,
        "kind": kind, "k": k,
    }


def _collect_repeated(
    summary: dict, kind: str, summaries: list[dict], samples: list[dict],
    availability: list[dict], expected: dict[str, int],
) -> None:
    identity = _identity(summary["metric"], summary["expression"], summary["k"], kind)
    if (summary["total_samples"] != len(expected) or summary["repeat_ids"] != list(REPEATS)
            or summary["has_failed_or_interrupted_sources"]):
        raise ValueError("Incomplete repeated summary")
    indexed = {(row["sample_id"], row["repeat"]): row for row in summary["rows"]}
    required = {(sample, repeat) for sample in expected for repeat in REPEATS}
    if (len(indexed) != len(summary["rows"]) or set(indexed) != required
            or any(row["fold"] != expected[row["sample_id"]] for row in indexed.values())):
        raise ValueError("Repeated summary sample/fold/repeat coverage mismatch")
    common = {sample for sample in expected
              if all(indexed[sample, repeat]["value"] is not None for repeat in REPEATS)}
    means = {row["sample_id"]: row for row in summary["samples"]}
    if (common != set(summary["common_sample_ids"]) or len(common) != summary["common_samples"]
            or set(means) != common or len(means) != len(summary["samples"])):
        raise ValueError("Repeated summary common sample mismatch")
    summaries.append({
        **identity, **{key: summary[key] for key in (
            "total_samples", "common_samples", "mean", "sample_sd", "repeat_sd",
            "mean_undefined_reason", "sample_sd_undefined_reason", "repeat_sd_undefined_reason",
        )}, "common_sample_ids": json.dumps(summary["common_sample_ids"]),
        "excluded_sample_ids": json.dumps(summary["excluded_sample_ids"]),
        **{f"repeat_{repeat}_mean": value for repeat, value in
           zip(REPEATS, summary["repeat_macro_means"], strict=True)},
    })
    for sample, fold in sorted(expected.items()):
        samples.append({
            **identity, "sample_id": sample, "fold": fold, "included": sample in common,
            "mean": means[sample]["mean"] if sample in common else None,
            **{f"repeat_{repeat}": indexed[sample, repeat]["value"] for repeat in REPEATS},
        })
    for row in summary["availability"]:
        availability.append({**identity, **{key: row[key] for key in (
            "condition_id", "repeat", "total_samples", "defined_samples",
        )}, "unavailable": json.dumps(row["unavailable"], ensure_ascii=False)})


def corrected_interaction(parts: dict[str, dict], expected: dict[str, int]) -> dict:
    """Use original per-sample/repeat values, then intersect all four conditions.

    The existing snapshot's interaction uses raw LLA. It must never be relabeled
    as LLA: derive the corrected contrast without modifying that snapshot.
    """
    first = parts["M11"]
    indexed = {condition: {(row["sample_id"], row["repeat"]): row for row in part["rows"]}
               for condition, part in parts.items()}
    rows = []
    for sample, fold in sorted(expected.items()):
        for repeat in REPEATS:
            sources = [indexed[c][sample, repeat] for c in FACTORIAL_CONDITIONS]
            unavailable = tuple(UnavailableScore(**item) for source in sources
                                for item in source["unavailable_sources"])
            value = None
            if not unavailable:
                a, b, c, d = (np.float32(source["value"]) for source in sources)
                value = float((a - b) - (c - d))
                if not np.isfinite(value):
                    raise ValueError("Nonfinite interaction")
            rows.append(SummaryRow(sample, fold, repeat, value, unavailable))
    available = tuple(Availability(
        item["condition_id"], item["repeat"], item["total_samples"], item["defined_samples"],
        tuple(UnavailableScore(**row) for row in item["unavailable"]),
    ) for condition in FACTORIAL_CONDITIONS for item in parts[condition]["availability"])
    return asdict(_summarize(tuple(rows), available, expected,
                            "(M11 - M10) - (M01 - M00)", first["metric"], first["k"]))


def _load_occupancy(
    reader: SourceReader, run: dict, expected: dict[str, int], frame: pd.DataFrame,
    hashes: dict[str, str],
) -> pd.DataFrame:
    rows = []
    seen = set()
    metadata = frame.set_index("sample_id").to_dict("index")
    for source in run["sources"]:
        condition, fold, repeat = (source[key] for key in ("condition", "fold", "repeat"))
        key = condition, fold, repeat
        if key in seen or key not in {(c, f, r) for c in MAIN_CONDITIONS
                                     for f in FOLDS for r in REPEATS}:
            raise ValueError("Duplicate or unexpected OOF source run")
        seen.add(key)
        suffix = f"{condition}/fold_{fold}/repeat_{repeat}"
        evaluation, cluster = f"results/evaluation/{suffix}", f"results/clustering/{suffix}"
        if source["completion"] != f"{evaluation}/completion.json":
            raise ValueError("OOF source path mismatch")
        done = reader.read_json(source["completion"], source["sha256"])
        if done["shared_inputs_sha256"] != source["shared_inputs_sha256"]:
            raise ValueError("OOF shared evaluation input mismatch")
        test_ids = sorted(sample for sample, test_fold in expected.items() if test_fold == fold)
        evaluation_run, _ = _run(reader, evaluation, condition, fold, repeat,
                                 test_ids, hashes, evaluation=True)
        reader.record(f"{cluster}/completion.json", evaluation_run["clustering_completion_sha256"])
        cluster_run, cluster_done = _run(reader, cluster, condition, fold, repeat,
                                         test_ids, hashes, evaluation=False)
        if evaluation_run["source"] != cluster_run["source"]:
            raise ValueError("Evaluation/clustering representation mismatch")
        reports = {item["sample_id"]: item for item in cluster_done["samples"]}
        if set(reports) != set(test_ids) or len(reports) != len(cluster_done["samples"]):
            raise ValueError("Occupancy sample coverage mismatch")
        for sample, report in reports.items():
            if (report["pixels"] != metadata[sample]["saved_pixel_count"]
                    or set(report["occupancy"]) != {str(k) for k in CLUSTER_COUNTS}):
                raise ValueError("Occupancy pixel count/K mismatch")
            for k in CLUSTER_COUNTS:
                counts = report["occupancy"][str(k)]
                rows.append({"condition": condition, "fold": fold, "repeat": repeat,
                             "sample_id": sample, "k": k, "scope": "clean_test",
                             **occupancy_values(counts, report["pixels"], k),
                             "cluster_fractions": json.dumps([n / report["pixels"] for n in counts])})
    if len(seen) != len(MAIN_CONDITIONS) * len(FOLDS) * len(REPEATS):
        raise ValueError("Missing OOF source run")
    return pd.DataFrame(rows)


def _occupancy_tables(frame: pd.DataFrame) -> tuple[list[dict], list[dict], pd.DataFrame]:
    summaries, samples, folds = [], [], []
    for condition in MAIN_CONDITIONS:
        for k in CLUSTER_COUNTS:
            group = frame[(frame.condition == condition) & (frame.k == k)]
            for metric in OCCUPANCY_METRICS:
                matrix = group.pivot(index="sample_id", columns="repeat", values=metric)
                matrix = matrix.reindex(columns=REPEATS).astype(float)
                if matrix.isna().any().any():
                    raise ValueError("Incomplete occupancy repeats")
                sample_means, repeat_means = matrix.mean(axis=1), matrix.mean(axis=0)
                identity = _identity(metric, condition, k, "condition")
                summaries.append({
                    **identity, "total_samples": len(matrix), "common_samples": len(matrix),
                    "mean": float(repeat_means.mean()), "sample_sd": float(sample_means.std(ddof=1)),
                    "repeat_sd": float(repeat_means.std(ddof=1)),
                    "common_sample_ids": json.dumps(matrix.index.tolist()), "excluded_sample_ids": "[]",
                    **{f"repeat_{r}_mean": float(repeat_means[r]) for r in REPEATS},
                    **({f"repeat_{r}_single_cluster_samples": int(matrix[r].sum()) for r in REPEATS}
                       if metric == "single_cluster" else {}),
                })
                fold_ids = group.drop_duplicates("sample_id").set_index("sample_id")["fold"]
                samples.extend({**identity, "sample_id": sample, "fold": int(fold_ids[sample]),
                                "included": True, "mean": float(sample_means[sample]),
                                **{f"repeat_{r}": float(matrix.loc[sample, r]) for r in REPEATS}}
                               for sample in matrix.index)
            # Raw cluster IDs are only meaningful inside this condition/fold/repeat.
            for (fold, repeat), subset in group.groupby(["fold", "repeat"], sort=True):
                counts = np.array([json.loads(item) for item in subset.cluster_counts])
                fractions = counts / subset.valid_pixels.to_numpy()[:, None]
                for index in range(k):
                    folds.append({"condition": condition, "k": k, "fold": int(fold),
                                  "repeat": int(repeat), "scope": "clean_test", "cluster_id": index + 1,
                                  "pooled_pixels": int(counts[:, index].sum()),
                                  "pooled_fraction": float(counts[:, index].sum() / counts.sum()),
                                  "sample_macro_fraction": float(fractions[:, index].mean())})
    return summaries, samples, pd.DataFrame(folds)


def paired_ari(left: dict, right: dict, expected: dict[str, int]) -> tuple[dict, list[dict]]:
    """Subtract each sample's three-pair ARI mean on the common defined sample set."""
    identity = _identity("ari", f"{left['condition_id']} - {right['condition_id']}", left["k"], "paired")
    indexed = [{row["sample_id"]: row for row in part["samples"]} for part in (left, right)]
    if (left["k"] != right["k"] or any(set(part) != set(expected) for part in indexed)
            or any(len(part["samples"]) != len(expected) for part in (left, right))
            or any(row["fold"] != expected[sample] for part in indexed for sample, row in part.items())):
        raise ValueError("Paired ARI sample/fold/K coverage mismatch")
    samples = []
    for sample, fold in sorted(expected.items()):
        a, b = (part[sample] for part in indexed)
        included = a["mean"] is not None and b["mean"] is not None
        value = float(np.float32(a["mean"]) - np.float32(b["mean"])) if included else None
        if included and not np.isfinite(value):
            raise ValueError("Nonfinite paired ARI difference")
        samples.append({
            **identity, "sample_id": sample, "fold": fold, "included": included, "mean": value,
            "undefined_reason": None if included else json.dumps([
                {"condition": part["condition_id"], "reason": row["undefined_reason"]}
                for part, row in ((left, a), (right, b)) if row["mean"] is None
            ]),
        })
    common = [row["sample_id"] for row in samples if row["included"]]
    values = np.array([row["mean"] for row in samples if row["included"]], dtype=np.float32)
    summary = {
        **identity, "total_samples": len(expected), "common_samples": len(common),
        "mean": float(values.mean(dtype=np.float32)) if len(values) else None,
        "sample_sd": float(values.std(ddof=1, dtype=np.float32)) if len(values) > 1 else None,
        "repeat_sd": None, "repeat_sd_undefined_reason": "not_applicable_to_ARI_pairs",
        "mean_undefined_reason": None if len(values) else "no_common_samples",
        "sample_sd_undefined_reason": None if len(values) > 1 else "fewer_than_two_common_samples",
        "common_sample_ids": json.dumps(common),
        "excluded_sample_ids": json.dumps([row["sample_id"] for row in samples if not row["included"]]),
    }
    return summary, samples


def load_report_data(experiment: Path, snapshot: str) -> ReportData:
    """Read hash-bound snapshot summaries and clean-test occupancy metadata only."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", snapshot):
        raise ValueError("Invalid snapshot name")
    reader = SourceReader(experiment)
    prefix = f"results/oof/{snapshot}"
    if reader.path(f"{prefix}/failure.json").exists():
        raise ValueError("Failed or interrupted OOF snapshot")
    done = reader.read_json(f"{prefix}/completion.json")
    frame, manifest_hashes = _manifest(reader)
    expected = {row.sample_id: int(row.test_fold) for row in frame.itertuples(index=False)}
    if (done.get("status") != "oof_aggregation_completed" or done.get("checks_passed") is not True
            or done.get("conditions") != list(MAIN_CONDITIONS)
            or done.get("sample_count") != len(expected)
            or done.get("source_run_count") != len(MAIN_CONDITIONS) * len(FOLDS) * len(REPEATS)
            or done.get("score_record_count") != len(expected) * len(MAIN_CONDITIONS) * 3 * 7 * 10):
        raise ValueError("Expected a complete main-seven-condition OOF snapshot")

    def read(name: str) -> dict:
        return reader.read_json(f"{prefix}/{name}", done["artifact_sha256"][name])

    run = read("run.json")
    required = {"schema_version": 1, "mode": "complete_selected_conditions_oof",
                "snapshot": snapshot, "conditions": list(MAIN_CONDITIONS),
                "folds": list(FOLDS), "repeats": list(REPEATS),
                "cluster_counts": list(CLUSTER_COUNTS), "expected_test_folds": expected,
                "manifest_artifact_sha256": manifest_hashes}
    if any(run.get(key) != value for key, value in required.items()):
        raise ValueError("OOF snapshot identity/manifest mismatch")
    summaries, samples, availability, pairs = [], [], [], []
    parts, ari_parts = {}, {}
    for condition in MAIN_CONDITIONS:
        for k in CLUSTER_COUNTS:
            saved = read(f"summaries/{condition}/k{k}.json")
            if saved["condition_id"] != condition or saved["k"] != k:
                raise ValueError("OOF summary identity mismatch")
            for metric in REPEATED:
                summary = saved["metrics"][metric]
                if (summary["expression"], summary["metric"], summary["k"]) != (condition, metric, k):
                    raise ValueError("OOF metric identity mismatch")
                _collect_repeated(summary, "condition", summaries, samples, availability, expected)
                parts[condition, k, metric] = summary
            ari = read(f"ari/{condition}/k{k}.json")
            if (ari["condition_id"] != condition or ari["k"] != k
                    or ari["total_samples"] != len(expected)
                    or len(ari["samples"]) != len(expected)
                    or {item["sample_id"] for item in ari["samples"]} != set(expected)):
                raise ValueError("ARI identity/sample coverage mismatch")
            identity = _identity("ari", condition, k, "condition")
            ari_parts[condition, k] = ari
            summaries.append({
                **identity, "total_samples": ari["total_samples"],
                "common_samples": ari["defined_samples"], "mean": ari["mean"],
                "sample_sd": ari["sample_sd"], "repeat_sd": None,
                "repeat_sd_undefined_reason": "not_applicable_to_ARI_pairs",
                "mean_undefined_reason": ari["mean_undefined_reason"],
                "sample_sd_undefined_reason": ari["sample_sd_undefined_reason"],
                "common_sample_ids": json.dumps(ari["defined_sample_ids"]),
                "excluded_sample_ids": json.dumps(ari["undefined_sample_ids"]),
            })
            for item in ari["samples"]:
                if (item["fold"] != expected[item["sample_id"]]
                        or [p["repeats"] for p in item["pairs"]] != [[1, 2], [1, 3], [2, 3]]):
                    raise ValueError("ARI fold/pair mismatch")
                samples.append({**identity, "sample_id": item["sample_id"], "fold": item["fold"],
                                "included": item["mean"] is not None, "mean": item["mean"],
                                "undefined_reason": item["undefined_reason"]})
                pairs.extend({**identity, "sample_id": item["sample_id"], "fold": item["fold"],
                              "repeat_a": pair["repeats"][0], "repeat_b": pair["repeats"][1],
                              "value": pair["value"], "undefined_reason": pair["undefined_reason"],
                              "used_clusters": json.dumps(pair["used_clusters"]),
                              "degeneracy_flags": json.dumps(pair["degeneracy_flags"])}
                             for pair in item["pairs"])
    for left, right in PLANNED_PAIRS:
        for k in CLUSTER_COUNTS:
            saved = read(f"comparisons/{left}_minus_{right}/k{k}.json")
            if (saved["condition"], saved["reference"], saved["k"]) != (left, right, k):
                raise ValueError("Paired comparison identity mismatch")
            for metric in REPEATED:
                summary = saved["metrics"][metric]
                if (summary["expression"], summary["metric"], summary["k"]) != (
                    f"{left} - {right}", metric, k,
                ):
                    raise ValueError("Paired metric identity mismatch")
                _collect_repeated(summary, "paired", summaries, samples, availability, expected)
            summary, sample_rows = paired_ari(ari_parts[left, k], ari_parts[right, k], expected)
            summaries.append(summary)
            samples.extend(sample_rows)
    for k in CLUSTER_COUNTS:
        for metric in CONTRAST_METRICS:
            summary = corrected_interaction({c: parts[c, k, metric] for c in FACTORIAL_CONDITIONS}, expected)
            # asdict preserves tuples; use the same serialized contract as saved summaries.
            _collect_repeated(json.loads(json.dumps(summary)), "interaction",
                              summaries, samples, availability, expected)
    occupancy = _load_occupancy(reader, run, expected, frame, manifest_hashes)
    occupancy_summaries, occupancy_samples, occupancy_folds = _occupancy_tables(occupancy)
    summaries.extend(occupancy_summaries)
    samples.extend(occupancy_samples)
    reader.verify_unchanged()
    return ReportData(pd.DataFrame(summaries), pd.DataFrame(samples), pd.DataFrame(availability),
                      pd.DataFrame(pairs), occupancy, occupancy_folds, reader)


def _save(figure: plt.Figure, output: Path, stem: str, dpi: int) -> None:
    try:
        figure.savefig(output / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def _align_lla_axes(axes: tuple[Axes, ...], *, tick_step: float | None) -> None:
    """Share the complete LLA extent, including repeat curves and plot margins."""
    if not axes:
        return
    if tick_step is None:
        # Paired plots keep the first LLA panel's existing automatic tick interval.
        ticks = axes[0].get_yticks()
        tick_step = float(ticks[1] - ticks[0])
    lower = min(axis.get_ylim()[0] for axis in axes)
    upper = max(axis.get_ylim()[1] for axis in axes)
    limits = (np.floor(lower / tick_step) * tick_step, np.ceil(upper / tick_step) * tick_step)
    for axis in axes:
        axis.set_ylim(limits)
        axis.yaxis.set_major_locator(MultipleLocator(tick_step))


def _curve_figure(
    data: ReportData, output: Path, stem: str, metrics: tuple[str, ...], dpi: int,
    *, kind: str = "condition", expressions: tuple[str, ...] = MAIN_CONDITIONS,
    layout: tuple[int, int] | None = None,
) -> None:
    nrows, ncols = layout or (1, len(metrics))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows + 0.7), squeeze=False)
    for axis, metric in zip(axes.flat, metrics, strict=True):
        for index, expression in enumerate(expressions):
            rows = data.summaries[(data.summaries.kind == kind)
                                  & (data.summaries.source_metric == metric)
                                  & (data.summaries.expression == expression)].set_index("k")
            rows = rows.reindex(CLUSTER_COUNTS)
            color = COLORS.get(expression, tuple(COLORS.values())[index])
            if metric != "ari":
                for repeat, style in zip(REPEATS, STYLES, strict=True):
                    axis.plot(CLUSTER_COUNTS, rows[f"repeat_{repeat}_mean"], color=color,
                              linestyle=style, alpha=0.35, linewidth=0.8)
            axis.plot(CLUSTER_COUNTS, rows["mean"], color=color, marker="o", markersize=4,
                      linewidth=2, label=expression)
        if kind != "condition":
            axis.axhline(0, color="0.4", linewidth=0.8)
        axis.set(xlabel="K", ylabel=LABELS[metric] + (" difference" if kind != "condition" else ""),
                 xticks=CLUSTER_COUNTS)
        if kind == "condition":
            axis.yaxis.set_major_locator(MultipleLocator(0.1))
        axis.grid(alpha=0.2)
    _align_lla_axes(tuple(axis for axis, metric in zip(axes.flat, metrics, strict=True)
                          if metric in LLA_METRICS), tick_step=0.1 if kind == "condition" else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if any(metric != "ari" for metric in metrics):
        handles += [Line2D([], [], color="0.5", linestyle=style) for style in STYLES]
        labels += [f"Repeat {repeat}" for repeat in REPEATS]
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.16 if nrows == 1 else 0.10, 1, 1))
    _save(fig, output, stem, dpi)


def _dot_figure(data: ReportData, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 7.6), squeeze=False)
    for axis, metric in zip(axes.flat, SUMMARY_METRICS, strict=True):
        for index, expression in enumerate(MAIN_CONDITIONS):
            rows = data.samples[(data.samples.kind == "condition") & (data.samples.k == K0)
                                & (data.samples.source_metric == metric)
                                & (data.samples.expression == expression)].sort_values("sample_id")
            usable = rows[rows.included & rows["mean"].notna()]
            color = COLORS.get(expression, tuple(COLORS.values())[index])
            # Deterministic offsets expose overlap without introducing a random seed.
            offsets = np.linspace(-0.20, 0.20, len(usable))
            axis.scatter(index + offsets, usable["mean"], s=13, color=color, alpha=0.5)
            summary = data.summaries[(data.summaries.kind == "condition") & (data.summaries.k == K0)
                                     & (data.summaries.source_metric == metric)
                                     & (data.summaries.expression == expression)].iloc[0]
            axis.scatter([index], [summary["mean"]], color="black", marker="_", s=190, zorder=4)
            axis.text(index, 0.98, f"n={len(usable)}", transform=axis.get_xaxis_transform(),
                      ha="center", va="top", fontsize=8)
        axis.set_ylabel(LABELS[metric])
        axis.set_xticks(range(len(MAIN_CONDITIONS)), MAIN_CONDITIONS)
        axis.margins(y=0.16)
        axis.yaxis.set_major_locator(MultipleLocator(0.1))
        axis.grid(axis="y", alpha=0.2)
    _align_lla_axes(tuple(axes[0]), tick_step=0.1)
    fig.text(0.5, 0.005, "K = 8; dots: sample means; black bars: sample macro means; n: common samples",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save(fig, output, "02_k8_distributions", dpi)


def render_report(experiment: Path, snapshot: str, output: Path, *, dpi: int = 240) -> Path:
    """Replace generated report files and remove superseded figures in this directory."""
    experiment, output = experiment.resolve(), output.resolve()
    if (experiment.is_relative_to(output) or (output.is_relative_to(experiment)
            and not output.is_relative_to(experiment / "results/figures"))):
        raise ValueError("Output must be in results/figures or outside the source experiment")
    if type(dpi) is not int or not 100 <= dpi <= 600:
        raise ValueError("DPI must be an integer between 100 and 600")
    artifacts = (*[f"{name}.png" for name in FIGURE_NAMES],
                 *[f"{name}.csv" for name in TABLE_NAMES], "report.json")
    generated = (*artifacts, "completion.json", *[f"{name}.png" for name in OBSOLETE_FIGURES])
    for name in generated:
        target = output / name
        if target.resolve().parent != output or target.is_dir():
            raise ValueError(f"Generated output path is not a regular file in report directory: {target}")
    data = load_report_data(experiment, snapshot)
    output.mkdir(parents=True, exist_ok=True)
    # An interrupted overwrite must not retain a success record for stale files.
    (output / "completion.json").unlink(missing_ok=True)
    tables = {
        "metrics_all_k": data.summaries[data.summaries.kind == "condition"],
        "paired_all_k": data.summaries[data.summaries.kind == "paired"],
        "interaction_all_k": data.summaries[data.summaries.kind == "interaction"],
        "sample_values": data.samples, "availability": data.availability,
        "ari_pairs": data.ari_pairs, "occupancy_samples": data.occupancy,
        "occupancy_folds": data.occupancy_folds,
    }
    for stem in ("metrics", "paired", "interaction"):
        table = tables[f"{stem}_all_k"]
        tables[f"{stem}_k8"] = table[table.k == K0]
    for name, table in tables.items():
        if "source_metric" in table:
            table = table.assign(_order=table.source_metric.map({m: i for i, m in enumerate(DISPLAY_ORDER)}))
            table = table.sort_values(["_order", "expression", "k"], kind="stable").drop(columns="_order")
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        _curve_figure(data, output, "01_main_metrics_k_sweep", SUMMARY_METRICS, dpi, layout=(2, 3))
        _dot_figure(data, output, dpi)
        _curve_figure(data, output, "03_paired_k_sweep", SUMMARY_METRICS, dpi,
                      kind="paired", expressions=tuple(f"M11 - {c}" for c in ("B0", "B1", "M00")),
                      layout=(2, 3))
    data.reader.verify_unchanged()
    for name in OBSOLETE_FIGURES:
        (output / f"{name}.png").unlink(missing_ok=True)
    (output / "report.json").unlink(missing_ok=True)
    _write_json(output / "report.json", {
        "snapshot": snapshot, "experiment_dir": str(experiment), "conditions": list(MAIN_CONDITIONS),
        "K0": K0, "cluster_counts": list(CLUSTER_COUNTS), "repeats": list(REPEATS),
        "metric_order": ["LLA", "LFR(TGN+FS)", "ARI", "Cosine-Silhouette", "Cluster Occupancy"],
        "table_metric_order": [LABELS[metric] for metric in DISPLAY_ORDER],
        "figure_layouts": {
            "01_main_metrics_k_sweep": "2x3; top: LLA 3/5/9; bottom: LFR(TGN+FS), ARI, Cosine-Silhouette",
            "02_k8_distributions": "2x3; same metric order as the main summary",
            "03_paired_k_sweep": "2x3; top: LLA 3/5/9; bottom: LFR(TGN+FS), ARI, Cosine-Silhouette; "
            "each panel shows M11-B0, M11-B1, M11-M00",
            "Cluster Occupancy": "CSV tables only",
            "interaction": "CSV tables only",
        },
        "definitions": {
            "LLA": "adjusted_lla_3/5/9; occupancy-corrected; three separate windows; higher is better",
            "LFR(TGN+FS)": "lfr_both; TGN and Fractional Shift; figures and CSV",
            "LFR(TGN)": "lfr_noise; TGN only; CSV only",
            "LFR(FS)": "lfr_shift; Fractional Shift only; CSV only",
            "LFR averaging": "saved five-perturbation mean; lower is better; "
            "contrasts retain condition-reference",
            "ARI": "three repeat-pair mean within sample, then sample macro; no repeat SD",
            "paired ARI": "condition-reference difference of each sample's three-pair mean, "
            "then macro mean and sample SD over samples defined in both conditions; no repeat SD",
            "Cosine-Silhouette": "geometric diagnostic within each representation",
            "Cluster Occupancy": "clean test only; per-sample counts/fractions and per-fold distributions; "
            "cluster IDs are not pooled across folds/repeats; single_cluster mean is a fraction",
            "averaging": "equal sample weights; common samples across all three repeats and compared conditions; "
            "sample SD and repeat SD have ddof=1; neither is a confidence interval",
            "missing": "undefined remains blank in CSV; reasons and availability retained; no zero imputation",
            "interaction": "derived from saved adjusted LLA and all three LFR variants "
            "per-sample/repeat scores; "
            "intersection of all four conditions and all three repeats",
            "curves": "solid=macro mean, dashed/dotted/dash-dot=repeat 1/2/3; gaps=undefined; "
            "common sample membership may differ with K; see CSV",
            "k8_dots": "each dot is a common sample's repeat mean (ARI: three-pair mean); "
            "black bar=macro mean; n=defined/common sample count",
            "y_axes": "LLA windows share limits within each figure, including all points/repeat curves; "
            "main/distribution ticks=0.1 for all panels; "
            "paired LLA keeps the first panel's original tick interval; "
            "different metrics keep independent ranges",
        },
        "source_verification": "hashes of consumed snapshot and source metadata, manifest and coverage; "
        "does not rerun the full OOF check, spectra, maps, training, or inference",
        "sources": list(data.reader.ledger.values()),
        "code_sha256": {name: _digest(Path(__file__).with_name(name)) for name in
                        ("oof_reporting.py", "oof_sanity.py", "aggregation.py", "oof_pipeline.py", "config.py")},
    })
    _write_json(output / "completion.json", {
        "status": "oof_figures_completed", "snapshot": snapshot,
        "artifact_sha256": {name: _digest(output / name) for name in sorted(artifacts)},
    })
    return output
