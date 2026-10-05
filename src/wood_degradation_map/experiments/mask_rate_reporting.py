"""Export mask-rate sensitivity tables and an optional four-panel K-sweep figure.

Saved adjusted LLA, LFR, silhouette and ARI summaries retain their precision.
Occupancy is reconstructed from clean-test ARI contingency margins; spectra,
label maps and model weights are not needed.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from .aggregation import REPEATED_METRICS, aggregate_ari
from .config import CLUSTER_COUNTS, FOLDS, REPEATS
from .diagnostic_metrics import ARIPair, RepeatARI
from .manifests import _digest, _write_json
from .oof_reporting import (
    LFR_METRICS, LLA_METRICS, MAIN_FIGURE_METRICS, OCCUPANCY_METRICS, STYLES,
    _collect_repeated, _save,
)
from .oof_sanity import SourceReader, _manifest, occupancy_values

MASK_CONDITIONS = ("M11-25", "M11", "M11-75")
MASK_PERCENT = dict(zip(MASK_CONDITIONS, (25, 50, 75), strict=True))
K0 = 8
METRICS = (*LLA_METRICS, *LFR_METRICS, "ari", "silhouette", *OCCUPANCY_METRICS)
LABELS = {
    **{f"adjusted_lla_{window}": f"LLA-{window}" for window in (3, 5, 9)},
    "lfr_both": "LFR(TGN+FS)", "lfr_noise": "LFR(TGN)", "lfr_shift": "LFR(FS)",
    "ari": "ARI", "silhouette": "Cosine-Silhouette", "used_clusters": "Used clusters",
    "max_occupancy": "Maximum cluster occupancy",
    "single_cluster": "Single-cluster sample fraction",
}
STATISTICS = (
    "mean", "sample_sd", "repeat_sd", "repeat_1_mean", "repeat_2_mean", "repeat_3_mean",
)
COLUMNS = (
    "condition_id", "mask_percent", "k", "metric", "source_metric", "window",
    "total_samples", "common_samples", *STATISTICS, "mean_undefined_reason",
    "sample_sd_undefined_reason", "repeat_sd_undefined_reason", "common_sample_ids",
    "excluded_sample_ids", "availability", "ari_degenerate_sample_ids", "derivation",
    "source_json", "source_sha256",
)
K8_COLUMNS = (
    "metric", "source_metric", "window", "k",
    *(f"mask_{percent}_{field}" for percent in MASK_PERCENT.values()
      for field in ("mean", "sample_sd", "repeat_sd", "common_samples")),
)
ARTIFACTS = ("metrics_all_k.csv", "metrics_k8.csv", "report.json")
FIGURE_NAME = "01_mask_rate_metrics_k_sweep.png"
FIGURE_LABELS = {"M11-25": "M11-25", "M11": "M11(-50)", "M11-75": "M11-75"}
FIGURE_COLORS = dict(zip(MASK_CONDITIONS, ("#0072b2", "#009e73", "#d55e00"), strict=True))
Y_AXIS_REFERENCE = "main_oof_v1/01_main_metrics_k_sweep.png"
# Match the specified saved main figure, including LFR's original 5% margins
# around its mean/repeat extrema (0.004211644642055035, 0.6826949119567871).
# Autoscaling from the mask-only curves would produce a different vertical scale.
FIGURE_Y_LIMITS = {
    **{metric: (0.2, 0.9) for metric in LLA_METRICS},
    "lfr_both": (-0.029712518723681573, 0.7166190753225237),
}
CSVValue = str | int | float | None
MetricRow = dict[str, CSVValue]


@dataclass
class MaskRateData:
    rows: list[MetricRow]
    reader: SourceReader
    sample_count: int
    source_run_count: int


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _identity(condition: str, k: int, metric: str, source: str, digest: str) -> MetricRow:
    return {
        "condition_id": condition, "mask_percent": MASK_PERCENT[condition], "k": k,
        "metric": LABELS[metric], "source_metric": metric,
        "window": int(metric.rsplit("_", 1)[1]) if metric in LLA_METRICS else None,
        "source_json": source, "source_sha256": digest,
    }


def _repeated_row(summary: dict, expected: dict[str, int]) -> MetricRow:
    # Reuse the main report's sample/fold/repeat and common-set validation.
    rows: list[dict] = []
    _collect_repeated(summary, "condition", rows, [], [], expected)
    row = {key: value for key, value in rows[0].items() if key in COLUMNS}
    row.update(
        common_sample_ids=_json(summary["common_sample_ids"]),
        excluded_sample_ids=_json(summary["excluded_sample_ids"]),
        availability=_json(summary["availability"]), ari_degenerate_sample_ids="[]",
        derivation="saved_oof_summary",
    )
    return row


def _ari_rows(
    ari: dict, condition: str, k: int, expected: dict[str, int], pixels: dict[str, int],
) -> list[MetricRow]:
    samples = {sample["sample_id"]: sample for sample in ari["samples"]}
    defined = sorted(sample for sample, value in samples.items() if value["mean"] is not None)
    if (ari["condition_id"] != condition or ari["k"] != k
            or ari["total_samples"] != len(expected) or set(samples) != set(expected)
            or len(samples) != len(ari["samples"])
            or ari["defined_samples"] != len(defined) or ari["defined_sample_ids"] != defined
            or ari["undefined_sample_ids"] != sorted(set(expected) - set(defined))):
        raise ValueError(f"ARI identity/sample coverage mismatch: {condition}/k{k}")
    records = [RepeatARI(**{
        **sample, "pairs": tuple(ARIPair(**{
            **pair, "repeats": tuple(pair["repeats"]),
            "used_clusters": tuple(pair["used_clusters"]),
            "degeneracy_flags": tuple(pair["degeneracy_flags"]),
            "contingency": tuple(tuple(row) for row in pair["contingency"]),
        }) for pair in sample["pairs"]),
    }) for sample in ari["samples"]]
    checked = asdict(aggregate_ari(
        records, expected_test_folds=expected, condition_id=condition, k=k,
    ))
    if any(ari[field] != checked[field] for field in
           ("mean", "sample_sd", "mean_undefined_reason", "sample_sd_undefined_reason")):
        raise ValueError(f"ARI macro summary mismatch: {condition}/k{k}")
    counts_by_sample: dict[str, dict[int, list[int]]] = {}
    pairs = [list(pair) for pair in combinations(REPEATS, 2)]
    for sample_id, sample in samples.items():
        if (sample["condition_id"] != condition or sample["k"] != k
                or sample["fold"] != expected[sample_id]
                or sample["valid_pixels"] != pixels[sample_id]
                or [pair["repeats"] for pair in sample["pairs"]] != pairs):
            raise ValueError(f"ARI fold/pixel/repeat coverage mismatch: {sample_id}")
        counts_by_repeat: dict[int, list[int]] = {}
        for pair in sample["pairs"]:
            table = pair["contingency"]
            if (len(table) != k or any(len(row) != k for row in table)
                    or any(type(count) is not int or count < 0 for row in table for count in row)):
                raise ValueError(f"Invalid ARI contingency counts: {sample_id}/k{k}")
            margins = ([sum(row) for row in table],
                       [sum(row[column] for row in table) for column in range(k)])
            for side, repeat in enumerate(pair["repeats"]):
                counts = margins[side]
                values = occupancy_values(counts, pixels[sample_id], k)
                if (values["used_clusters"] != pair["used_clusters"][side]
                        or (repeat in counts_by_repeat and counts_by_repeat[repeat] != counts)):
                    raise ValueError(
                        f"Inconsistent ARI repeat margins: {sample_id}/repeat_{repeat}"
                    )
                counts_by_repeat[repeat] = counts
        if set(counts_by_repeat) != set(REPEATS):
            raise ValueError(f"Missing occupancy repeat: {sample_id}")
        counts_by_sample[sample_id] = counts_by_repeat
    rows: list[MetricRow] = [{
        "source_metric": "ari", "total_samples": len(expected), "common_samples": len(defined),
        "mean": ari["mean"], "sample_sd": ari["sample_sd"], "repeat_sd": None,
        "repeat_sd_undefined_reason": "not_applicable_to_ARI_pairs",
        "mean_undefined_reason": ari["mean_undefined_reason"],
        "sample_sd_undefined_reason": ari["sample_sd_undefined_reason"],
        "common_sample_ids": _json(defined),
        "excluded_sample_ids": _json(ari["undefined_sample_ids"]),
        "availability": _json([
            {"sample_id": sample["sample_id"], "reason": sample["undefined_reason"]}
            for sample in ari["samples"] if sample["undefined_reason"] is not None
        ]),
        "ari_degenerate_sample_ids": _json([
            sample["sample_id"] for sample in ari["samples"]
            if any(pair["degeneracy_flags"] for pair in sample["pairs"])
        ]), "derivation": "saved_oof_ari_summary",
    }]
    sample_ids = sorted(expected)
    for metric in OCCUPANCY_METRICS:
        matrix = np.array([
            [occupancy_values(counts_by_sample[sample][repeat], pixels[sample], k)[metric]
             for repeat in REPEATS] for sample in sample_ids
        ], dtype=np.float64)
        # Average within sample and across repeats; never pool pixels or cluster IDs.
        sample_means, repeat_means = matrix.mean(axis=1), matrix.mean(axis=0)
        rows.append({
            "source_metric": metric, "total_samples": len(expected),
            "common_samples": len(expected),
            "mean": float(repeat_means.mean()), "sample_sd": float(sample_means.std(ddof=1)),
            "repeat_sd": float(repeat_means.std(ddof=1)),
            **{f"repeat_{repeat}_mean": float(value)
               for repeat, value in zip(REPEATS, repeat_means, strict=True)},
            "common_sample_ids": _json(sample_ids), "excluded_sample_ids": "[]",
            "availability": _json([
                {"condition_id": condition, "repeat": repeat, "total_samples": len(expected),
                 "defined_samples": len(expected), "unavailable": []} for repeat in REPEATS
            ]), "ari_degenerate_sample_ids": "[]",
            "derivation": "clean_counts_from_saved_ari_contingency",
        })
    return rows


def load_mask_rate_data(experiment: Path, snapshot: str) -> MaskRateData:
    """Validate consumed snapshot/manifest metadata and prepare the 231 metric rows."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", snapshot):
        raise ValueError("Invalid snapshot name")
    reader = SourceReader(experiment)
    prefix = f"results/oof/{snapshot}"
    if reader.path(f"{prefix}/failure.json").exists():
        raise ValueError("Failed or interrupted OOF snapshot")
    done = reader.read_json(f"{prefix}/completion.json")
    frame, manifest_hashes = _manifest(reader)
    expected = {row.sample_id: int(row.test_fold) for row in frame.itertuples(index=False)}
    pixels = {row.sample_id: int(row.saved_pixel_count) for row in frame.itertuples(index=False)}
    conditions = sorted(MASK_CONDITIONS)
    run_count = len(MASK_CONDITIONS) * len(FOLDS) * len(REPEATS)
    score_count = len(expected) * len(MASK_CONDITIONS) * len(REPEATS)
    score_count *= len(CLUSTER_COUNTS) * len(REPEATED_METRICS)
    required_files = {"run.json", "scores.json"} | {
        f"{directory}/{condition}/k{k}.json" for directory in ("summaries", "ari")
        for condition in MASK_CONDITIONS for k in CLUSTER_COUNTS
    }
    if (done.get("status") != "oof_aggregation_completed" or done.get("checks_passed") is not True
            or done.get("conditions") != conditions or done.get("sample_count") != len(expected)
            or done.get("source_run_count") != run_count
            or done.get("score_record_count") != score_count
            or set(done.get("artifact_sha256", {})) != required_files):
        raise ValueError("Expected a complete M11-25/M11/M11-75 OOF snapshot")

    def read(name: str) -> dict:
        digest = done["artifact_sha256"][name]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid OOF source hash: {name}")
        return reader.read_json(f"{prefix}/{name}", digest)

    run = read("run.json")
    required = {
        "schema_version": 1, "mode": "complete_selected_conditions_oof", "snapshot": snapshot,
        "conditions": conditions, "folds": list(FOLDS), "repeats": list(REPEATS),
        "cluster_counts": list(CLUSTER_COUNTS), "expected_test_folds": expected,
        "manifest_artifact_sha256": manifest_hashes,
    }
    sources = run["sources"]
    source_keys = [(source["condition"], source["fold"], source["repeat"]) for source in sources]
    expected_keys = {(c, f, r) for c in MASK_CONDITIONS for f in FOLDS for r in REPEATS}
    if (any(run.get(key) != value for key, value in required.items())
            or len(source_keys) != run_count or set(source_keys) != expected_keys):
        raise ValueError("OOF snapshot identity/manifest/source coverage mismatch")
    rows: list[MetricRow] = []
    for condition in MASK_CONDITIONS:
        for k in CLUSTER_COUNTS:
            source = f"summaries/{condition}/k{k}.json"
            saved = read(source)
            if saved["condition_id"] != condition or saved["k"] != k:
                raise ValueError(f"OOF summary identity mismatch: {source}")
            for metric in (*LLA_METRICS, *LFR_METRICS, "silhouette"):
                summary = saved["metrics"][metric]
                if ((summary["expression"], summary["metric"], summary["k"])
                        != (condition, metric, k)):
                    raise ValueError(f"OOF metric identity mismatch: {source}/{metric}")
                rows.append({
                    **_repeated_row(summary, expected),
                    **_identity(condition, k, metric, source, done["artifact_sha256"][source]),
                })
            source = f"ari/{condition}/k{k}.json"
            for row in _ari_rows(read(source), condition, k, expected, pixels):
                metric = str(row["source_metric"])
                rows.append({**row, **_identity(condition, k, metric, source,
                                               done["artifact_sha256"][source])})
    for row in rows:
        for field in STATISTICS:
            value = row.get(field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(
                    f"Invalid {field}: {row['condition_id']}/k{row['k']}/{row['source_metric']}"
                )
    rows.sort(key=lambda row: (
        row["k"], METRICS.index(str(row["source_metric"])), row["mask_percent"],
    ))
    return MaskRateData(rows, reader, len(expected), run_count)


def _k8_rows(rows: list[MetricRow]) -> list[MetricRow]:
    indexed = {(row["source_metric"], row["condition_id"]): row for row in rows if row["k"] == K0}
    result = []
    for metric in METRICS:
        first = indexed[metric, MASK_CONDITIONS[0]]
        row = {key: first[key] for key in ("metric", "source_metric", "window", "k")}
        for condition, percent in MASK_PERCENT.items():
            source = indexed[metric, condition]
            row.update({f"mask_{percent}_{field}": source.get(field)
                        for field in ("mean", "sample_sd", "repeat_sd", "common_samples")})
        result.append(row)
    return result


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[MetricRow]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def _curve_values(rows: list[MetricRow], field: str) -> np.ndarray:
    # Undefined scores form gaps, rather than fabricated zero-valued observations.
    return np.array([np.nan if row.get(field) is None else row[field] for row in rows],
                    dtype=np.float64)


def _render_k_sweep(data: MaskRateData, output: Path, dpi: int) -> None:
    indexed = {(row["condition_id"], row["source_metric"], row["k"]): row for row in data.rows}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure, axes = plt.subplots(2, 2, figsize=(3.5, 4.1), squeeze=False)
        for axis, metric in zip(axes.flat, MAIN_FIGURE_METRICS, strict=True):
            for condition in MASK_CONDITIONS:
                rows = [indexed[condition, metric, k] for k in CLUSTER_COUNTS]
                color = FIGURE_COLORS[condition]
                for repeat, style in zip(REPEATS, STYLES, strict=True):
                    axis.plot(CLUSTER_COUNTS, _curve_values(rows, f"repeat_{repeat}_mean"),
                              color=color, linestyle=style, alpha=0.35, linewidth=0.5)
                axis.plot(CLUSTER_COUNTS, _curve_values(rows, "mean"), color=color,
                          marker="o", markersize=2, linewidth=1, label=FIGURE_LABELS[condition])
            axis.set(xlabel="K", ylabel=LABELS[metric], xticks=CLUSTER_COUNTS)
            axis.xaxis.label.set_size(9)
            axis.yaxis.label.set_size(9)
            axis.xaxis.labelpad = 2
            axis.yaxis.labelpad = 2
            axis.tick_params(axis="both", labelsize=8, pad=2, length=2.5)
            axis.set_ylim(FIGURE_Y_LIMITS[metric])
            axis.yaxis.set_major_locator(MultipleLocator(0.1))
            axis.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles += [Line2D([], [], color="0.5", linestyle=style) for style in STYLES]
        labels += [f"Repeat {repeat}" for repeat in REPEATS]
        figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5),
                      ncol=1, frameon=False, fontsize=8, handlelength=1.5, handletextpad=0.4)
        figure.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
        _save(figure, output, Path(FIGURE_NAME).stem, dpi)


def export_mask_rate_report(
    experiment: Path, snapshot: str, output: Path, *, overwrite: bool = False,
    with_figure: bool = False, dpi: int = 240,
) -> Path:
    """Write CSVs and provenance, optionally adding the 2x2 K-dependence PNG."""
    experiment, output = experiment.resolve(), output.resolve()
    if (experiment.is_relative_to(output) or (output.is_relative_to(experiment)
            and not output.is_relative_to(experiment / "results/figures"))):
        raise ValueError("Output must be in results/figures or outside the source experiment")
    if with_figure and (type(dpi) is not int or not 100 <= dpi <= 600):
        raise ValueError("DPI must be an integer between 100 and 600")
    artifacts = (*ARTIFACTS, FIGURE_NAME) if with_figure else ARTIFACTS
    for name in (*artifacts, "completion.json"):
        target = output / name
        if target.is_symlink() or target.is_dir() or target.resolve().parent != output:
            raise ValueError(f"Generated output must be a regular file: {target}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Report already exists; use --overwrite: {target}")
    data = load_mask_rate_data(experiment, snapshot)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".mask-rate-", dir=output.parent) as temporary:
        staging = Path(temporary)
        _write_csv(staging / "metrics_all_k.csv", COLUMNS, data.rows)
        _write_csv(staging / "metrics_k8.csv", K8_COLUMNS, _k8_rows(data.rows))
        if with_figure:
            _render_k_sweep(data, staging, dpi)
        _write_json(staging / "report.json", {
            "snapshot": snapshot, "conditions": list(MASK_CONDITIONS), "mask_percent": MASK_PERCENT,
            "cluster_counts": list(CLUSTER_COUNTS), "K0": K0, "repeats": list(REPEATS),
            "sample_count": data.sample_count, "source_run_count": data.source_run_count,
            "files": {"metrics_all_k.csv": {"rows": len(data.rows), "columns": len(COLUMNS)},
                      "metrics_k8.csv": {"rows": len(METRICS), "columns": len(K8_COLUMNS)}},
            **({"figure": {
                "file": FIGURE_NAME, "layout": "2x2; top: LLA-3, LLA-5; bottom: LLA-9, LFR(TGN+FS)",
                "source_metrics": list(MAIN_FIGURE_METRICS), "condition_labels": FIGURE_LABELS,
                "colors": FIGURE_COLORS, "dpi": dpi, "body_size_inches": [3.5, 4.1],
                "curves": "solid=OOF macro mean; dashed/dotted/dash-dot=learning repeats 1/2/3; "
                "undefined values form gaps",
                "y_axes": "fixed to the saved main K-sweep figure, including LFR margins; ticks=0.1",
                "y_limits": FIGURE_Y_LIMITS, "y_axis_reference": Y_AXIS_REFERENCE,
            }} if with_figure else {}),
            "definitions": {
                "LLA": "saved adjusted_lla_3/5/9; occupancy-corrected; separate windows",
                "mean": "equal sample macro weights; common samples across three learning repeats",
                "sample_sd": "SD of within-sample three-repeat means; ddof=1",
                "repeat_sd": "SD of three repeat macro means; ddof=1; blank for ARI",
                "ARI": "saved three repeat-pair mean within sample, then equal sample macro",
                "occupancy": "clean-test ARI contingency margins; "
                "repeat reconstructions must agree; "
                "cluster IDs are not pooled across folds/repeats",
                "missing": "blank numeric fields; sample IDs, availability and reasons "
                "in all-K table",
            },
            "source_verification": "consumed snapshot/manifest hashes "
            "and sample/fold/repeat coverage; "
            "does not rerun the full OOF check or evaluation",
            "sources": list(data.reader.ledger.values()),
            "code_sha256": {name: _digest(Path(__file__).with_name(name)) for name in (
                "mask_rate_reporting.py", "oof_reporting.py", "oof_sanity.py", "config.py",
                "aggregation.py", "diagnostic_metrics.py",
            )},
        })
        data.reader.verify_unchanged()
        _write_json(staging / "completion.json", {
            "status": "mask_rate_report_completed" if with_figure else "mask_rate_csv_completed",
            "checks_passed": True, "snapshot": snapshot,
            "conditions": list(MASK_CONDITIONS), "sample_count": data.sample_count,
            "source_run_count": data.source_run_count,
            "artifact_sha256": {name: _digest(staging / name) for name in artifacts},
        })
        # Invalidate old success records before the first replacement. An interrupted
        # publication leaves no completion record for a mixture of old and new files.
        (output / "completion.json").unlink(missing_ok=True)
        (output / "report.json").unlink(missing_ok=True)
        for name in (*artifacts, "completion.json"):
            (staging / name).replace(output / name)
    return output
