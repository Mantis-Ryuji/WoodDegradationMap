"""Integrate all four Table 5 paths into the two main tables and appendix."""

from __future__ import annotations

import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.stats import t

from .bacteria_config import CONDITIONS, RUN_SEEDS, TABLE4_CLASSES, bacteria_config
from .bacteria_data import PreparedBacteria
from .bacteria_evaluation import TABLE4_REPORTED, TABLE5_REPORTED, _read_score
from .bacteria_training import _code_hashes, _run_paths
from .bacteria_variant_training import ALL_VARIANTS, NEW_VARIANTS, variant_config, variant_paths
from .manifests import _digest, _read_json


SELECTION_ORDER = ("cls_unfrozen", "z_unfrozen", "cls_frozen", "z_frozen")


def _verified_score(
    data: PreparedBacteria, output_dir: Path, condition: str, variant: str, seed: int,
) -> dict[str, object]:
    if variant == "cls_unfrozen":
        results, weights = _run_paths(output_dir, "finetune", "reference30", condition, seed)
        score = _read_score(data, output_dir, "finetune", "reference30", condition, seed)
        mode, expected_config = "bacteria_id_finetune", bacteria_config()
        expected_code = _code_hashes()
    else:
        if variant not in NEW_VARIANTS:
            raise ValueError(f"Unknown Table 5 variant: {variant}")
        results, weights = variant_paths(output_dir, condition, variant, seed)
        path = results / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete planned Bacteria-ID run: {path}")
        score = _read_json(path)
        mode, expected_config = "bacteria_id_finetune_variant", variant_config(variant)
        from . import bacteria_variant_training
        expected_code = {**_code_hashes(), "bacteria_variant_training.py":
                         _digest(Path(bacteria_variant_training.__file__))}
    run = _read_json(results / "run.json")
    pretrain_results, pretrain_weights = _run_paths(
        output_dir, "pretrain", "reference30", condition, seed,
    )
    if (run.get("mode") != mode or run.get("condition") != condition
            or run.get("run_seed") != seed
            or (variant != "cls_unfrozen" and run.get("variant") != variant)
            or run.get("processed_manifest_sha256") != _digest(data.root / "manifest.json")
            or run.get("pretrain_run_sha256") != _digest(pretrain_results / "run.json")
            or run.get("pretrain_weights_sha256") != _digest(pretrain_weights / "last_model.pt")
            or run.get("contract", {}).get("config") != expected_config
            or run.get("contract", {}).get("code_sha256") != expected_code
            or score.get("status") != "completed"
            or score.get("condition") != condition or score.get("run_seed") != seed
            or score.get("processed_manifest_sha256") != _digest(data.root / "manifest.json")
            or score.get("test_count") != 3000
            or not 1 <= score.get("best_epoch", 0) <= 200
            or score.get("selection") != bacteria_config()["table5"]["finetune"]["checkpoint_selection"]
            or score.get("best_weights_sha256") != _digest(weights / "best_model.pt")):
        raise ValueError(f"Table 5 run provenance or score mismatch: {results}")
    accuracy = score.get("test_accuracy_percent")
    if (type(accuracy) not in (float, int) or not math.isfinite(accuracy)
            or not 0 <= accuracy <= 100 or not math.isclose(
                accuracy, 100 * score.get("test_accuracy", math.nan), abs_tol=1e-9,
            )):
        raise ValueError(f"Invalid Table 5 test Accuracy: {results}")
    return score


def select_variant(means_by_variant: dict[str, dict[str, float]]) -> str:
    """Choose one path by the unrounded mean of all ten held-out test scores."""
    if set(means_by_variant) != set(ALL_VARIANTS):
        raise ValueError("All four Table 5 variants are required")
    pooled: dict[str, float] = {}
    for variant in SELECTION_ORDER:
        values = means_by_variant[variant]
        if set(values) != set(CONDITIONS) or any(
            not math.isfinite(value) or not 0 <= value <= 100 for value in values.values()
        ):
            raise ValueError(f"Invalid condition means for {variant}")
        pooled[variant] = sum(values.values()) / len(CONDITIONS)
    return max(SELECTION_ORDER, key=pooled.__getitem__)


def _write_changed(path: Path, content: str) -> None:
    """Make repeated reporting a no-op while replacing only generated files."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) != 5 or attempt == 4:
                raise
            time.sleep(min(0.5 * 2**attempt, 4.0))


def render_integrated_tables(data: PreparedBacteria, output_dir: Path) -> tuple[Path, Path, Path]:
    """Require every run, then update the standard two tables plus full Table 5 appendix."""
    output_dir = output_dir.resolve()
    table4: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    table4_scores: dict[str, dict[str, list[list[float]]]] = {}
    table5: dict[str, dict[str, dict[str, object]]] = {}
    for condition in CONDITIONS:
        table4[condition] = {}
        table4_scores[condition] = {}
        for corpus in TABLE4_CLASSES:
            rows = [_read_score(data, output_dir, "table4", corpus, condition, seed)
                    for seed in RUN_SEEDS]
            values = np.array([[row["metrics_percent"][metric]
                                for metric in ("ACC", "NMI", "AMI")]
                               for row in rows], dtype=np.float64)
            if values.shape != (5, 3) or not np.isfinite(values).all():
                raise ValueError(f"Invalid Table 4 scores: {condition}, {corpus}")
            table4_scores[condition][corpus] = values.tolist()
            table4[condition][corpus] = {
                metric: {"mean": float(values[:, i].mean()),
                         "sample_sd": float(values[:, i].std(ddof=1))}
                for i, metric in enumerate(("ACC", "NMI", "AMI"))
            }
    for variant in ALL_VARIANTS:
        table5[variant] = {}
        for condition in CONDITIONS:
            rows = [_verified_score(data, output_dir, condition, variant, seed)
                    for seed in RUN_SEEDS]
            values = np.array([row["test_accuracy_percent"] for row in rows], dtype=np.float64)
            if values.shape != (5,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid Table 5 Accuracy: {variant}, {condition}")
            sample_sd = float(values.std(ddof=1))
            table5[variant][condition] = {
                "run_seeds": list(RUN_SEEDS), "accuracy_percent": values.tolist(),
                "mean_percent": float(values.mean()), "sample_sd_percent": sample_sd,
                "t_critical_df4": float(t.ppf(0.975, df=4)),
                "ci95_halfwidth_percent": float(t.ppf(0.975, df=4) * sample_sd / math.sqrt(5)),
            }
    selected = select_variant({variant: {condition: table5[variant][condition]["mean_percent"]
                                         for condition in CONDITIONS}
                               for variant in ALL_VARIANTS})
    pooled = {variant: sum(table5[variant][condition]["mean_percent"]
                           for condition in CONDITIONS) / len(CONDITIONS)
              for variant in ALL_VARIANTS}

    lines4 = [
        "# SMAE Table 4 + ChemoMAE (Bacteria-ID)", "",
        "| Method | Bacteria-4 ACC | NMI | AMI | Bacteria-6 ACC | NMI | AMI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in TABLE4_REPORTED:
        lines4.append("| " + name + " | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    for condition in CONDITIONS:
        values = [table4[condition][corpus][metric] for corpus in TABLE4_CLASSES
                  for metric in ("ACC", "NMI", "AMI")]
        lines4.append("| ChemoMAE(" + condition + ") | " + " | ".join(
            f"{value['mean']:.2f} ± {value['sample_sd']:.2f}" for value in values
        ) + " |")
    lines4 += [
        "", "Values are percentages. ChemoMAE ± is five-seed sample SD; the published "
        "SMAE Table 4 rows report point values only. Other-method rows are quoted from "
        "[Ren et al. (2025)](https://doi.org/10.1016/j.eswa.2025.128576). "
        "ChemoMAE uses SNV, 128-dimensional unit latents and Cosine-KMeans. "
        "Bacteria-4/6 IDs follow SMAE supplementary Fig. S3, although its Bacteria-6 "
        "composition differs from RamanCluster's supplement. The reference data are "
        "split 60/20/20 with seed 42; pretraining and CKmeans fit use train, scoring "
        "uses held-out test. Original split indices and preprocessing are unavailable, "
        "so the comparison is not an exact replication.", "",
    ]
    lines5 = [
        "# SMAE Table 5 + ChemoMAE (Bacteria-ID)", "",
        "| Method | Supervised learning Accuracy | w/o pretraining Accuracy | w/ pretraining Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, supervised, without, with_pretrain in TABLE5_REPORTED:
        lines5.append(f"| {name} | {supervised} | {without} | {with_pretrain} |")
    for condition in CONDITIONS:
        row = table5[selected][condition]
        lines5.append(f"| ChemoMAE({condition}) | — | — | "
                      f"{row['mean_percent']:.2f} ± {row['ci95_halfwidth_percent']:.1f} |")
    lines5 += [
        "", f"Selected classifier path: **{selected}**. One path was selected for both "
        "ChemoMAE conditions by the highest unrounded mean test Accuracy across "
        "M00/M11 × five seeds. All four paths and individual seeds appear in "
        "[the appendix](table5_variants_appendix.md). This test-based selection is "
        "exploratory and makes the displayed ChemoMAE Accuracy optimistically biased; "
        "the ± interval does not correct selection. Each run's checkpoint was selected "
        "by validation Accuracy before a single test evaluation. Accuracy and ± are "
        "percent and five-seed 95% CI half-width; ChemoMAE uses t(0.975,4) × sample "
        "SD / sqrt(5). Other-method rows quote [published SMAE Table 5]"
        "(https://doi.org/10.1016/j.eswa.2025.128576), not local reruns; their "
        "unpublished CI calculation, encoder updates and run starts cannot be "
        "confirmed. ChemoMAE uses SNV, its own reference30 pretraining and train-only "
        "TGN/FS finetuning; validation split, seed and checkpoint tie-break are "
        "documented local supplements. This is not a strict equal-settings comparison.", "",
    ]
    appendix = [
        "# Bacteria-ID Table 5: all ChemoMAE classifier paths", "",
        "All values are test Accuracy (%) from validation-selected checkpoints. "
        "Each path uses the same five pretraining seeds for M00/M11. "
        f"**Main-table selection: {selected}** by the highest pooled M00/M11 "
        "ten-run mean test Accuracy. This test-based selection is exploratory; the "
        "95% CIs below do not account for path selection.", "",
        "| Path | M00 mean ± 95% CI | M11 mean ± 95% CI | Pooled test mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for variant in SELECTION_ORDER:
        cells = [table5[variant][condition] for condition in CONDITIONS]
        appendix.append(f"| {variant} | " + " | ".join(
            f"{row['mean_percent']:.2f} ± {row['ci95_halfwidth_percent']:.1f}" for row in cells
        ) + f" | {pooled[variant]:.2f} |")
    appendix += ["", "| Path | Pretraining | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for variant in SELECTION_ORDER:
        for condition in CONDITIONS:
            scores = table5[variant][condition]["accuracy_percent"]
            appendix.append(f"| {variant} | {condition} | " +
                            " | ".join(f"{score:.2f}" for score in scores) + " |")
    appendix += ["", "CLS is the 256-dimensional token before `to_latent`; z is the "
                "unit-normalized 128-dimensional projection. The classifier is "
                "LayerNorm followed by Linear(30). Frozen paths update only the head; "
                "unfrozen paths also update the encoder. The training split, TGN/FS, "
                "optimizer, scheduler and early stopping are shared. `cls_unfrozen` "
                "uses the already completed original runs; the other paths reuse "
                "the corresponding pretraining checkpoint.", ""]

    details_path = output_dir / "table_statistics.json"
    if details_path.exists():
        previous = _read_json(details_path)
        if (previous.get("processed_manifest_sha256") != _digest(data.root / "manifest.json")
                or set(previous.get("table4", {})) != set(CONDITIONS)):
            raise ValueError("Existing table statistics do not match this prepared dataset")
    else:
        previous = {}
    details = {
        "schema_version": 1, "processed_manifest_sha256": _digest(data.root / "manifest.json"),
        "table4": {condition: {corpus: {
            "run_seeds": list(RUN_SEEDS),
            "mean_percent": {metric: table4[condition][corpus][metric]["mean"]
                             for metric in ("ACC", "NMI", "AMI")},
            "sample_sd_percent": {metric: table4[condition][corpus][metric]["sample_sd"]
                                  for metric in ("ACC", "NMI", "AMI")},
            "scores_percent": table4_scores[condition][corpus],
        } for corpus in TABLE4_CLASSES} for condition in CONDITIONS},
        "table5": {condition: table5[selected][condition] for condition in CONDITIONS},
        "table5_selected_variant": selected,
        "table5_selection": "highest pooled unrounded test Accuracy over both conditions and five seeds each; tie order cls_unfrozen, z_unfrozen, cls_frozen, z_frozen",
        "table5_pooled_test_mean_percent": pooled,
        "table5_variants": table5,
        "scipy": version("scipy"),
    }
    paths = (output_dir / "table4.md", output_dir / "table5.md",
             output_dir / "table5_variants_appendix.md")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, lines in zip(paths, (lines4, lines5, appendix), strict=True):
        _write_changed(path, "\n".join(lines))
    _write_changed(details_path, json.dumps(details, ensure_ascii=False, indent=2,
                                            allow_nan=False) + "\n")
    return paths
