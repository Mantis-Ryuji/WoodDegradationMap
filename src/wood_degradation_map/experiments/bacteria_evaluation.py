"""Bacteria-ID Cosine-KMeans evaluation and the two SMAE comparison tables."""

from __future__ import annotations

import math
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans
from chemomae.models.chemo_mae import ChemoMAE
from scipy.optimize import linear_sum_assignment
from scipy.stats import t
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from .bacteria_config import CONDITIONS, RUN_SEEDS, TABLE4_CLASSES, bacteria_config, purpose_seed
from .bacteria_data import CHANNELS, PreparedBacteria
from .bacteria_training import _code_hashes, _run_paths, _validate_run, load_pretrained
from .manifests import _digest, _read_json, _write_json
from .neural import fp32_inference
from .training import runtime_record


TABLE4_REPORTED: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("SimCLR", (65.30, 54.40, 54.40, 53.50, 47.60, 47.40)),
    ("SCCL", (71.00, 61.70, 61.60, 52.00, 46.40, 46.20)),
    ("CC", (71.60, 61.00, 60.80, 54.70, 53.50, 53.30)),
    ("TS-TCC", (75.00, 73.50, 73.40, 70.50, 65.70, 64.80)),
    ("RamanCluster", (77.00, 75.00, 74.60, 74.10, 73.00, 72.60)),
    ("SMAE", (83.80, 76.20, 76.10, 81.30, 76.80, 76.70)),
)
TABLE5_REPORTED: tuple[tuple[str, str, str, str], ...] = (
    ("ResNet", "83.40 ± 0.4", "76.50 ± 1.2", "—"),
    ("RamanNet", "85.40 ± 0.4", "77.40 ± 1.2", "—"),
    ("ConvMSANet", "84.80 ± 0.3", "76.80 ± 1.4", "—"),
    ("TCLP", "—", "—", "82.30 ± 0.5"),
    ("Jensen et al. (2024)", "—", "—", "81.60 ± 0.7"),
    ("SMAE", "85.40 ± 0.3", "77.80 ± 1.4", "83.90 ± 0.4"),
)


def extract_latents(
    model: ChemoMAE, spectra: np.ndarray, *, device: torch.device, batch_size: int = 512,
) -> np.ndarray:
    """Extract all-visible FP32 L2 latents, including the last partial batch."""
    if (spectra.ndim != 2 or spectra.shape[1] != CHANNELS or spectra.dtype != np.float32
            or len(spectra) == 0 or batch_size < 1):
        raise ValueError("Expected nonempty FP32 Bacteria-ID spectra and positive batch size")
    model.eval()
    output = np.empty((len(spectra), 128), dtype=np.float32)
    with fp32_inference(device):
        for start in range(0, len(spectra), batch_size):
            end = min(start + batch_size, len(spectra))
            chunk = torch.from_numpy(np.ascontiguousarray(spectra[start:end])).to(device)
            checked: list[bool] = []

            def check_projection(_module: torch.nn.Module, _inputs: tuple[object, ...],
                                 latent: torch.Tensor) -> None:
                norms = torch.linalg.vector_norm(latent, dim=1)
                if (not bool(torch.isfinite(latent).all()) or not bool(torch.isfinite(norms).all())
                        or not bool((norms >= 1e-12).all())):
                    raise ValueError(f"Invalid pre-normalization latent at rows {start}:{end}")
                checked.append(True)

            hook = model.encoder.to_latent.register_forward_hook(check_projection)
            try:
                latent = model.encoder(chunk, torch.ones_like(chunk, dtype=torch.bool))
            finally:
                hook.remove()
            if len(checked) != 1 or latent.shape != (end - start, 128):
                raise RuntimeError("Full-visible latent extraction did not use the expected projection")
            norms = torch.linalg.vector_norm(latent, dim=1)
            if (not bool(torch.isfinite(latent).all()) or not bool((norms > 0).all())
                    or not bool((norms - 1).abs().max() < 1e-5)):
                raise ValueError("Invalid 128-dimensional unit latents")
            output[start:end] = latent.cpu().numpy()
    return output


def cluster_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict[str, float]:
    """Compute whole-test Hungarian ACC, arithmetic NMI and max-normalized AMI."""
    if (y_true.ndim != 1 or y_pred.shape != y_true.shape or len(y_true) == 0
            or np.any(y_true < 0) or np.any(y_true >= k)
            or np.any(y_pred < 0) or np.any(y_pred >= k)):
        raise ValueError("Expected aligned 0..K-1 test labels and predictions")
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (y_true, y_pred), 1)
    true_rows, predicted_columns = linear_sum_assignment(counts, maximize=True)
    return {
        "ACC": 100.0 * float(counts[true_rows, predicted_columns].sum()) / len(y_true),
        "NMI": 100.0 * float(normalized_mutual_info_score(
            y_true, y_pred, average_method="arithmetic")),
        "AMI": 100.0 * float(adjusted_mutual_info_score(
            y_true, y_pred, average_method="max")),
    }


def evaluate_table4(
    data: PreparedBacteria, output_dir: Path, corpus: str, condition: str,
    run_seed: int, *, device: torch.device,
) -> dict[str, object]:
    """Fit one train-only ChemoMAE CKmeans and score fixed centers on test."""
    _validate_run(corpus, condition, run_seed)
    if corpus not in TABLE4_CLASSES or device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Table 4 requires Bacteria-4/6 and CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    results, weights = _run_paths(output_dir.resolve(), "table4", corpus, condition, run_seed)
    if results.exists() or weights.exists():
        raise FileExistsError("Table 4 evaluation already has output for this run")
    model = load_pretrained(output_dir, data, corpus, condition, run_seed, device)
    pretrain_results, pretrain_weights = _run_paths(
        output_dir.resolve(), "pretrain", corpus, condition, run_seed,
    )
    train_x, _ = data.rows("reference", f"{corpus}_train")
    test_x, test_labels = data.rows("reference", f"{corpus}_test")
    train_z = extract_latents(model, train_x, device=device)
    test_z = extract_latents(model, test_x, device=device)
    del model, train_x, test_x
    k = len(TABLE4_CLASSES[corpus])
    settings = bacteria_config()["table4"]["clustering"]
    kmeans = CosineKMeans(
        k, tol=settings["tol"], max_iter=settings["max_iter"],
        device=device, random_state=purpose_seed(corpus, run_seed, "kmeans"),
    ).to(dtype=torch.float32)
    started = time.perf_counter()
    with fp32_inference(device):
        kmeans.fit(torch.from_numpy(train_z))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fit_seconds = time.perf_counter() - started
        predicted = kmeans.predict(torch.from_numpy(test_z)).cpu().numpy().astype(np.int64)
    metrics = cluster_metrics(test_labels.astype(np.int64), predicted, k)
    results.mkdir(parents=True, exist_ok=False)
    weights.mkdir(parents=True, exist_ok=False)
    with (weights / "centroids.npz").open("xb") as destination:
        np.savez_compressed(destination, centroids=kmeans.centroids.detach().cpu().numpy())
    run: dict[str, object] = {
        "status": "completed", "corpus": corpus, "condition": condition,
        "run_seed": run_seed, "k": k, "train_count": len(train_z),
        "test_count": len(test_z), "dimension": train_z.shape[1],
        "fit_scope": "train only", "score_scope": "test only, fixed centers",
        "isolate_ids": list(TABLE4_CLASSES[corpus]),
        "seed": purpose_seed(corpus, run_seed, "kmeans"),
        "metrics_percent": metrics, "reference_inertia": float(kmeans.inertia_),
        "fit_seconds": fit_seconds,
        "processed_manifest_sha256": _digest(data.root / "manifest.json"),
        "split_sha256": _digest(data.root / "splits.npz"),
        "pretrain_run_sha256": _digest(pretrain_results / "run.json"),
        "pretrain_weights_sha256": _digest(pretrain_weights / "last_model.pt"),
        "centroids_sha256": _digest(weights / "centroids.npz"),
        "code_sha256": _code_hashes(), "runtime": runtime_record(device),
        "versions": {"chemomae": version("chemomae"), "torch": str(torch.__version__),
                     "numpy": version("numpy"), "scipy": version("scipy"),
                     "scikit_learn": version("scikit-learn")},
    }
    _write_json(results / "metrics.json", run)
    return run


def _read_score(
    data: PreparedBacteria, output_dir: Path, stage: str, corpus: str,
    condition: str, run_seed: int,
) -> dict[str, object]:
    results, _ = _run_paths(output_dir.resolve(), stage, corpus, condition, run_seed)
    path = results / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Incomplete planned Bacteria-ID run: {path}")
    score = _read_json(path)
    if (score.get("status") != "completed" or score.get("condition") != condition
            or score.get("run_seed") != run_seed
            or (stage == "table4" and score.get("corpus") != corpus)
            or score.get("processed_manifest_sha256") != _digest(data.root / "manifest.json")):
        raise ValueError(f"Score record does not match the fixed run: {path}")
    if stage == "table4":
        k = len(TABLE4_CLASSES[corpus])
        if (score.get("k") != k or score.get("dimension") != 128
                or score.get("train_count") != k * 1200
                or score.get("test_count") != k * 400
                or score.get("split_sha256") != _digest(data.root / "splits.npz")
                or set(score.get("metrics_percent", {})) != {"ACC", "NMI", "AMI"}):
            raise ValueError(f"Incomplete Table 4 scoring contract: {path}")
    elif (score.get("test_count") != 3000 or not 1 <= score.get("best_epoch", 0) <= 200
          or score.get("selection") != bacteria_config()["table5"]["finetune"]["checkpoint_selection"]):
        raise ValueError(f"Incomplete Table 5 scoring contract: {path}")
    return score


def render_tables(data: PreparedBacteria, output_dir: Path) -> tuple[Path, Path]:
    """Write exactly Table 4 and 5 Markdown after all 30 planned scores exist."""
    output_dir = output_dir.resolve()
    table4: dict[str, dict[str, dict[str, float]]] = {}
    table5: dict[str, tuple[float, float]] = {}
    table4_details: dict[str, dict[str, dict[str, object]]] = {}
    table5_details: dict[str, dict[str, object]] = {}
    details: dict[str, object] = {
        "schema_version": 1, "processed_manifest_sha256": _digest(data.root / "manifest.json"),
        "table4": table4_details, "table5": table5_details, "scipy": version("scipy"),
    }
    for condition in CONDITIONS:
        table4[condition] = {}
        for corpus in TABLE4_CLASSES:
            rows = [_read_score(data, output_dir, "table4", corpus, condition, seed)
                    for seed in RUN_SEEDS]
            values = np.array([[row["metrics_percent"][name] for name in ("ACC", "NMI", "AMI")]
                               for row in rows], dtype=np.float64)
            if values.shape != (5, 3) or not np.isfinite(values).all():
                raise ValueError(f"Invalid Table 4 scores: {corpus}, {condition}")
            table4[condition][corpus] = {
                name: float(values[:, index].mean())
                for index, name in enumerate(("ACC", "NMI", "AMI"))
            }
            table4_details.setdefault(condition, {})[corpus] = {
                "run_seeds": list(RUN_SEEDS), "scores_percent": values.tolist(),
                "mean_percent": table4[condition][corpus],
                "sample_sd_percent": {
                    name: float(values[:, index].std(ddof=1))
                    for index, name in enumerate(("ACC", "NMI", "AMI"))
                },
            }
        fine_rows = [_read_score(data, output_dir, "finetune", "reference30", condition, seed)
                     for seed in RUN_SEEDS]
        values = np.array([row["test_accuracy_percent"] for row in fine_rows], dtype=np.float64)
        if values.shape != (5,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
            raise ValueError(f"Invalid Table 5 accuracy scores: {condition}")
        table5[condition] = (float(values.mean()), float(t.ppf(0.975, df=4)
                                                       * values.std(ddof=1) / math.sqrt(5)))
        table5_details[condition] = {
            "run_seeds": list(RUN_SEEDS), "accuracy_percent": values.tolist(),
            "mean_percent": table5[condition][0],
            "sample_sd_percent": float(values.std(ddof=1)),
            "t_critical_df4": float(t.ppf(0.975, df=4)),
            "ci95_halfwidth_percent": table5[condition][1],
        }

    lines4 = [
        "# SMAE Table 4 + ChemoMAE (Bacteria-ID)", "",
        "| Method | Bacteria-4 ACC | NMI | AMI | Bacteria-6 ACC | NMI | AMI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, scores in TABLE4_REPORTED:
        lines4.append("| " + name + " | " + " | ".join(f"{value:.2f}" for value in scores) + " |")
    for condition in CONDITIONS:
        scores = [table4[condition][corpus][metric]
                  for corpus in TABLE4_CLASSES for metric in ("ACC", "NMI", "AMI")]
        lines4.append("| ChemoMAE(" + condition + ") | " +
                      " | ".join(f"{value:.2f}" for value in scores) + " |")
    lines4 += [
        "", "Values are percentages. Other-method rows are reported values from the "
        "[published SMAE Table 4](https://doi.org/10.1016/j.eswa.2025.128576); "
        "only ChemoMAE rows are new measurements. ChemoMAE uses SNV, 128-dimensional "
        "unit latents, ChemoMAE Cosine-KMeans, and the mean of five seeds. "
        "Bacteria-4/6 isolate IDs follow SMAE supplementary Fig. S3; the Bacteria-6 "
        "composition differs from RamanCluster's supplementary Fig. S2. "
        "The reference data are split 60/20/20 with seed 42; pretraining and CKmeans "
        "fit use train, and scoring uses held-out test. The original papers' exact split "
        "indices and preprocessing are unavailable, so the experimental conditions "
        "are not claimed identical.", "",
    ]
    lines5 = [
        "# SMAE Table 5 + ChemoMAE (Bacteria-ID)", "",
        "| Method | Supervised learning Accuracy | w/o pretraining Accuracy | w/ pretraining Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, supervised, without, with_pretrain in TABLE5_REPORTED:
        lines5.append(f"| {name} | {supervised} | {without} | {with_pretrain} |")
    for condition in CONDITIONS:
        mean, halfwidth = table5[condition]
        lines5.append(f"| ChemoMAE({condition}) | — | — | {mean:.2f} ± {halfwidth:.1f} |")
    lines5 += [
        "", "Accuracy is in percent; ± denotes a 95% confidence-interval half-width. "
        "Other-method rows are reported values from the "
        "[published SMAE Table 5](https://doi.org/10.1016/j.eswa.2025.128576); "
        "only ChemoMAE rows are new measurements. TCLP and Jensen et al. (2024) "
        "are the SMAE paper's encoder-replaced comparisons. ChemoMAE uses five "
        "independent pretrain-to-finetune runs, SNV, train-only TGN/FS, and full-encoder "
        "CLS finetuning. The ChemoMAE interval uses t(0.975, 4) times sample SD / sqrt(5); "
        "SMAE's unpublished interval calculation is not asserted to match. "
        "The validation-best checkpoint (earliest on tie), run seeds and finetune "
        "split are documented local supplements to unpublished settings.", "",
    ]
    paths = output_dir / "table4.md", output_dir / "table5.md"
    statistics_path = output_dir / "table_statistics.json"
    if any(path.exists() for path in (*paths, statistics_path)):
        raise FileExistsError("Comparison tables already exist; review before replacing")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, lines in zip(paths, (lines4, lines5), strict=True):
        with path.open("x", encoding="utf-8") as destination:
            destination.write("\n".join(lines))
    _write_json(statistics_path, details)
    return paths
