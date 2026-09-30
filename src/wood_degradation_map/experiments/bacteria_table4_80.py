"""Isolated 80/20 Bacteria-ID Table 4 experiment using the saved legacy test rows."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import time
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans
from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.trainer import Trainer, TrainerConfig

from .bacteria_config import CONDITIONS, RUN_SEEDS, TABLE4_CLASSES, bacteria_config, purpose_seed
from .bacteria_data import PreparedBacteria
from .bacteria_evaluation import TABLE4_REPORTED, cluster_metrics, extract_latents
from .bacteria_training import (
    BacteriaRandomness, BacteriaTrainingData, _run_paths, _validate_run,
    build_bacteria_model,
)
from .manifests import _digest, _read_json, _write_json
from .neural import build_optimizer, fp32_inference
from .records import make_run_record
from .training import ExperimentTrainer, runtime_record


PROTOCOL = "bacteria_id_table4_80_20_v1"
METRICS = ("ACC", "NMI", "AMI")
PURPOSES = ("model_init", "train_order", "mask", "pretrain_augmentation", "kmeans")
# The first pretraining run finished before the evaluation metadata path was fixed.
# Its recorded module hash remains valid because no pretraining logic changed.
_PRE_SCORE_FIX_SOURCE_SHA256 = "38ec7174b2950315c740ab9936b6aa50f8b972d30bc7d818ae41a065040592fe"


def table4_config() -> dict[str, object]:
    """Snapshot only the Table 4 settings; leave the Table 5 contract untouched."""
    original = bacteria_config()
    recipe = original["pretrain"]
    clustering = original["table4"]["clustering"]
    if (recipe["epochs"] != 800 or recipe["batch_size"] != 1024
            or recipe["drop_last"] is not True or recipe["early_stopping"] is not False
            or original["model"]["latent_dim"] != 128
            or clustering["initializations"] != 1):
        raise ValueError("Shared Bacteria-ID recipe no longer matches fixed 80/20 Table 4")
    return {
        "protocol": PROTOCOL,
        "model": original["model"],
        "pretrain": original["pretrain"],
        "augmentation": original["augmentation"],
        "preprocessing": original["preprocessing"],
        "table4": {
            "isolate_ids": original["table4"]["isolate_ids"],
            "split": [0.8, 0.2],
            "split_source": "saved legacy train union validation; saved legacy test unchanged",
            "validation": None,
            "checkpoint": "raw weights after fixed epoch 800",
            "clustering": original["table4"]["clustering"],
            "metrics": original["table4"]["metrics"],
        },
        "run_seeds": list(RUN_SEEDS),
    }


def _index_digest(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def derive_table4_splits(
    labels: np.ndarray, old_splits: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate the saved three-way splits and construct the exact 80/20 rows."""
    if labels.ndim != 1:
        raise ValueError("Reference labels must be one-dimensional")
    result: dict[str, np.ndarray] = {}
    for corpus, ids in TABLE4_CLASSES.items():
        parts: list[np.ndarray] = []
        for name, per_class in (("train", 1200), ("validation", 400), ("test", 400)):
            key = f"{corpus}_{name}"
            if key not in old_splits:
                raise ValueError(f"Saved legacy split is missing {key}")
            rows = old_splits[key]
            if (rows.ndim != 1 or rows.dtype.kind not in "iu"
                    or len(rows) != len(ids) * per_class or np.any(rows < 0)
                    or np.any(rows >= len(labels)) or len(np.unique(rows)) != len(rows)):
                raise ValueError(f"Invalid saved legacy source rows: {key}")
            if any(np.count_nonzero(labels[rows] == label) != per_class for label in ids):
                raise ValueError(f"Unexpected per-class count in saved legacy split: {key}")
            parts.append(np.asarray(rows, dtype=np.int64))
        combined = np.concatenate(parts)
        target = np.flatnonzero(np.isin(labels, ids))
        if len(np.unique(combined)) != len(target) or not np.array_equal(np.sort(combined), target):
            raise ValueError(f"Legacy {corpus} split does not partition every target source row")
        train = np.sort(np.concatenate(parts[:2]))
        test = parts[2].copy()  # Preserve the saved test order, not only its set of rows.
        if (len(train) != len(ids) * 1600 or len(test) != len(ids) * 400
                or np.intersect1d(train, test).size
                or not np.array_equal(np.sort(np.concatenate((train, test))), target)):
            raise ValueError(f"New {corpus} 80/20 split fails coverage or non-overlap")
        result[f"{corpus}_train"] = train
        result[f"{corpus}_test"] = test
    return result


def _manifest_fields(base: PreparedBacteria, splits: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "prepared",
        "protocol": PROTOCOL,
        "config": table4_config(),
        "source_processed_dir": str(base.root.resolve()),
        "source_processed_manifest_sha256": _digest(base.root / "manifest.json"),
        "source_legacy_splits_sha256": _digest(base.root / "splits.npz"),
        "split_file": "splits.npz",
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "per_class_counts": {
            corpus: {
                str(label): {
                    split: int(np.count_nonzero(
                        base.labels["reference"][splits[f"{corpus}_{split}"]] == label
                    ))
                    for split in ("train", "test")
                }
                for label in ids
            }
            for corpus, ids in TABLE4_CLASSES.items()
        },
        "test_source_index_sha256": {
            corpus: _index_digest(base.splits[f"{corpus}_test"])
            for corpus in TABLE4_CLASSES
        },
        "seeds": {
            corpus: {
                str(seed): {purpose: purpose_seed(corpus, seed, purpose) for purpose in PURPOSES}
                for seed in RUN_SEEDS
            }
            for corpus in TABLE4_CLASSES
        },
        "planned_runs": [
            {"corpus": corpus, "condition": condition, "run_seed": seed}
            for corpus in TABLE4_CLASSES for condition in CONDITIONS for seed in RUN_SEEDS
        ],
    }


def prepare_table4_80(processed_dir: Path, output_dir: Path) -> dict[str, object]:
    """Write a new manifest and splits without editing the legacy prepared inputs."""
    base = PreparedBacteria(processed_dir)
    splits = derive_table4_splits(base.labels["reference"], base.splits)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Table 4 80/20 output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "splits.npz", **splits)
    manifest = {**_manifest_fields(base, splits),
                "split_file_sha256": _digest(output_dir / "splits.npz")}
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


class Table4Data:
    """Verified view of the legacy SNV arrays with the new persisted split."""

    def __init__(self, processed_dir: Path, output_dir: Path) -> None:
        self.base = PreparedBacteria(processed_dir)
        self.root = output_dir.resolve()
        manifest = _read_json(self.root / "manifest.json")
        split_path = self.root / "splits.npz"
        with np.load(split_path, allow_pickle=False) as saved:
            self.splits = {name: saved[name].copy() for name in saved.files}
        expected = derive_table4_splits(self.base.labels["reference"], self.base.splits)
        if (set(self.splits) != set(expected)
                or any(not np.array_equal(self.splits[name], rows)
                       for name, rows in expected.items())):
            raise ValueError("80/20 source rows differ from legacy train, validation or test")
        if manifest != {**_manifest_fields(self.base, expected),
                        "split_file_sha256": _digest(split_path)}:
            raise ValueError(
                "Table 4 80/20 manifest or source inputs differ from the fixed protocol"
            )
        self.manifest = manifest

    def rows(self, corpus: str, split: str) -> tuple[np.ndarray, np.ndarray]:
        if corpus not in TABLE4_CLASSES or split not in ("train", "test"):
            raise ValueError("Only Bacteria-4/6 80/20 train and test are available")
        indices = self.splits[f"{corpus}_{split}"]
        return (np.array(self.base.spectra["reference"][indices], dtype=np.float32, copy=True),
                np.array(self.base.labels["reference"][indices], dtype=np.int64, copy=True))


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    names = (
        "bacteria_table4_80.py", "bacteria_training.py", "bacteria_evaluation.py",
        "bacteria_config.py", "bacteria_data.py", "config.py", "neural.py",
        "training.py", "manifests.py", "records.py",
    )
    return {name: _digest(root / name) for name in names}


class Table4Pretrainer(ExperimentTrainer):
    """Fixed 800-epoch pretraining from new initialization on 80% train rows."""

    def __init__(
        self, data: Table4Data, output_dir: Path, corpus: str, condition: str,
        run_seed: int, *, device: torch.device, resume_from: Path | None = None,
    ) -> None:
        _validate_run(corpus, condition, run_seed)
        if corpus not in TABLE4_CLASSES or device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Table 4 80/20 pretraining requires Bacteria-4/6 and CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        values, _ = data.rows(corpus, "train")
        train = BacteriaTrainingData(corpus, torch.from_numpy(values), len(values))
        self.recipe = table4_config()["pretrain"]
        self.steps_per_epoch = len(values) // self.recipe["batch_size"]
        if self.steps_per_epoch < 1:
            raise ValueError("80/20 train selection is smaller than the fixed batch size")
        self.target_epochs = self.recipe["epochs"]
        self.batches_per_epoch = self.steps_per_epoch
        self.is_smoke = False
        self.randomness = BacteriaRandomness(corpus, condition, run_seed, device)
        self.train = train
        self.completed_epochs = self.attempted_updates = self.optimizer_updates = 0
        self.nonzero_lr_updates = 0
        self._epoch_in_progress = self._failed = self._fit_called = False
        self._epoch_stats: dict[str, object] = {}
        self.trace: list[dict[str, object]] = []
        self.results_dir, weights_dir = _run_paths(
            output_dir.resolve(), "pretrain", corpus, condition, run_seed,
        )
        checkpoint_dir = weights_dir / "checkpoints"
        self.run_record = make_run_record({
            "schema_version": 2, "mode": PROTOCOL + "_pretrain", "corpus": corpus,
            "condition": condition, "run_seed": run_seed,
            "train_spectra": len(values), "batch_size": self.recipe["batch_size"],
            "steps_per_epoch": self.steps_per_epoch,
            "planned_updates": self.steps_per_epoch * self.target_epochs,
            "processed_manifest_sha256": _digest(data.root / "manifest.json"),
            "split_sha256": _digest(data.root / "splits.npz"),
            "source_split": f"{corpus}_train",
            "seeds": {purpose: purpose_seed(corpus, run_seed, purpose) for purpose in
                      PURPOSES if purpose != "kmeans"},
            "config": table4_config(), "runtime": runtime_record(device),
            "code_sha256": _code_hashes(),
            "resume_boundary": "completed epoch; interrupted epoch is replayed",
        })
        if resume_from is None:
            if self.results_dir.exists() or weights_dir.exists():
                raise FileExistsError("80/20 run exists; resume only from this run's last.pt")
        else:
            resume_from = resume_from.resolve()
            if resume_from != (checkpoint_dir / "last.pt").resolve() or not resume_from.is_file():
                raise ValueError("Resume file must be this 80/20 run's last.pt")
            if _read_json(self.results_dir / "run.json") != self.run_record:
                raise ValueError("80/20 run/config/data/code/runtime differs; resume rejected")
        model = build_bacteria_model(corpus, run_seed).to(device)
        optimizer = build_optimizer(model)
        cfg = TrainerConfig(
            out_dir=weights_dir, device=str(device), amp=True, amp_dtype="fp16",
            enable_tf32=False, grad_clip=None, use_ema=False, loss_type="mse",
            loss_region="masked", reduction="mean", resume_from=resume_from,
        )
        Trainer.__init__(self, model, optimizer, (), scheduler=None,
                         augmenter=self.randomness.augmenter, cfg=cfg)
        self.history_path = self.results_dir / "training_history.json"
        self.history = []
        if resume_from is None:
            self.results_dir.mkdir(parents=True, exist_ok=False)
            _write_json(self.results_dir / "run.json", self.run_record)


def verify_pretraining(
    data: Table4Data, output_dir: Path, corpus: str, condition: str, run_seed: int,
) -> tuple[dict[str, object], dict[str, object], Path]:
    """Reject missing, legacy, incomplete or differently split pretraining weights."""
    _validate_run(corpus, condition, run_seed)
    if corpus not in TABLE4_CLASSES:
        raise ValueError("Table 4 uses only Bacteria-4/6")
    results, weights = _run_paths(output_dir.resolve(), "pretrain", corpus, condition, run_seed)
    run = _read_json(results / "run.json")
    done = _read_json(results / "completion.json")
    expected_train = len(data.splits[f"{corpus}_train"])
    batch_size = table4_config()["pretrain"]["batch_size"]
    batches = expected_train // batch_size
    attempts = batches * 800
    path = weights / "last_model.pt"
    expected_seeds = {purpose: purpose_seed(corpus, run_seed, purpose) for purpose in
                      PURPOSES if purpose != "kmeans"}
    if (run.get("mode") != PROTOCOL + "_pretrain" or run.get("corpus") != corpus
            or run.get("condition") != condition or run.get("run_seed") != run_seed
            or run.get("train_spectra") != expected_train or run.get("batch_size") != batch_size
            or run.get("steps_per_epoch") != batches or run.get("planned_updates") != attempts
            or run.get("source_split") != f"{corpus}_train"
            or run.get("seeds") != expected_seeds
            or run.get("processed_manifest_sha256") != _digest(data.root / "manifest.json")
            or run.get("split_sha256") != _digest(data.root / "splits.npz")
            or run.get("contract", {}).get("config") != table4_config()
            or done.get("status") != "training_completed" or done.get("completed_epochs") != 800
            or done.get("attempted_updates") != attempts
            or type(done.get("optimizer_updates")) is not int
            or not 0 <= done["optimizer_updates"] <= attempts
            or done.get("amp_skips") != attempts - done["optimizer_updates"]
            or done.get("weights_sha256") != _digest(path)):
        raise ValueError(f"Incomplete or incompatible 80/20 pretraining: {results}")
    return run, done, path


def load_pretrained(
    data: Table4Data, output_dir: Path, corpus: str, condition: str,
    run_seed: int, device: torch.device,
) -> ChemoMAE:
    _, _, path = verify_pretraining(data, output_dir, corpus, condition, run_seed)
    model = build_bacteria_model(corpus, run_seed)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device)


def discard_incomplete_score(results: Path, weights: Path) -> bool:
    """Remove only the centroids-only output left by the score metadata failure."""
    centroids = weights / "centroids.npz"
    if (results.is_symlink() or weights.is_symlink() or centroids.is_symlink()
            or not results.is_dir() or not weights.is_dir()
            or not centroids.is_file() or any(results.iterdir())):
        return False
    if list(weights.iterdir()) != [centroids]:
        return False
    centroids.unlink()
    weights.rmdir()
    results.rmdir()
    return True


def evaluate_table4_80(
    data: Table4Data, output_dir: Path, corpus: str, condition: str,
    run_seed: int, *, device: torch.device,
) -> dict[str, object]:
    """Fit all 80% train latents and assign the unchanged 20% test with fixed centers."""
    _validate_run(corpus, condition, run_seed)
    if corpus not in TABLE4_CLASSES or device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Table 4 80/20 clustering requires Bacteria-4/6 and CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    output_dir = output_dir.resolve()
    results, weights = _run_paths(output_dir, "table4", corpus, condition, run_seed)
    if results.exists() or weights.exists():
        raise FileExistsError("80/20 clustering output already exists for this run")
    _, _, pretrain_path = verify_pretraining(
        data, output_dir, corpus, condition, run_seed,
    )
    pretrain_results, _ = _run_paths(output_dir, "pretrain", corpus, condition, run_seed)
    pretrain_run_sha256 = _digest(pretrain_results / "run.json")
    model = load_pretrained(data, output_dir, corpus, condition, run_seed, device)
    train_x, _ = data.rows(corpus, "train")
    test_x, test_labels = data.rows(corpus, "test")
    train_z = extract_latents(model, train_x, device=device)
    test_z = extract_latents(model, test_x, device=device)
    del model, train_x, test_x
    k = len(TABLE4_CLASSES[corpus])
    settings = table4_config()["table4"]["clustering"]
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
    metrics = cluster_metrics(test_labels, predicted, k)
    results.mkdir(parents=True, exist_ok=False)
    weights.mkdir(parents=True, exist_ok=False)
    with (weights / "centroids.npz").open("xb") as destination:
        np.savez_compressed(destination, centroids=kmeans.centroids.detach().cpu().numpy())
    record: dict[str, object] = {
        "status": "completed", "protocol": PROTOCOL, "corpus": corpus,
        "condition": condition, "run_seed": run_seed, "k": k,
        "train_count": len(train_z), "test_count": len(test_z), "dimension": train_z.shape[1],
        "fit_scope": "all new 80% train latents", "score_scope": "saved legacy test; fixed centers",
        "isolate_ids": list(TABLE4_CLASSES[corpus]),
        "clustering_settings": settings,
        "kmeans_seed": purpose_seed(corpus, run_seed, "kmeans"),
        "metrics_percent": metrics, "reference_inertia": float(kmeans.inertia_),
        "fit_seconds": fit_seconds,
        "processed_manifest_sha256": _digest(data.root / "manifest.json"),
        "split_sha256": _digest(data.root / "splits.npz"),
        "test_source_index_sha256": _index_digest(data.splits[f"{corpus}_test"]),
        "pretrain_run_sha256": pretrain_run_sha256,
        "pretrain_weights_file": str(pretrain_path),
        "pretrain_weights_sha256": _digest(pretrain_path),
        "centroids_sha256": _digest(weights / "centroids.npz"),
        "code_sha256": _code_hashes(), "runtime": runtime_record(device),
        "versions": {"chemomae": version("chemomae"), "torch": str(torch.__version__),
                     "numpy": version("numpy"), "scipy": version("scipy"),
                     "scikit_learn": version("scikit-learn")},
    }
    _write_json(results / "metrics.json", record)
    return record


def verify_score(
    data: Table4Data, output_dir: Path, corpus: str, condition: str, run_seed: int,
) -> dict[str, object]:
    """Check every score and its final checkpoint before using it in the main table."""
    _, done, pretrain_path = verify_pretraining(data, output_dir, corpus, condition, run_seed)
    pretrain_results, _ = _run_paths(output_dir.resolve(), "pretrain", corpus, condition, run_seed)
    results, weights = _run_paths(output_dir.resolve(), "table4", corpus, condition, run_seed)
    score = _read_json(results / "metrics.json")
    metrics = score.get("metrics_percent", {})
    if (score.get("status") != "completed" or score.get("protocol") != PROTOCOL
            or score.get("corpus") != corpus or score.get("condition") != condition
            or score.get("run_seed") != run_seed or score.get("k") != len(TABLE4_CLASSES[corpus])
            or score.get("train_count") != len(data.splits[f"{corpus}_train"])
            or score.get("test_count") != len(data.splits[f"{corpus}_test"])
            or score.get("dimension") != 128
            or score.get("fit_scope") != "all new 80% train latents"
            or score.get("score_scope") != "saved legacy test; fixed centers"
            or score.get("isolate_ids") != list(TABLE4_CLASSES[corpus])
            or score.get("clustering_settings") != table4_config()["table4"]["clustering"]
            or score.get("kmeans_seed") != purpose_seed(corpus, run_seed, "kmeans")
            or score.get("processed_manifest_sha256") != _digest(data.root / "manifest.json")
            or score.get("split_sha256") != _digest(data.root / "splits.npz")
            or score.get("test_source_index_sha256") != _index_digest(data.splits[f"{corpus}_test"])
            or score.get("pretrain_run_sha256") != _digest(pretrain_results / "run.json")
            or score.get("pretrain_weights_file") != str(pretrain_path)
            or score.get("pretrain_weights_sha256") != done["weights_sha256"]
            or score.get("centroids_sha256") != _digest(weights / "centroids.npz")
            or not isinstance(score.get("code_sha256"), dict)
            or set(metrics) != set(METRICS)
            or any(not isinstance(metrics[name], (int, float)) or not math.isfinite(metrics[name])
                   or metrics[name] > 100 for name in METRICS)
            or not 0 <= metrics["ACC"] <= 100 or not 0 <= metrics["NMI"] <= 100
            or not -100 <= metrics["AMI"] <= 100):
        raise ValueError(f"Invalid or incompatible 80/20 Table 4 score: {results}")
    return score


def summarize_scores(values: np.ndarray) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation for all five prespecified seeds."""
    if values.shape != (5, 3) or not np.isfinite(values).all():
        raise ValueError("Expected five finite ACC/NMI/AMI triples")
    return {
        name: {"mean": float(values[:, index].mean()),
               "sample_sd": float(values[:, index].std(ddof=1))}
        for index, name in enumerate(METRICS)
    }


def _score_figure(values: Mapping[tuple[str, str], np.ndarray]) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3, sharey=True)
    for row, corpus in enumerate(TABLE4_CLASSES):
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            for x, condition in enumerate(CONDITIONS):
                scores = values[(corpus, condition)][:, column]
                axis.scatter(x + np.linspace(-0.10, 0.10, len(scores)), scores,
                             color=("#386cb0", "#e6550d")[x], s=18, alpha=0.8)
                axis.errorbar(x, scores.mean(), yerr=scores.std(ddof=1), fmt="o",
                              color="black", capsize=4, markersize=5)
            axis.set_xticks((0, 1), CONDITIONS)
            axis.set_title(f"{corpus} · {metric}")
            axis.grid(axis="y", alpha=0.25)
            if column == 0:
                axis.set_ylabel("Test score (%)")
    figure.suptitle("Table 4 · 80% train / 20% held-out test · five seeds")
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=160)
    return output.getvalue()


def render_table4_80(data: Table4Data, output_dir: Path) -> tuple[Path, ...]:
    """Require all 20 runs, then write the primary Table 4 and its audit outputs."""
    output_dir = output_dir.resolve()
    scores_by_condition: dict[tuple[str, str], np.ndarray] = {}
    details: dict[str, object] = {
        "schema_version": 1, "protocol": PROTOCOL,
        "processed_manifest_sha256": _digest(data.root / "manifest.json"),
        "split_sha256": _digest(data.root / "splits.npz"),
        "sd_ddof": 1, "conditions": {},
    }
    seed_rows: list[dict[str, object]] = []
    shared_pretrain_code: dict[str, str] | None = None
    shared_evaluation_code: dict[str, str] | None = None
    current_table4_source_sha256 = _digest(Path(__file__))
    for corpus in TABLE4_CLASSES:
        for condition in CONDITIONS:
            runs = [verify_score(data, output_dir, corpus, condition, seed) for seed in RUN_SEEDS]
            values = np.array([[run["metrics_percent"][metric] for metric in METRICS]
                               for run in runs], dtype=np.float64)
            summary = summarize_scores(values)
            scores_by_condition[(corpus, condition)] = values
            details["conditions"][f"{corpus}/{condition}"] = {
                "isolate_ids": list(TABLE4_CLASSES[corpus]),
                "train_count": len(data.splits[f"{corpus}_train"]),
                "test_count": len(data.splits[f"{corpus}_test"]),
                "test_source_index_sha256": _index_digest(data.splits[f"{corpus}_test"]),
                "run_seeds": list(RUN_SEEDS), "scores_percent": values.tolist(),
                "metrics": summary,
            }
            for seed, run in zip(RUN_SEEDS, runs, strict=True):
                evaluation_code = run["code_sha256"]
                if shared_evaluation_code is None:
                    shared_evaluation_code = evaluation_code
                elif evaluation_code != shared_evaluation_code:
                    raise ValueError("Table 4 evaluations used different source code versions")
                pretrain_run, done, checkpoint = verify_pretraining(
                    data, output_dir, corpus, condition, seed,
                )
                code_hashes = pretrain_run["contract"]["code_sha256"]
                if not isinstance(code_hashes, dict):
                    raise ValueError("Table 4 pretraining code hashes are missing")
                source_hash = code_hashes.get("bacteria_table4_80.py")
                if source_hash != current_table4_source_sha256 and not (
                    (corpus, condition, seed) == ("bacteria4", "M00", 0)
                    and source_hash == _PRE_SCORE_FIX_SOURCE_SHA256
                ):
                    raise ValueError("Table 4 pretraining source is incompatible")
                # This patch changed only evaluation metadata and reporting in this module.
                training_dependencies = {
                    name: digest for name, digest in code_hashes.items()
                    if name != "bacteria_table4_80.py"
                }
                if shared_pretrain_code is None:
                    shared_pretrain_code = training_dependencies
                elif training_dependencies != shared_pretrain_code:
                    raise ValueError("Table 4 pretraining runs used different source code versions")
                seed_rows.append({
                    "corpus": corpus, "condition": condition, "seed": seed,
                    **run["metrics_percent"],
                    "pretrain_source_sha256": source_hash,
                    "train_count": run["train_count"], "test_count": run["test_count"],
                    "epochs": done["completed_epochs"],
                    "batch_size": table4_config()["pretrain"]["batch_size"],
                    "batches_per_epoch": (
                        run["train_count"] // table4_config()["pretrain"]["batch_size"]
                    ),
                    "attempted_updates": done["attempted_updates"],
                    "optimizer_updates": done["optimizer_updates"],
                    "amp_skips": done["amp_skips"],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": done["weights_sha256"],
                })

    details["seed_runs"] = seed_rows
    details["pretrain_code_sha256"] = shared_pretrain_code
    details["evaluation_code_sha256"] = shared_evaluation_code

    lines = [
        "# Table 4 · Bacteria-ID clustering (80/20)", "",
        "| Method | Bacteria-4 ACC | NMI | AMI | Bacteria-6 ACC | NMI | AMI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in TABLE4_REPORTED:
        lines.append("| " + name + " | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    for condition in CONDITIONS:
        cells = []
        for corpus in TABLE4_CLASSES:
            for metric in METRICS:
                values = scores_by_condition[(corpus, condition)][:, METRICS.index(metric)]
                cells.append(f"{values.mean():.2f} ± {values.std(ddof=1):.2f}")
        lines.append(f"| ChemoMAE({condition}) | " + " | ".join(cells) + " |")
    lines += [
        "", "All values are percentages. ChemoMAE: five independent seeds, mean ± sample SD "
        "(ddof=1); no seed selection. The six comparison rows are unchanged reported values "
        "from [Ren et al. (2025), Table 4](https://doi.org/10.1016/j.eswa.2025.128576). "
        "ChemoMAE uses SNV, 128-dimensional unit latents, 800 fixed pretraining epochs "
        "on the 80% train partition, and Cosine-KMeans fit on all train latents. The "
        "held-out 20% test partition is assigned to fixed centers. The literature "
        "methods' training fractions, exact split "
        "indices and evaluation procedures cannot be confirmed to match these conditions; "
        "this is not a strict same-condition comparison. Bacteria-4 uses IDs 0–3 and "
        "Bacteria-6 uses IDs 0–5 following SMAE supplementary Fig. S3.", "",
    ]
    csv_buffer = io.StringIO(newline="")
    fields = list(seed_rows[0])
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(seed_rows)
    png = _score_figure(scores_by_condition)
    paths = (output_dir / "table4.md", output_dir / "table4_seed_scores.csv",
             output_dir / "table4_statistics.json", output_dir / "table4_seed_scores.png")
    if any(path.exists() for path in paths):
        raise FileExistsError("80/20 Table 4 outputs already exist; review before regeneration")
    with paths[0].open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    with paths[1].open("x", encoding="utf-8", newline="") as stream:
        stream.write(csv_buffer.getvalue())
    _write_json(paths[2], details)
    with paths[3].open("xb") as stream:
        stream.write(png)
    return paths
