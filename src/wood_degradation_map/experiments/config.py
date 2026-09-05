"""Fixed scientific settings and deterministic, purpose-specific seed planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal


ROOT_SEED = 20260905
FOLDS = (1, 2, 3, 4, 5)
REPEATS = (1, 2, 3)
CLUSTER_COUNTS = (2, 4, 6, 8, 10, 12, 14)
PIXELS_PER_SAMPLE = 8192
ADOPTED_SAMPLE_IDS = (
    "KYOw02702", "KYOw02707", "KYOw02708", "KYOw02709", "KYOw02715", "KYOw02716", "KYOw02717",
    "KYOw02719", "KYOw02720", "KYOw02751", "KYOw02752", "KYOw02754", "KYOw02756", "KYOw02758",
    "KYOw02759", "KYOw02760", "KYOw02762", "KYOw02763", "KYOw02764", "KYOw02766", "KYOw02767",
    "KYOw02768", "KYOw02769", "KYOw02770", "KYOw02771", "KYOw02772", "KYOw02773", "KYOw02774",
    "KYOw02775", "KYOw02776", "KYOw02777", "KYOw02780", "KYOw02783", "KYOw02784", "KYOw02787",
    "KYOw02788", "KYOw02789", "KYOw02790", "KYOw16662", "KYOw16666", "KYOw16700", "KYOw16702",
    "KYOw16711", "KYOw16714", "KYOw16716", "KYOw16719", "KYOw16737", "KYOw16744", "KYOw16750",
)
RunPurpose = Literal["model_init", "pixel_order", "mask", "train_aug", "pca"]
Perturbation = Literal["noise", "shift", "both"]
RUN_PURPOSES: tuple[RunPurpose, ...] = (
    "model_init", "pixel_order", "mask", "train_aug", "pca",
)
PERTURBATIONS: tuple[Perturbation, ...] = ("noise", "shift", "both")


@dataclass(frozen=True)
class Condition:
    """Differences from the common recipe; B0/B1 have no reconstruction loss."""

    condition_id: str
    representation: str
    output_dim: int
    n_mask: int | None
    loss_region: str | None
    noise_prob: float
    shift_prob: float
    experiment: str = "main"


CONDITIONS = (
    Condition("B0", "snv", 256, None, None, 0.0, 0.0),
    Condition("B1", "pca", 16, None, None, 0.0, 0.0),
    Condition("A0", "chemomae", 16, 0, "all", 0.0, 0.0),
    Condition("M00", "chemomae", 16, 8, "masked", 0.0, 0.0),
    Condition("M10", "chemomae", 16, 8, "masked", 0.5, 0.0),
    Condition("M01", "chemomae", 16, 8, "masked", 0.0, 0.5),
    Condition("M11", "chemomae", 16, 8, "masked", 0.5, 0.5),
    Condition("M11-25", "chemomae", 16, 4, "masked", 0.5, 0.5, "mask_sensitivity"),
    Condition("M11-75", "chemomae", 16, 12, "masked", 0.5, 0.5, "mask_sensitivity"),
)


def experiment_config() -> dict[str, object]:
    """Return a fresh JSON-compatible snapshot of the fixed CV protocol."""
    return {
        "schema_version": 1,
        "protocol": "docs/design/experiment_protocol.md",
        "preprocessing_id": "production_v1",
        "input": {"dataset": "snv", "bands": 256, "dtype": "float32"},
        "split": {
            "unit": "KYOw", "n_folds": 5, "stratification": None,
            "algorithm": "sorted IDs, NumPy PCG64 permutation, array_split",
            "fold_ids": list(FOLDS), "repeat_ids": list(REPEATS),
            "adopted_sample_ids": list(ADOPTED_SAMPLE_IDS),
            "cross_KYOw_source_relationship": "unknown; KYOw-only split approved 2026-09-05",
        },
        "sampling": {
            "q": PIXELS_PER_SAMPLE, "replace": False, "distribution": "uniform",
            "algorithm": "NumPy PCG64 choice; HDF5 rows sorted after selection",
            "shared_across": ["condition", "K", "training_repeat", "epoch"],
            "test": "all valid saved rows",
        },
        "seeds": {"root_seed": ROOT_SEED, "algorithm": "sha256-json-v1-first-32-bits"},
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "representation": {"l2_normalize_rows": True, "b0_pca_normalization_eps": 1e-6},
        "pca": {
            "n_components": 16, "copy": True, "whiten": False, "svd_solver": "auto",
            "tol": 0.0, "iterated_power": "auto", "n_oversamples": 10,
            "power_iteration_normalizer": "auto", "random_state": None,
            "rng_policy": "isolated NumPy random state set to the recorded PCA seed",
            "fit_scope": "shared train pixels", "additional_autoscaling": False,
        },
        "chemomae": {
            "version": "0.2.1", "seq_len": 256, "d_model": 256, "nhead": 8,
            "num_layers": 8, "dim_feedforward": 1024, "dropout": 0.0,
            "n_patches": 16, "latent_dim": 16, "latent_normalize": True,
            "decoder_num_layers": 1, "initialization": "ChemoMAE 0.2.1 defaults",
        },
        "training": {
            "epochs": 800, "batch_size": 1024, "world_size": 1, "accum_iter": 1,
            "optimizer": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8, "amsgrad": False,
            "weight_decay": 0.05, "zero_weight_decay": ["bias", "normalization_parameters"],
            "base_lr": 1.5e-4, "peak_lr": 6e-4, "min_lr": 0.0, "warmup_epochs": 40,
            "scheduler": "linear warmup then half-cycle cosine; epoch + batch/steps_per_epoch",
            "lr_update": "before batch", "amp_dtype": "fp16", "grad_scaler": True,
            "weight_dtype": "float32", "grad_clip": None, "drop_path": 0.0,
            "shuffle": True, "drop_last": True, "use_ema": False,
            "checkpoint": "last_model.pt; final raw weights at epoch 800",
            "independent_run_resume_from": None, "early_stopping": False,
            "target": "clean SNV", "patch_target_normalization": False,
        },
        "augmentation": {
            "noise_angle_deg_range": [0.0, 2.5], "shift_delta_range": [-2.0, 2.0],
            "shuffle_order_per_batch": True, "recenter_after_each_op": True,
            "renorm_to_input_norm": True, "eps": 1e-8,
        },
        "extraction": {
            "visible_mask": "all", "model_mode": "eval", "amp": False,
            "dtype": "float32", "tf32": False, "augmenter": None,
        },
        "clustering": {
            "K": list(CLUSTER_COUNTS), "K0": 8, "max_iter": 500, "tol": 1e-4,
            "initializations_per_fit": 1, "dtype": "float32", "device": "cuda",
            "normalization_eps": 1e-6, "other_settings": "ChemoMAE 0.2.1 defaults",
            "fit_scope": "shared train pixels", "test_centers": "fixed",
        },
        "evaluation": {
            "dtype": "float32", "tf32": False, "lla_windows": [3, 5, 9],
            "perturbations": list(PERTURBATIONS), "perturbation_repeats": 5,
            "perturbation_active_operation_prob": 1.0,
            "perturbation_shared_across": ["condition", "K", "training_repeat"],
            "silhouette_eps": 1e-12, "latent_normalization_eps": 1e-12,
            "aggregation": "sample macro; sample SD separate from training-repeat SD",
            "ari_pairs": [[1, 2], [1, 3], [2, 3]],
            "background_label": 0, "cluster_labels": "1..K",
        },
    }


def _seed(purpose: str, *context: int | str) -> int:
    # Python hash() is intentionally avoided because it changes between processes.
    payload = json.dumps(["wood-degradation-map-seeds-v1", ROOT_SEED, purpose, *context],
                         ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def split_seed() -> int:
    return _seed("split")


def sampling_seed(fold: int, sample_id: str) -> int:
    if fold not in FOLDS:
        raise ValueError("fold must be 1..5")
    return _seed("sampling", fold, sample_id)


def run_seed(purpose: RunPurpose, fold: int, repeat: int) -> int:
    if purpose not in RUN_PURPOSES or fold not in FOLDS or repeat not in REPEATS:
        raise ValueError("Invalid run seed purpose, fold, or repeat")
    return _seed(purpose, fold, repeat)


def kmeans_seed(fold: int, repeat: int, k: int) -> int:
    if fold not in FOLDS or repeat not in REPEATS or k not in CLUSTER_COUNTS:
        raise ValueError("Invalid KMeans fold, repeat, or K")
    return _seed("kmeans", fold, repeat, k)


def perturbation_seed(sample_id: str, kind: Perturbation, draw: int) -> int:
    if kind not in PERTURBATIONS or draw not in range(1, 6):
        raise ValueError("Invalid perturbation kind or draw")
    return _seed("evaluation_perturbation", sample_id, kind, draw)


def seed_plan(test_folds: dict[str, int]) -> dict[str, object]:
    """List actual seeds; shared inputs have no condition/K/training-repeat key."""
    records: list[dict[str, object]] = [{"purpose": "split", "seed": split_seed()}]
    for sample_id, test_fold in sorted(test_folds.items()):
        for fold in FOLDS:
            if fold != test_fold:
                records.append({"purpose": "sampling", "sample_id": sample_id, "fold": fold,
                                "seed": sampling_seed(fold, sample_id)})
        for kind in PERTURBATIONS:
            for draw in range(1, 6):
                records.append({"purpose": "evaluation_perturbation", "sample_id": sample_id,
                                "kind": kind, "draw": draw,
                                "seed": perturbation_seed(sample_id, kind, draw)})
    for fold in FOLDS:
        for repeat in REPEATS:
            for purpose in RUN_PURPOSES:
                records.append({"purpose": purpose, "fold": fold, "repeat": repeat,
                                "seed": run_seed(purpose, fold, repeat)})
            for k in CLUSTER_COUNTS:
                records.append({"purpose": "kmeans", "fold": fold, "repeat": repeat, "K": k,
                                "seed": kmeans_seed(fold, repeat, k)})
    if len({record["seed"] for record in records}) != len(records):
        raise ValueError("Seed collision in the fixed plan; do not silently choose another seed")
    return {"root_seed": ROOT_SEED, "algorithm": "sha256-json-v1-first-32-bits",
            "records": records}
