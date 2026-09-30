"""Fixed Bacteria-ID auxiliary protocol; independent of the 256-band wood data."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Literal

from .config import experiment_config


Corpus = Literal["bacteria4", "bacteria6", "reference30"]
ConditionId = Literal["M00", "M11"]
Purpose = Literal[
    "model_init", "train_order", "mask", "pretrain_augmentation", "kmeans",
    "finetune_head", "finetune_order", "finetune_augmentation",
]

CORPORA: tuple[Corpus, ...] = ("bacteria4", "bacteria6", "reference30")
CONDITIONS: tuple[ConditionId, ...] = ("M00", "M11")
RUN_SEEDS = tuple(range(5))
PURPOSES: tuple[Purpose, ...] = (
    "model_init", "train_order", "mask", "pretrain_augmentation", "kmeans",
    "finetune_head", "finetune_order", "finetune_augmentation",
)
TABLE4_CLASSES = {"bacteria4": (0, 1, 2, 3), "bacteria6": (0, 1, 2, 3, 4, 5)}
SCHEMA_VERSION = 1


def purpose_seed(corpus: Corpus, run_seed: int, purpose: Purpose) -> int:
    """Produce the documented first-32-bit SHA-256 seed, without condition ID."""
    if corpus not in CORPORA or run_seed not in RUN_SEEDS or purpose not in PURPOSES:
        raise ValueError("Unknown Bacteria-ID corpus, run seed or RNG purpose")
    payload = json.dumps(
        ["bacteria-id-seeds-v1", corpus, run_seed, purpose],
        ensure_ascii=True, separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_manifest() -> dict[str, dict[str, dict[str, int]]]:
    """List every fixed run and reject accidental 32-bit seed collisions."""
    result: dict[str, dict[str, dict[str, int]]] = {}
    seen: dict[int, tuple[str, int, str]] = {}
    for corpus in CORPORA:
        result[corpus] = {}
        for run_seed in RUN_SEEDS:
            seeds = {purpose: purpose_seed(corpus, run_seed, purpose) for purpose in PURPOSES}
            for purpose, value in seeds.items():
                identity = (corpus, run_seed, purpose)
                if value in seen:
                    raise ValueError(f"Seed collision: {seen[value]} and {identity}")
                seen[value] = identity
            result[corpus][str(run_seed)] = seeds
    return result


def bacteria_config() -> dict[str, object]:
    """Return a JSON-compatible snapshot of settings and their provenance."""
    wood = experiment_config()
    model = deepcopy(wood["chemomae"])
    model.update(seq_len=1000, n_patches=20, latent_dim=128, n_mask=10)
    pretrain = deepcopy(wood["training"])
    augmentation = deepcopy(wood["augmentation"])
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "docs/design/bacteria_id_experiment_plan.md",
        "model": model,
        "pretrain": pretrain,
        "augmentation": augmentation,
        "conditions": {
            "M00": {"pretrain_noise_prob": 0.0, "pretrain_shift_prob": 0.0},
            "M11": {"pretrain_noise_prob": 0.5, "pretrain_shift_prob": 0.5},
        },
        "preprocessing": {"method": "per-spectrum SNV", "ddof": 1, "dtype": "float32"},
        "table4": {
            "isolate_ids": {key: list(value) for key, value in TABLE4_CLASSES.items()},
            "split_seed": 42, "split": [0.6, 0.2, 0.2],
            "clustering": {"method": "ChemoMAE CosineKMeans", "max_iter": 500,
                           "tol": 1e-4, "initializations": 1},
            "metrics": {"ACC": "Hungarian", "NMI_average_method": "arithmetic",
                        "AMI_average_method": "max"},
        },
        "table5": {
            "split": {"method": "StratifiedKFold", "n_splits": 5,
                      "shuffle": True, "random_state": 42, "fold_key": 1},
            "finetune": {
                "max_epochs": 200, "batch_size": 16, "optimizer": "AdamW",
                "lr": 1e-4, "betas": [0.9, 0.95], "weight_decay": 1e-5,
                "scheduler": {"mode": "max", "factor": 0.3, "patience": 10,
                              "threshold": 1e-4, "step_order": "after train, previous validation"},
                "early_stopping": {"monitor": "validation_accuracy", "patience": 20,
                                   "delta": 1e-4},
                "checkpoint_selection": "maximum validation accuracy; earliest epoch on tie",
                "head": "CLS256 -> LayerNorm256 -> Linear30",
                "encoder_trainable": True, "amp": False,
                "augmentation": "TGN and FS, each p=0.5, train only",
            },
            "report": "5 independent pretrain+finetune seeds; mean +/- t(0.975,4) * sample_sd/sqrt(5)",
        },
        "run_seeds": list(RUN_SEEDS),
        "provenance": {
            "model_pretrain_augmentation": "WoodDegradationMap adopted recipe; user-selected Raman dimensions",
            "table5_finetune": "SMAE published paper and pinned public code, with user-selected ChemoMAE encoder/TGN/FS",
            "splits_seed_ci_checkpoint": "documented local supplementation of unpublished details",
        },
    }
