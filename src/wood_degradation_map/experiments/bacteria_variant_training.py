"""Additional Table 5 classifier paths sharing the completed Bacteria-ID pretraining."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from chemomae.models.chemo_mae import ChemoMAE
from torch import nn

from .bacteria_config import bacteria_config, purpose_seed
from .bacteria_data import PreparedBacteria
from .bacteria_training import (
    BacteriaClassifier, BacteriaFinetuner, _augmentation, _check_batch,
    _code_hashes, _validate_run, load_pretrained,
)
from .manifests import _digest, _read_json, _write_json
from .neural import TorchRandomStream
from .records import make_run_record
from .training import runtime_record


Variant = Literal["cls_frozen", "z_unfrozen", "z_frozen"]
NEW_VARIANTS: tuple[Variant, ...] = ("z_unfrozen", "cls_frozen", "z_frozen")
ALL_VARIANTS = ("cls_unfrozen", *NEW_VARIANTS)


def variant_paths(
    output_dir: Path, condition: str, variant: Variant, run_seed: int,
) -> tuple[Path, Path]:
    """Use distinct result and checkpoint roots for the three new paths."""
    _validate_run("reference30", condition, run_seed)
    if variant not in NEW_VARIANTS:
        raise ValueError(f"Unknown additional Table 5 variant: {variant}")
    suffix = Path("finetune_variants") / "reference30" / condition / variant / f"seed_{run_seed}"
    return output_dir.resolve() / "results" / suffix, output_dir.resolve() / "checkpoints" / suffix


def variant_config(variant: Variant) -> dict[str, object]:
    """Record the actual head and trainable encoder in the run contract."""
    if variant not in NEW_VARIANTS:
        raise ValueError(f"Unknown additional Table 5 variant: {variant}")
    config = bacteria_config()
    finetune = config["table5"]["finetune"]
    finetune["head"] = ("CLS256 -> LayerNorm256 -> Linear30" if variant == "cls_frozen"
                        else "unit z128 -> LayerNorm128 -> Linear30")
    finetune["encoder_trainable"] = variant == "z_unfrozen"
    finetune["variant"] = variant
    return config


class BacteriaVariantClassifier(nn.Module):
    """Frozen CLS or unit-normalized z, with the same LayerNorm+linear head."""

    def __init__(self, pretrained: ChemoMAE, run_seed: int, variant: Variant) -> None:
        super().__init__()
        if variant not in NEW_VARIANTS:
            raise ValueError(f"Unknown additional Table 5 variant: {variant}")
        self.variant = variant
        if variant == "cls_frozen":
            self.cls = BacteriaClassifier(pretrained, run_seed)
            self.cls.encoder.requires_grad_(False)
        else:
            self.encoder = pretrained.encoder
            if not self.encoder.latent_normalize:
                raise ValueError("The z classifier requires unit-normalized latents")
            if variant == "z_frozen":
                self.encoder.requires_grad_(False)
            with TorchRandomStream(purpose_seed("reference30", run_seed, "finetune_head"),
                                   torch.device("cpu")).scope():
                self.norm = nn.LayerNorm(128)
                self.head = nn.Linear(128, 30)
                nn.init.xavier_uniform_(self.head.weight)
                nn.init.zeros_(self.head.bias)
                nn.init.ones_(self.norm.weight)
                nn.init.zeros_(self.norm.bias)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        if self.variant == "cls_frozen":
            return self.cls(spectra)
        _check_batch(spectra)
        z = self.encoder(spectra, torch.ones_like(spectra, dtype=torch.bool))
        if z.shape != (len(spectra), 128):
            raise RuntimeError("ChemoMAE encoder did not return 128-dimensional z")
        return self.head(self.norm(z))


class BacteriaVariantFinetuner(BacteriaFinetuner):
    """Reuse the original fit/resume/selection loop with a distinct classifier path."""

    def __init__(
        self, data: PreparedBacteria, output_dir: Path, condition: str, run_seed: int,
        variant: Variant, *, device: torch.device, resume_from: Path | None = None,
    ) -> None:
        _validate_run("reference30", condition, run_seed)
        if variant not in NEW_VARIANTS:
            raise ValueError(f"Unknown additional Table 5 variant: {variant}")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Production finetuning requires a single CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.data = data
        self.condition, self.run_seed, self.variant = condition, run_seed, variant
        self.recipe = bacteria_config()["table5"]["finetune"]
        self.results_dir, self.weights_dir = variant_paths(output_dir, condition, variant, run_seed)
        self.last_path = self.weights_dir / "last.pt"
        self.best_path = self.weights_dir / "best_model.pt"
        pretrain_results = output_dir.resolve() / "results/pretrain/reference30" / condition / f"seed_{run_seed}"
        pretrain_weights = output_dir.resolve() / "checkpoints/pretrain/reference30" / condition / f"seed_{run_seed}"
        code_hashes = {**_code_hashes(), Path(__file__).name: _digest(Path(__file__))}
        self.run_record = make_run_record({
            "schema_version": 2, "mode": "bacteria_id_finetune_variant",
            "condition": condition, "run_seed": run_seed, "variant": variant,
            "processed_manifest_sha256": _digest(data.root / "manifest.json"),
            "pretrain_run_sha256": _digest(pretrain_results / "run.json"),
            "pretrain_weights_sha256": _digest(pretrain_weights / "last_model.pt"),
            "seeds": {purpose: purpose_seed("reference30", run_seed, purpose) for purpose in
                      ("finetune_head", "finetune_order", "finetune_augmentation")},
            "config": variant_config(variant), "runtime": runtime_record(device),
            "code_sha256": code_hashes,
        })
        if resume_from is None:
            if self.results_dir.exists() or self.weights_dir.exists():
                raise FileExistsError("Finetune run exists; resume explicitly from last.pt")
        else:
            if resume_from.resolve() != self.last_path.resolve() or not resume_from.is_file():
                raise ValueError("Resume file must be this variant run's last.pt")
            if _read_json(self.results_dir / "run.json") != self.run_record:
                raise ValueError("Variant finetune run/config/data/code/runtime differs; resume rejected")
        pretrained = load_pretrained(output_dir, data, "reference30", condition, run_seed,
                                     torch.device("cpu"))
        self.model = BacteriaVariantClassifier(pretrained, run_seed, variant).to(device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.recipe["lr"], betas=tuple(self.recipe["betas"]),
            weight_decay=self.recipe["weight_decay"],
        )
        scheduler_cfg = self.recipe["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=scheduler_cfg["mode"], factor=scheduler_cfg["factor"],
            patience=scheduler_cfg["patience"], threshold=scheduler_cfg["threshold"],
            threshold_mode="rel",
        )
        self.order = torch.Generator(device="cpu").manual_seed(
            purpose_seed("reference30", run_seed, "finetune_order"),
        )
        self.augmentation = TorchRandomStream(
            purpose_seed("reference30", run_seed, "finetune_augmentation"), device,
        )
        self.augmenter = _augmentation(0.5, 0.5)
        self.train_x, train_y = data.rows("finetune", "finetune_train")
        self.val_x, val_y = data.rows("finetune", "finetune_validation")
        self.train_x = torch.from_numpy(self.train_x)
        self.val_x = torch.from_numpy(self.val_x)
        self.train_y = torch.from_numpy(np.array(train_y, dtype=np.int64, copy=True))
        self.val_y = torch.from_numpy(np.array(val_y, dtype=np.int64, copy=True))
        self.epoch = 0
        self.best_accuracy = -math.inf
        self.best_epoch = 0
        self.early_best = -math.inf
        self.early_wait = 0
        self.previous_validation = 0.0
        self.history: list[dict[str, object]] = []
        self.best_state: dict[str, torch.Tensor] | None = None
        if resume_from is None:
            self.results_dir.mkdir(parents=True, exist_ok=False)
            self.weights_dir.mkdir(parents=True, exist_ok=False)
            _write_json(self.results_dir / "run.json", self.run_record)
        else:
            self._restore()
