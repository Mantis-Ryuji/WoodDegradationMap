"""Checkpointed M00/M11 Bacteria-ID pretraining and SMAE-style classification."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from chemomae.models.chemo_mae import ChemoMAE, make_patch_mask
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig
from chemomae.training.trainer import Trainer, TrainerConfig
from torch import nn

from .bacteria_config import CONDITIONS, CORPORA, RUN_SEEDS, bacteria_config, purpose_seed
from .bacteria_data import CHANNELS, PreparedBacteria
from .manifests import _digest, _read_json, _write_json
from .neural import TorchRandomStream, build_optimizer, fp32_inference
from .records import make_run_record
from .training import ExperimentTrainer, runtime_record


def _validate_run(corpus: str, condition: str, run_seed: int) -> None:
    if corpus not in CORPORA or condition not in CONDITIONS or run_seed not in RUN_SEEDS:
        raise ValueError("Only fixed Bacteria-ID corpora, M00/M11 and seeds 0..4 are allowed")


def _check_batch(batch: torch.Tensor) -> None:
    if (batch.ndim != 2 or batch.shape[1] != CHANNELS or not len(batch)
            or batch.dtype != torch.float32 or batch.device.type not in ("cpu", "cuda")):
        raise ValueError("Expected nonempty FP32 Bacteria-ID spectra of length 1000")
    if not bool(torch.isfinite(batch).all()) or not bool((torch.linalg.vector_norm(batch, dim=1) > 0).all()):
        raise ValueError("Bacteria-ID spectra contain nonfinite or zero-norm rows")


def _augmentation(noise_prob: float, shift_prob: float) -> SpectraAugmenter:
    settings = dict(bacteria_config()["augmentation"])
    for key in ("noise_angle_deg_range", "shift_delta_range"):
        settings[key] = tuple(settings[key])
    return SpectraAugmenter(SpectraAugmenterConfig(
        noise_prob=noise_prob, shift_prob=shift_prob, **settings,
    )).train()


def build_bacteria_model(corpus: str, run_seed: int) -> ChemoMAE:
    """Initialize the documented 1000-channel model in an isolated CPU stream."""
    _validate_run(corpus, "M00", run_seed)
    settings = dict(bacteria_config()["model"])
    required = settings.pop("version")
    settings.pop("initialization")
    if version("chemomae") != required or torch.get_default_dtype() != torch.float32:
        raise ValueError(f"Requires ChemoMAE {required} and default FP32 initialization")
    with TorchRandomStream(purpose_seed(corpus, run_seed, "model_init"), torch.device("cpu")).scope():
        with torch.device("cpu"):
            return ChemoMAE(**settings)


class BacteriaRandomness:
    """Independent train-order, patch-mask and augmentation RNG streams."""

    def __init__(self, corpus: str, condition: str, run_seed: int, device: torch.device) -> None:
        _validate_run(corpus, condition, run_seed)
        self.pixel_order = torch.Generator(device="cpu").manual_seed(
            purpose_seed(corpus, run_seed, "train_order"),
        )
        self.mask = TorchRandomStream(purpose_seed(corpus, run_seed, "mask"), device)
        self.augmentation = TorchRandomStream(
            purpose_seed(corpus, run_seed, "pretrain_augmentation"), device,
        )
        probability = 0.5 if condition == "M11" else 0.0
        self.augmenter = _augmentation(probability, probability)
        self.condition = condition

    def epoch_batches(self, train_count: int) -> Iterator[torch.Tensor]:
        batch_size = bacteria_config()["pretrain"]["batch_size"]
        if train_count < batch_size:
            raise ValueError("Pretrain selection is smaller than fixed batch size 1024")
        order = torch.randperm(train_count, generator=self.pixel_order, device="cpu")
        for start in range(0, train_count - batch_size + 1, batch_size):
            yield order[start:start + batch_size]

    def prepare(self, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply extra Aug to input only, then draw ten masked patches per row."""
        _check_batch(clean)
        if clean.device != self.mask.device:
            raise ValueError("Bacteria-ID batch and RNG device differ")
        with torch.autocast(device_type=clean.device.type, enabled=False):
            if self.condition == "M11":
                with self.augmentation.scope():
                    augmented = self.augmenter(clean.clone())
            else:
                augmented = clean.clone()
            _check_batch(augmented)
            with self.mask.scope():
                visible = ~make_patch_mask(
                    len(clean), seq_len=CHANNELS, n_patches=20, n_mask=10,
                    device=clean.device,
                )
        return augmented, visible


@dataclass(frozen=True)
class BacteriaTrainingData:
    corpus: str
    spectra: torch.Tensor
    source_rows: int

    @classmethod
    def from_prepared(cls, data: PreparedBacteria, corpus: str) -> BacteriaTrainingData:
        if corpus not in CORPORA:
            raise ValueError(f"Unknown Bacteria-ID corpus: {corpus}")
        selection = "reference30_pretrain" if corpus == "reference30" else f"{corpus}_train"
        values, _ = data.rows("reference", selection)
        return cls(corpus, torch.from_numpy(values), len(values))


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    names = ("bacteria_config.py", "bacteria_data.py", "bacteria_training.py",
             "bacteria_evaluation.py", "config.py", "neural.py", "training.py")
    return {name: _digest(directory / name) for name in names}


def _run_paths(output_dir: Path, stage: str, corpus: str, condition: str, run_seed: int) -> tuple[Path, Path]:
    suffix = Path(stage) / corpus / condition / f"seed_{run_seed}"
    return output_dir / "results" / suffix, output_dir / "checkpoints" / suffix


class BacteriaPretrainer(ExperimentTrainer):
    """Use the unchanged wood pretraining loop with Raman data/model/RNG adapters."""

    def __init__(
        self, train: BacteriaTrainingData, data: PreparedBacteria, output_dir: Path,
        condition: str, run_seed: int, *, device: torch.device,
        resume_from: Path | None = None,
    ) -> None:
        corpus = train.corpus
        _validate_run(corpus, condition, run_seed)
        if (device.type != "cuda" or not torch.cuda.is_available()
                or train.spectra.dtype != torch.float32 or train.spectra.device.type != "cpu"
                or train.spectra.ndim != 2 or train.spectra.shape[1] != CHANNELS):
            raise ValueError("Production pretraining requires CUDA and CPU FP32 1000-channel input")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.recipe = bacteria_config()["pretrain"]
        self.steps_per_epoch = len(train.spectra) // self.recipe["batch_size"]
        if self.steps_per_epoch < 1:
            raise ValueError("Pretrain selection is smaller than batch size 1024")
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
        self.results_dir, weights_dir = _run_paths(output_dir.resolve(), "pretrain", corpus, condition, run_seed)
        checkpoint_dir = weights_dir / "checkpoints"
        self.run_record = make_run_record({
            "schema_version": 2, "mode": "bacteria_id_pretrain", "corpus": corpus,
            "condition": condition, "run_seed": run_seed,
            "train_spectra": len(train.spectra), "batch_size": self.recipe["batch_size"],
            "steps_per_epoch": self.steps_per_epoch,
            "planned_updates": self.steps_per_epoch * self.recipe["epochs"],
            "processed_manifest_sha256": _digest(data.root / "manifest.json"),
            "source_split": "reference30_pretrain" if corpus == "reference30" else f"{corpus}_train",
            "seeds": {purpose: purpose_seed(corpus, run_seed, purpose) for purpose in
                      ("model_init", "train_order", "mask", "pretrain_augmentation")},
            "config": bacteria_config(), "runtime": runtime_record(device),
            "code_sha256": _code_hashes(),
            "resume_boundary": "completed epoch; interrupted epoch is replayed",
        })
        if resume_from is None:
            if self.results_dir.exists() or weights_dir.exists():
                raise FileExistsError("Run exists; resume explicitly from its last.pt checkpoint")
        else:
            resume_from = resume_from.resolve()
            if resume_from != (checkpoint_dir / "last.pt").resolve() or not resume_from.is_file():
                raise ValueError("Resume file must be this run's last.pt")
            if _read_json(self.results_dir / "run.json") != self.run_record:
                raise ValueError("Pretrain run/config/data/code/runtime differs; resume rejected")
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


def load_pretrained(
    output_dir: Path, processed: PreparedBacteria, corpus: str, condition: str,
    run_seed: int, device: torch.device,
) -> ChemoMAE:
    """Accept only a complete, same-data 800-epoch raw pretraining run."""
    _validate_run(corpus, condition, run_seed)
    results, weights = _run_paths(output_dir.resolve(), "pretrain", corpus, condition, run_seed)
    run = _read_json(results / "run.json")
    done = _read_json(results / "completion.json")
    if (run.get("mode") != "bacteria_id_pretrain" or run.get("corpus") != corpus
            or run.get("condition") != condition or run.get("run_seed") != run_seed
            or run.get("processed_manifest_sha256") != _digest(processed.root / "manifest.json")
            or run.get("contract", {}).get("config") != bacteria_config()
            or done.get("status") != "training_completed" or done.get("completed_epochs") != 800):
        raise ValueError("Required same-protocol Bacteria-ID pretraining is incomplete")
    path = weights / "last_model.pt"
    if done.get("weights_sha256") != _digest(path):
        raise ValueError("Raw pretrain weights differ from completion record")
    model = build_bacteria_model(corpus, run_seed)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device)


class BacteriaClassifier(nn.Module):
    """CLS output of the full-visible pretrained encoder into SMAE-style head."""

    def __init__(self, pretrained: ChemoMAE, run_seed: int) -> None:
        super().__init__()
        self.encoder = pretrained.encoder
        self.encoder.to_latent.requires_grad_(False)  # Projection is outside the CLS classification path.
        with TorchRandomStream(purpose_seed("reference30", run_seed, "finetune_head"),
                               torch.device("cpu")).scope():
            self.norm = nn.LayerNorm(256)
            self.head = nn.Linear(256, 30)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        _check_batch(spectra)
        captured: list[torch.Tensor] = []

        def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured.append(inputs[0])

        handle = self.encoder.to_latent.register_forward_pre_hook(capture)
        try:
            self.encoder(spectra, torch.ones_like(spectra, dtype=torch.bool))
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != (len(spectra), 256):
            raise RuntimeError("ChemoMAE CLS capture failed")
        return self.head(self.norm(captured[0]))


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _accuracy(
    model: BacteriaClassifier, spectra: torch.Tensor, labels: torch.Tensor,
    device: torch.device, *, batch_size: int = 16,
) -> float:
    model.eval()
    correct = 0
    with fp32_inference(device):
        for start in range(0, len(spectra), batch_size):
            end = min(start + batch_size, len(spectra))
            x = spectra[start:end].to(device)
            prediction = model(x).argmax(dim=1)
            correct += int((prediction == labels[start:end].to(device)).sum())
    return correct / len(spectra)


class BacteriaFinetuner:
    """Five independently initialized pretrain→finetune runs, with epoch resume."""

    def __init__(
        self, data: PreparedBacteria, output_dir: Path, condition: str, run_seed: int,
        *, device: torch.device, resume_from: Path | None = None,
    ) -> None:
        _validate_run("reference30", condition, run_seed)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Production finetuning requires a single CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.data = data
        self.condition, self.run_seed = condition, run_seed
        self.recipe = bacteria_config()["table5"]["finetune"]
        self.results_dir, self.weights_dir = _run_paths(
            output_dir.resolve(), "finetune", "reference30", condition, run_seed,
        )
        self.last_path = self.weights_dir / "last.pt"
        self.best_path = self.weights_dir / "best_model.pt"
        pretrain_results, pretrain_weights = _run_paths(
            output_dir.resolve(), "pretrain", "reference30", condition, run_seed,
        )
        self.run_record = make_run_record({
            "schema_version": 2, "mode": "bacteria_id_finetune",
            "condition": condition, "run_seed": run_seed,
            "processed_manifest_sha256": _digest(data.root / "manifest.json"),
            "pretrain_run_sha256": _digest(pretrain_results / "run.json"),
            "pretrain_weights_sha256": _digest(pretrain_weights / "last_model.pt"),
            "seeds": {purpose: purpose_seed("reference30", run_seed, purpose) for purpose in
                      ("finetune_head", "finetune_order", "finetune_augmentation")},
            "config": bacteria_config(), "runtime": runtime_record(device),
            "code_sha256": _code_hashes(),
        })
        if resume_from is None:
            if self.results_dir.exists() or self.weights_dir.exists():
                raise FileExistsError("Finetune run exists; resume explicitly from last.pt")
        else:
            if resume_from.resolve() != self.last_path.resolve() or not resume_from.is_file():
                raise ValueError("Resume file must be this finetune run's last.pt")
            if _read_json(self.results_dir / "run.json") != self.run_record:
                raise ValueError("Finetune run/config/data/code/runtime differs; resume rejected")
        pretrained = load_pretrained(output_dir, data, "reference30", condition, run_seed,
                                     torch.device("cpu"))
        self.model = BacteriaClassifier(pretrained, run_seed).to(device)
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

    def _restore(self) -> None:
        state = torch.load(self.last_path, map_location="cpu", weights_only=True)
        if (state.get("run") != self.run_record or type(state.get("epoch")) is not int
                or not 0 <= state["epoch"] <= self.recipe["max_epochs"]):
            raise ValueError("Finetune checkpoint run or epoch differs")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.order.set_state(state["order"])
        self.augmentation.restore(state["augmentation"])
        self.epoch = state["epoch"]
        self.best_accuracy = state["best_accuracy"]
        self.best_epoch = state["best_epoch"]
        self.early_best = state["early_best"]
        self.early_wait = state["early_wait"]
        self.previous_validation = state["previous_validation"]
        self.history = state["history"]
        self.best_state = state["best_state"]
        if (len(self.history) != self.epoch or self.best_state is None
                or not 1 <= self.best_epoch <= self.epoch):
            raise ValueError("Finetune checkpoint progress is inconsistent")

    def _save_epoch(self) -> None:
        state = {
            "run": self.run_record, "epoch": self.epoch,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "order": self.order.get_state(),
            "augmentation": self.augmentation.state(), "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch, "early_best": self.early_best,
            "early_wait": self.early_wait, "previous_validation": self.previous_validation,
            "history": self.history, "best_state": self.best_state,
        }
        _atomic_torch_save(self.last_path, state)
        _atomic_json(self.results_dir / "training_history.json", self.history)

    def fit(self) -> dict[str, object]:
        if (self.results_dir / "metrics.json").exists():
            raise FileExistsError("Finetune result already completed")
        attempt_started = time.perf_counter()
        batch_size = self.recipe["batch_size"]
        old_matmul = torch.backends.cuda.matmul.fp32_precision
        old_cudnn = torch.backends.cudnn.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            for epoch in range(self.epoch + 1, self.recipe["max_epochs"] + 1):
                if self.early_wait >= self.recipe["early_stopping"]["patience"]:
                    break
                epoch_started = time.perf_counter()
                self.model.train()
                order = torch.randperm(len(self.train_x), generator=self.order)
                loss_sum = 0.0
                for start in range(0, len(order), batch_size):
                    rows = order[start:start + batch_size]
                    clean = self.train_x[rows].to(self.device)
                    with self.augmentation.scope(), torch.autocast("cuda", enabled=False):
                        augmented = self.augmenter(clean.clone())
                    _check_batch(augmented)
                    targets = self.train_y[rows].to(self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", enabled=False):
                        logits = self.model(augmented)
                        loss = nn.functional.cross_entropy(logits, targets)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError(f"Nonfinite finetune loss at epoch {epoch}")
                    loss.backward()
                    self.optimizer.step()
                    loss_sum += float(loss.detach()) * len(rows)
                # SMAE's public Engine calls scheduler once after training with previous val accuracy.
                self.scheduler.step(self.previous_validation)
                accuracy = _accuracy(self.model, self.val_x, self.val_y, self.device)
                self.previous_validation = accuracy
                if accuracy > self.best_accuracy:
                    self.best_accuracy, self.best_epoch = accuracy, epoch
                    self.best_state = {key: value.detach().cpu().clone()
                                       for key, value in self.model.state_dict().items()}
                if accuracy >= self.early_best + self.recipe["early_stopping"]["delta"]:
                    self.early_best, self.early_wait = accuracy, 0
                else:
                    self.early_wait += 1
                self.epoch = epoch
                self.history.append({
                    "epoch": epoch, "train_loss": loss_sum / len(self.train_x),
                    "validation_accuracy": accuracy, "lr_after_scheduler": self.optimizer.param_groups[0]["lr"],
                    "best_epoch": self.best_epoch, "early_wait": self.early_wait,
                    "epoch_seconds": time.perf_counter() - epoch_started,
                })
                self._save_epoch()
                print(f"[Bacteria-ID finetune {self.condition} seed {self.run_seed}] "
                      f"epoch={epoch} val_acc={accuracy:.6f} best_epoch={self.best_epoch}", flush=True)
                if self.early_wait >= self.recipe["early_stopping"]["patience"]:
                    break
        finally:
            torch.backends.cuda.matmul.fp32_precision = old_matmul
            torch.backends.cudnn.fp32_precision = old_cudnn
        if self.best_state is None:
            raise RuntimeError("Finetune did not produce a validation checkpoint")
        self.model.load_state_dict(self.best_state, strict=True)
        _atomic_torch_save(self.best_path, self.best_state)
        test_x, test_y = self.data.rows("test", "test_all")
        test_accuracy = _accuracy(
            self.model, torch.from_numpy(test_x),
            torch.from_numpy(np.array(test_y, dtype=np.int64, copy=True)), self.device,
        )
        result: dict[str, object] = {
            "status": "completed", "condition": self.condition, "run_seed": self.run_seed,
            "epochs": self.epoch, "best_epoch": self.best_epoch,
            "best_validation_accuracy": self.best_accuracy,
            "test_count": len(test_x), "test_accuracy": test_accuracy,
            "test_accuracy_percent": 100.0 * test_accuracy,
            "best_weights_sha256": _digest(self.best_path),
            "processed_manifest_sha256": _digest(self.data.root / "manifest.json"),
            "selection": self.recipe["checkpoint_selection"],
            "training_seconds": sum(float(row["epoch_seconds"]) for row in self.history),
            "current_attempt_seconds": time.perf_counter() - attempt_started,
        }
        _write_json(self.results_dir / "metrics.json", result)
        return result
