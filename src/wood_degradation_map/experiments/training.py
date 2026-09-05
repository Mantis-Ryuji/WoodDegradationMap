"""Extend ChemoMAE Trainer for the fixed CV recipe and explicit epoch-boundary resume."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import torch
from chemomae.training.trainer import Trainer, TrainerConfig

from .config import experiment_config, run_seed
from .data import FoldData
from .manifests import _digest, _read_json, _write_json
from .neural import TrainingRandomness, build_model, build_optimizer, learning_rate, neural_condition


@dataclass(frozen=True)
class TrainingData:
    """The shared train selection in canonical sample/HDF5-row order, held on CPU."""

    fold: int
    sample_ids: tuple[str, ...]
    spectra: torch.Tensor

    @classmethod
    def from_fold(cls, data: FoldData) -> TrainingData:
        # FoldData validates selected spectra and never returns held-out rows here.
        return cls(data.fold, data.train_sample_ids, torch.from_numpy(data.train_matrix()))


def runtime_record(device: torch.device) -> dict[str, object]:
    record = {
        "python": platform.python_version(), "platform": platform.system(),
        "torch": str(torch.__version__), "chemomae": version("chemomae"),
        "numpy": version("numpy"), "h5py": version("h5py"),
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "cudnn_fp32_precision": torch.backends.cudnn.fp32_precision,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record.update({"gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
                       "gpu_capability": list(torch.cuda.get_device_capability(device)),
                       "cudnn_version": torch.backends.cudnn.version()})
    return record


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    names = ("training.py", "neural.py", "config.py", "data.py", "input_validation.py",
             "manifests.py", "baselines.py")
    return {name: _digest(directory / name) for name in names}


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class ExperimentTrainer(Trainer):
    """Use the reference fit/AMP/loss machinery with protocol-specific extensions.

    Production fits always target epoch 800. Smoke fits have two deliberately
    truncated epochs, a separate output tree, and cannot export production weights.
    CPU execution exists only for smoke fixtures; the CLI requires CUDA.
    Instances fit once. Resume creates a new instance with an explicit same-run path.
    """

    def __init__(
        self, train: TrainingData, experiment_dir: Path, condition_id: str, repeat: int,
        *, device: torch.device, resume_from: Path | None = None,
        smoke_batches: int | None = None, smoke_id: str | None = None,
    ) -> None:
        condition = neural_condition(condition_id)
        self.recipe = experiment_config()["training"]
        self.is_smoke = smoke_batches is not None
        if device.type not in ("cpu", "cuda") or (device.type == "cpu" and not self.is_smoke):
            raise ValueError("Production training requires a single CUDA device")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; do not change the fixed batch size/device")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
        if (train.spectra.dtype != torch.float32 or train.spectra.device.type != "cpu"
                or train.spectra.ndim != 2 or train.spectra.shape[1] != 256):
            raise ValueError("TrainingData must hold CPU FP32 spectra with 256 columns")
        self.steps_per_epoch = len(train.spectra) // self.recipe["batch_size"]
        if self.steps_per_epoch < 1:
            raise ValueError("Train selection is smaller than the fixed batch size")
        if self.is_smoke:
            if (type(smoke_batches) is not int or not 1 <= smoke_batches <= self.steps_per_epoch
                    or not smoke_id or any(c not in "abcdefghijklmnopqrstuvwxyz"
                                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in smoke_id)):
                raise ValueError("Smoke requires a positive batch limit and a simple smoke_id")
        elif smoke_id is not None:
            raise ValueError("smoke_id is only valid for smoke runs")
        self.target_epochs = 2 if self.is_smoke else self.recipe["epochs"]
        self.batches_per_epoch = smoke_batches if self.is_smoke else self.steps_per_epoch
        self.randomness = TrainingRandomness(condition_id, train.fold, repeat, device)
        self.train = train
        self.completed_epochs = 0
        self.attempted_updates = 0
        self.optimizer_updates = 0
        self.nonzero_lr_updates = 0
        self._epoch_in_progress = False
        self._failed = False
        self._fit_called = False
        self._epoch_stats: dict[str, object] = {}
        self.trace: list[dict[str, object]] = []

        experiment = experiment_dir.resolve()
        branch = f"neural_smoke/{smoke_id}" if self.is_smoke else "neural"
        suffix = f"{branch}/{condition_id}/fold_{train.fold}/repeat_{repeat}"
        self.results_dir = experiment / "results" / suffix
        weights_dir = experiment / "checkpoints" / suffix
        checkpoint_dir = weights_dir / "checkpoints"  # Preserve Trainer's internal layout.
        manifest_record = _read_json(experiment / "manifests/complete.json")
        self.run_record = {
            "schema_version": 1, "mode": "smoke" if self.is_smoke else "training",
            "smoke_id": smoke_id, "smoke_batches_per_epoch": smoke_batches,
            "condition": condition_id, "fold": train.fold, "repeat": repeat,
            "train_sample_ids": list(train.sample_ids), "train_pixels": len(train.spectra),
            "batch_size": self.recipe["batch_size"], "steps_per_full_epoch": self.steps_per_epoch,
            "planned_production_updates": self.steps_per_epoch * self.recipe["epochs"],
            "config": experiment_config(), "runtime": runtime_record(device),
            "manifest_artifact_sha256": manifest_record["artifact_sha256"],
            "code_sha256": _code_hashes(),
            "seeds": {purpose: run_seed(purpose, train.fold, repeat)
                      for purpose in ("model_init", "pixel_order", "mask", "train_aug")},
            "resume_boundary": "completed epoch; interrupted epoch is replayed",
        }
        if resume_from is None:
            if self.results_dir.exists() or weights_dir.exists():
                raise FileExistsError("Run output exists; resume explicitly from its checkpoint")
        else:
            resume_from = resume_from.resolve()
            if resume_from.parent != checkpoint_dir or not resume_from.is_file():
                raise ValueError("Resume checkpoint must be an existing file inside this run")
            if _read_json(self.results_dir / "run.json") != self.run_record:
                raise ValueError("Run/config/manifest/code/runtime mismatch; cannot resume")

        model = build_model(condition_id, train.fold, repeat).to(device)
        optimizer = build_optimizer(model)
        cfg = TrainerConfig(
            out_dir=weights_dir, device=str(device), amp=True, amp_dtype="fp16",
            enable_tf32=False, grad_clip=None, use_ema=False, loss_type="mse",
            loss_region=condition.loss_region, reduction="mean", resume_from=resume_from,
        )
        super().__init__(model, optimizer, (), scheduler=None, augmenter=self.randomness.augmenter,
                         cfg=cfg)
        self.history_path = self.results_dir / "training_history.json"
        # The checkpoint is authoritative; never accept history ahead of its weights.
        self.history = []
        if resume_from is None:
            self.results_dir.mkdir(parents=True, exist_ok=False)
            _write_json(self.results_dir / "run.json", self.run_record)

    def train_one_epoch(self) -> float:
        if self._failed or self._epoch_in_progress or self.completed_epochs >= self.target_epochs:
            raise RuntimeError("Invalid training position; recreate the trainer for explicit resume")
        self._epoch_in_progress = True
        success = False
        epoch = self.completed_epochs
        self.model.train()
        self.augmenter.train()
        epoch_updates = self.optimizer_updates
        total_loss = 0.0
        first_lr = learning_rate(epoch, 0, self.steps_per_epoch)

        def count_update(optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict) -> None:
            self.optimizer_updates += 1
            if optimizer.param_groups[0]["lr"] > 0:
                self.nonzero_lr_updates += 1

        hook = self.optimizer.register_step_post_hook(count_update)
        try:
            batches = self.randomness.epoch_batches(len(self.train.spectra))
            for batch_index, rows in enumerate(islice(batches, self.batches_per_epoch)):
                lr = learning_rate(epoch, batch_index, self.steps_per_epoch)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                clean = self._to_x(self.train.spectra[rows])
                augmented, visible = self.randomness.prepare(clean)
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_ctx():
                    reconstructed, _, actual_visible = self.model(augmented, visible_mask=visible)
                    loss = self._compute_loss(reconstructed, clean, actual_visible)
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"Nonfinite train loss at epoch={epoch + 1}, batch={batch_index}")
                previous_updates = self.optimizer_updates
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.attempted_updates += 1
                scalar_loss = float(loss.detach())
                total_loss += scalar_loss
                if self.is_smoke:
                    self.trace.append({
                        "epoch": epoch + 1, "batch": batch_index, "lr": lr, "loss": scalar_loss,
                        "optimizer_step": self.optimizer_updates > previous_updates,
                        "amp_scale": self.scaler.get_scale(),
                        "train_rows_sha256": _tensor_digest(rows),
                        "augmented_sha256": _tensor_digest(augmented),
                        "visible_mask_sha256": _tensor_digest(visible),
                    })
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.completed_epochs += 1
            updates = self.optimizer_updates - epoch_updates
            self._epoch_stats = {
                "batches": self.batches_per_epoch, "optimizer_updates": updates,
                "amp_skips": self.batches_per_epoch - updates,
                "first_lr": first_lr, "last_lr": lr, "amp_scale": self.scaler.get_scale(),
                "cumulative_attempted_updates": self.attempted_updates,
                "cumulative_optimizer_updates": self.optimizer_updates,
                "cumulative_nonzero_lr_updates": self.nonzero_lr_updates,
            }
            success = True
            return total_loss / self.batches_per_epoch
        finally:
            hook.remove()
            self._epoch_in_progress = False
            self._failed = not success

    def _save_history(self, rec: dict) -> None:
        super()._save_history({**rec, **self._epoch_stats})

    def _checkpoint_state(self, epoch: int) -> dict:
        if (self._failed or self._epoch_in_progress or epoch != self.completed_epochs
                or len(self.history) != epoch
                or self.attempted_updates != epoch * self.batches_per_epoch):
            raise RuntimeError("Only a completed epoch boundary can be checkpointed")
        state = super()._checkpoint_state(epoch)
        state.update({
            "experiment": self.run_record,
            "progress": {"attempted_updates": self.attempted_updates,
                         "optimizer_updates": self.optimizer_updates,
                         "nonzero_lr_updates": self.nonzero_lr_updates},
            "rng": {"pixel_order": self.randomness.pixel_order.get_state(),
                    "mask": self.randomness.mask.state(),
                    "augmentation": self.randomness.augmentation.state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": (torch.cuda.get_rng_state(self.device)
                                   if self.device.type == "cuda" else None)},
        })
        return state

    def save_checkpoint(self, epoch: int) -> None:
        super().save_checkpoint(epoch)
        if self.is_smoke and epoch == 1:
            self._atomic_torch_save(self._checkpoint_state(epoch), self.ckpt_dir / "epoch_1.pt")
        record = {
            "completed_epochs": epoch, "attempted_updates": self.attempted_updates,
            "optimizer_updates": self.optimizer_updates,
            "amp_skips": self.attempted_updates - self.optimizer_updates,
            "checkpoint_sha256": _digest(self.ckpt_dir / "last.pt"),
        }
        temporary = self.results_dir / "checkpoint.json.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.results_dir / "checkpoint.json")

    def load_checkpoint(self, path: str | Path) -> int:
        if Path(path).resolve().parent != self.ckpt_dir.resolve():
            raise ValueError("Checkpoint belongs to a different run directory")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("experiment") != self.run_record:
            raise ValueError("Checkpoint run/config/manifest/code/runtime mismatch")
        if (state.get("amp") != {"enabled": True, "dtype": "fp16"}
                or state.get("loss_region") != self.cfg.loss_region
                or state.get("scheduler") is not None or state.get("ema") is not None
                or state.get("selection_rule") != "raw_last"):
            raise ValueError("Checkpoint violates the fixed AMP/loss/raw-weight recipe")
        epoch = state.get("epoch")
        progress = state.get("progress", {})
        attempted = progress.get("attempted_updates")
        updates = progress.get("optimizer_updates")
        nonzero = progress.get("nonzero_lr_updates")
        history = state.get("history")
        if (type(epoch) is not int or not 0 <= epoch <= self.target_epochs
                or type(attempted) is not int or attempted != epoch * self.batches_per_epoch
                or type(updates) is not int or not 0 <= updates <= attempted
                or type(nonzero) is not int or not 0 <= nonzero <= updates
                or not isinstance(history, list) or len(history) != epoch
                or [row.get("epoch") for row in history] != list(range(1, epoch + 1))):
            raise ValueError("Checkpoint epoch/history/update counters are inconsistent")
        if any(not math.isfinite(row["train_loss"]) for row in history):
            raise ValueError("Checkpoint contains nonfinite train loss history")
        if history and (
                history[-1].get("cumulative_attempted_updates") != attempted
                or history[-1].get("cumulative_optimizer_updates") != updates
                or history[-1].get("cumulative_nonzero_lr_updates") != nonzero
                or sum(row["optimizer_updates"] for row in history) != updates
                or any(row["batches"] != self.batches_per_epoch
                       or row["amp_skips"] != row["batches"] - row["optimizer_updates"]
                       for row in history)):
            raise ValueError("Checkpoint history disagrees with update counters")
        for value in state["model"].values():
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError("Checkpoint model must contain finite FP32 weights")
        saved_groups = state["optimizer"]["param_groups"]
        if len(saved_groups) != len(self.optimizer.param_groups):
            raise ValueError("Checkpoint optimizer parameter grouping differs")
        for saved, current in zip(saved_groups, self.optimizer.param_groups, strict=True):
            if (len(saved["params"]) != len(current["params"])
                    or any(saved[key] != current[key]
                           for key in ("weight_decay", "betas", "eps", "amsgrad"))):
                raise ValueError("Checkpoint optimizer violates the fixed recipe")
        if (self.scaler.is_enabled() and not state.get("scaler")) or (
                not self.scaler.is_enabled() and state.get("scaler") is not None):
            raise ValueError("Checkpoint GradScaler state is missing or incompatible")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler.is_enabled():
            # Unlike the reference loader, a failed scaler restore must propagate.
            self.scaler.load_state_dict(state["scaler"])
        rng = state["rng"]
        self.randomness.pixel_order.set_state(rng["pixel_order"])
        self.randomness.mask.restore(rng["mask"])
        self.randomness.augmentation.restore(rng["augmentation"])
        torch.set_rng_state(rng["torch_cpu"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(rng["torch_cuda"], self.device)
        self.completed_epochs = epoch
        self.attempted_updates = attempted
        self.optimizer_updates = updates
        self.nonzero_lr_updates = nonzero
        self.history = history
        return epoch + 1

    def save_weights_only(self, filename: str = "last_model.pt") -> None:
        if (self._failed or self.completed_epochs != self.target_epochs
                or len(self.history) != self.target_epochs):
            raise RuntimeError("Final weights require the complete fixed training budget")
        filename = "smoke_model.pt" if self.is_smoke else "last_model.pt"
        super().save_weights_only(filename)
        report = {
            "status": "smoke_fit_completed" if self.is_smoke else "training_completed",
            "completed_epochs": self.completed_epochs,
            "attempted_updates": self.attempted_updates,
            "optimizer_updates": self.optimizer_updates,
            "nonzero_lr_updates": self.nonzero_lr_updates,
            "amp_skips": self.attempted_updates - self.optimizer_updates,
            "training_seconds": sum(row["time_sec"] for row in self.history),
            "weights_file": str(self.out_dir / filename),
            "weights_sha256": _digest(self.out_dir / filename),
            "checkpoint_file": str(self.ckpt_dir / "last.pt"),
        }
        temporary = self.results_dir / "completion.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.results_dir / "completion.json")

    def fit(self, epochs: int | None = None) -> dict:
        if epochs is not None and epochs != self.target_epochs:
            raise ValueError(f"This run has a fixed budget of {self.target_epochs} epochs")
        if self._fit_called or self._failed:
            raise RuntimeError("Create a new trainer to resume explicitly")
        self._fit_called = True
        if self.cfg.resume_from is None:
            self.save_checkpoint(0)
        result = super().fit(self.target_epochs)
        if self.is_smoke:
            result["final_model"] = "smoke_model.pt"
        result["scope"] = "smoke" if self.is_smoke else "training"
        return result
