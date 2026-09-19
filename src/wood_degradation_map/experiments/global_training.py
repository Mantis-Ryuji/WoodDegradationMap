"""Global run construction around the unchanged, checkpointed CV training loop."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch
from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.trainer import Trainer, TrainerConfig

from .global_manifest import NEURAL_CONDITIONS, GlobalData, global_config, global_seed
from .manifests import _digest, _read_json, _require, _write_json
from .neural import TorchRandomStream, TrainingRandomness, build_augmenter, build_optimizer
from .neural import neural_condition
from .records import make_run_record
from .training import ExperimentTrainer, _code_hashes, runtime_record


@dataclass(frozen=True)
class GlobalTrainingData:
    sample_ids: tuple[str, ...]
    spectra: torch.Tensor

    @classmethod
    def from_data(cls, data: GlobalData) -> GlobalTrainingData:
        return cls(data.train_sample_ids, torch.from_numpy(data.train_matrix()))


def build_global_model(condition_id: str) -> ChemoMAE:
    _require(condition_id in NEURAL_CONDITIONS, "Not a global neural condition")
    settings = dict(global_config()["chemomae"])
    _require(version("chemomae") == settings.pop("version"), "ChemoMAE version differs")
    settings.pop("initialization")
    _require(torch.get_default_dtype() == torch.float32, "Model initialization requires FP32")
    with TorchRandomStream(global_seed("model_init"), torch.device("cpu")).scope():
        with torch.device("cpu"):
            return ChemoMAE(**settings, n_mask=neural_condition(condition_id).n_mask)


class GlobalRandomness(TrainingRandomness):
    def __init__(self, condition_id: str, device: torch.device) -> None:
        _require(condition_id in NEURAL_CONDITIONS, "Not a global neural condition")
        self.condition = neural_condition(condition_id)
        self.mask = TorchRandomStream(global_seed("mask"), device)
        self.augmentation = TorchRandomStream(global_seed("train_aug"), device)
        self.pixel_order = torch.Generator(device="cpu").manual_seed(global_seed("pixel_order"))
        self.augmenter = build_augmenter(condition_id).train()


def global_code_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {**_code_hashes(), **{name: _digest(directory / name) for name in (
        "global_manifest.py", "global_training.py",
    )}}


class GlobalTrainer(ExperimentTrainer):
    """Only run construction differs; fit, AMP, loss and resume use the CV loop.

    Keeping the CV modules unchanged preserves the code hashes of completed CV
    artifacts. No fold ID, CV seed or CV output path enters a global run.
    """

    def __init__(
        self, train: GlobalTrainingData, experiment: Path, condition_id: str,
        *, device: torch.device, resume_from: Path | None = None,
        smoke_batches: int | None = None, smoke_id: str | None = None,
    ) -> None:
        _require(condition_id in NEURAL_CONDITIONS, "Not a global neural condition")
        condition = neural_condition(condition_id)
        self.recipe = global_config()["training"]
        self.is_smoke = smoke_batches is not None
        _require(device.type in ("cpu", "cuda") and (device.type == "cuda" or self.is_smoke),
                 "Production training requires a single CUDA device")
        if device.type == "cuda":
            _require(torch.cuda.is_available(), "CUDA is unavailable")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
        _require(train.spectra.dtype == torch.float32 and train.spectra.device.type == "cpu"
                 and train.spectra.ndim == 2 and train.spectra.shape[1] == 256,
                 "GlobalTrainingData requires CPU FP32 spectra with 256 columns")
        self.steps_per_epoch = len(train.spectra) // self.recipe["batch_size"]
        _require(self.steps_per_epoch >= 1, "Not enough global fit pixels for one batch")
        if self.is_smoke:
            _require(type(smoke_batches) is int and 1 <= smoke_batches <= self.steps_per_epoch
                     and bool(smoke_id) and all(c in "abcdefghijklmnopqrstuvwxyz"
                                               "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                                               for c in smoke_id), "Invalid smoke settings")
        else:
            _require(smoke_id is None, "smoke_id is for smoke only")
        self.target_epochs = 2 if self.is_smoke else self.recipe["epochs"]
        self.batches_per_epoch = smoke_batches if self.is_smoke else self.steps_per_epoch
        self.randomness = GlobalRandomness(condition_id, device)
        self.train = train
        self.completed_epochs = self.attempted_updates = self.optimizer_updates = 0
        self.nonzero_lr_updates = 0
        self._epoch_in_progress = self._failed = self._fit_called = False
        self._epoch_stats: dict[str, object] = {}
        self.trace: list[dict[str, object]] = []
        experiment = experiment.resolve()
        branch = f"neural_smoke/{smoke_id}" if self.is_smoke else "neural"
        suffix = f"{branch}/{condition_id}/repeat_1"
        self.results_dir = experiment / "results" / suffix
        weights_dir = experiment / "checkpoints" / suffix
        checkpoint_dir = weights_dir / "checkpoints"
        self.run_record = make_run_record({
            "scope": "global_fit", "mode": "smoke" if self.is_smoke else "training",
            "smoke_id": smoke_id, "smoke_batches_per_epoch": smoke_batches,
            "condition": condition_id, "repeat": 1,
            "train_sample_ids": list(train.sample_ids), "train_pixels": len(train.spectra),
            "batch_size": self.recipe["batch_size"], "steps_per_full_epoch": self.steps_per_epoch,
            "planned_production_updates": self.steps_per_epoch * self.recipe["epochs"],
            "config": global_config(), "runtime": runtime_record(device),
            "manifest_artifact_sha256": _read_json(
                experiment / "manifests/complete.json",
            )["artifact_sha256"],
            "code_sha256": global_code_hashes(),
            "seeds": {purpose: global_seed(purpose)
                      for purpose in ("model_init", "pixel_order", "mask", "train_aug")},
            "resume_boundary": "completed epoch; interrupted epoch is replayed",
        })
        if resume_from is None:
            if self.results_dir.exists() or weights_dir.exists():
                raise FileExistsError("Global run output exists; resume explicitly")
        else:
            resume_from = resume_from.resolve()
            _require(resume_from.parent == checkpoint_dir and resume_from.is_file(),
                     "Resume checkpoint must be an existing file inside this global run")
            _require(_read_json(self.results_dir / "run.json") == self.run_record,
                     "Global run/config/manifest/code/runtime mismatch; cannot resume")
        model = build_global_model(condition_id).to(device)
        optimizer = build_optimizer(model)
        cfg = TrainerConfig(
            out_dir=weights_dir, device=str(device), amp=True, amp_dtype="fp16",
            enable_tf32=False, grad_clip=None, use_ema=False, loss_type="mse",
            loss_region=condition.loss_region, reduction="mean", resume_from=resume_from,
        )
        # Deliberately skip ExperimentTrainer's CV-specific constructor, retaining
        # all its loop/checkpoint methods and the reference Trainer infrastructure.
        Trainer.__init__(self, model, optimizer, (), scheduler=None,
                         augmenter=self.randomness.augmenter, cfg=cfg)
        self.history_path = self.results_dir / "training_history.json"
        self.history = []
        if resume_from is None:
            self.results_dir.mkdir(parents=True, exist_ok=False)
            _write_json(self.results_dir / "run.json", self.run_record)


def check_global_training(trainer: GlobalTrainer) -> dict[str, object]:
    """Check actual epoch-800 checkpoint and raw weights before skipping a run."""
    _require(not trainer.is_smoke, "Smoke is not production completion")
    checkpoint = trainer.ckpt_dir / "last.pt"
    trainer.load_checkpoint(checkpoint)
    report = _read_json(trainer.results_dir / "completion.json")
    weights = trainer.out_dir / "last_model.pt"
    _require(report.get("status") == "training_completed"
             and report.get("completed_epochs") == trainer.target_epochs == trainer.completed_epochs,
             "Global training is incomplete")
    _require(trainer.nonzero_lr_updates > 0, "No effective global optimizer updates")
    for key in ("attempted_updates", "optimizer_updates", "nonzero_lr_updates"):
        _require(report.get(key) == getattr(trainer, key), f"Completion {key} differs")
    _require(report.get("amp_skips") == trainer.attempted_updates - trainer.optimizer_updates
             and report.get("training_seconds") == sum(row["time_sec"] for row in trainer.history),
             "Completion counters or duration differ")
    _require(Path(report["weights_file"]).resolve() == weights.resolve()
             and Path(report["checkpoint_file"]).resolve() == checkpoint.resolve()
             and report.get("weights_sha256") == _digest(weights), "Final weights differ")
    boundary = _read_json(trainer.results_dir / "checkpoint.json")
    _require(boundary.get("checkpoint_sha256") == _digest(checkpoint)
             and boundary.get("completed_epochs") == trainer.completed_epochs
             and boundary.get("attempted_updates") == trainer.attempted_updates
             and boundary.get("optimizer_updates") == trainer.optimizer_updates,
             "Checkpoint record differs")
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    current = trainer.model.state_dict()
    _require(saved.keys() == current.keys() and all(torch.equal(saved[key], value.detach().cpu())
                                                  for key, value in current.items()),
             "Final weights differ from the last checkpoint")
    return report
