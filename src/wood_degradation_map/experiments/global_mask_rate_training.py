"""One supplementary M11-75 fit without changing completed global contracts."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.trainer import Trainer, TrainerConfig

from .global_clustering import source_record
from .global_manifest import GlobalData, global_config, global_seed
from .global_training import GlobalRandomness, GlobalTrainer, GlobalTrainingData, global_code_hashes
from .manifests import _digest, _read_json, _require, _write_json
from .neural import TorchRandomStream, build_optimizer, neural_condition
from .records import make_run_record, matches_run
from .training import runtime_record

CONDITION = "M11-75"
SNAPSHOT = "global_mask_rate_k8_v1"


def supplement_config() -> dict[str, object]:
    """Keep the original bundle immutable; record the added condition in this run."""
    config = global_config()
    config["conditions"] = [*config["conditions"], asdict(neural_condition(CONDITION))]
    return config


def training_paths(experiment: Path, smoke_id: str | None = None) -> tuple[Path, Path]:
    branch = "neural" if smoke_id is None else f"neural_smoke/{smoke_id}"
    suffix = f"{branch}/{CONDITION}/repeat_1"
    return experiment / "results" / suffix, experiment / "checkpoints" / suffix


def training_fields(
    experiment: Path, sample_ids: tuple[str, ...], pixels: int, reference: dict[str, object],
) -> dict[str, object]:
    config = supplement_config()
    steps = pixels // config["training"]["batch_size"]
    return {
        "scope": "global_mask_rate_fit", "condition": CONDITION, "repeat": 1,
        "train_sample_ids": list(sample_ids), "train_pixels": pixels,
        "batch_size": config["training"]["batch_size"], "steps_per_full_epoch": steps,
        "planned_production_updates": steps * config["training"]["epochs"],
        "config": config,
        "code_sha256": {**global_code_hashes(), Path(__file__).name: _digest(Path(__file__))},
        "manifest_artifact_sha256": _read_json(
            experiment / "manifests/complete.json")["artifact_sha256"],
        "reference_M11": reference,
        "seeds": {purpose: global_seed(purpose)
                  for purpose in ("model_init", "pixel_order", "mask", "train_aug")},
        "interpretation": "appendix sensitivity comparison; descriptive full fit, not OOF",
    }


def build_model() -> ChemoMAE:
    settings = dict(global_config()["chemomae"])
    _require(version("chemomae") == settings.pop("version"), "ChemoMAE version differs")
    settings.pop("initialization")
    _require(torch.get_default_dtype() == torch.float32, "Model initialization requires FP32")
    with TorchRandomStream(global_seed("model_init"), torch.device("cpu")).scope():
        with torch.device("cpu"):
            return ChemoMAE(**settings, n_mask=neural_condition(CONDITION).n_mask)


class MaskRateRandomness(GlobalRandomness):
    def __init__(self, device: torch.device) -> None:
        super().__init__("M11", device)
        condition = neural_condition(CONDITION)
        _require(condition.n_mask == 12 and all(
            getattr(condition, key) == getattr(self.condition, key)
            for key in ("loss_region", "noise_prob", "shift_prob", "output_dim")
        ), "M11-75 must differ from M11 only in mask rate and experiment identity")
        self.condition = condition


class MaskRateTrainer(GlobalTrainer):
    """Reuse the complete training/resume loop, with a separate run constructor.

    The original modules are hash-bound to completed M11 artifacts. Keeping this
    constructor separate avoids rewriting their historical reader contracts.
    """

    def __init__(
        self, train: GlobalTrainingData, experiment: Path, reference: dict[str, object],
        *, device: torch.device, resume_from: Path | None = None,
        smoke_batches: int | None = None, smoke_id: str | None = None,
    ) -> None:
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
                 "Expected CPU FP32 spectra with 256 columns")
        self.steps_per_epoch = len(train.spectra) // self.recipe["batch_size"]
        _require(self.steps_per_epoch > 0, "Not enough fit pixels for one batch")
        if self.is_smoke:
            _require(type(smoke_batches) is int and 1 <= smoke_batches <= self.steps_per_epoch
                     and isinstance(smoke_id, str) and bool(smoke_id)
                     and all(c.isascii() and (c.isalnum() or c in "_-") for c in smoke_id),
                     "Invalid smoke settings")
        else:
            _require(smoke_id is None, "smoke_id is for smoke only")
        self.target_epochs = 2 if self.is_smoke else self.recipe["epochs"]
        self.batches_per_epoch = smoke_batches if self.is_smoke else self.steps_per_epoch
        self.randomness = MaskRateRandomness(device)
        self.train = train
        self.completed_epochs = self.attempted_updates = self.optimizer_updates = 0
        self.nonzero_lr_updates = 0
        self._epoch_in_progress = self._failed = self._fit_called = False
        self._epoch_stats: dict[str, object] = {}
        self.trace: list[dict[str, object]] = []
        experiment = experiment.resolve()
        self.results_dir, weights = training_paths(experiment, smoke_id)
        self.run_record = make_run_record({
            **training_fields(experiment, train.sample_ids, len(train.spectra), reference),
            "mode": "smoke" if self.is_smoke else "training", "smoke_id": smoke_id,
            "smoke_batches_per_epoch": smoke_batches, "runtime": runtime_record(device),
            "resume_boundary": "completed epoch; interrupted epoch is replayed",
        })
        if resume_from is None:
            if self.results_dir.exists() or weights.exists():
                raise FileExistsError("M11-75 output exists; resume explicitly")
        else:
            resume_from = resume_from.resolve()
            _require(resume_from.parent == weights / "checkpoints" and resume_from.is_file(),
                     "Resume checkpoint must be inside this M11-75 run")
            _require(_read_json(self.results_dir / "run.json") == self.run_record,
                     "M11-75 run/config/manifest/code/runtime mismatch; cannot resume")
        model = build_model().to(device)
        cfg = TrainerConfig(
            out_dir=weights, device=str(device), amp=True, amp_dtype="fp16",
            enable_tf32=False, grad_clip=None, use_ema=False, loss_type="mse",
            loss_region=neural_condition(CONDITION).loss_region, reduction="mean",
            resume_from=resume_from,
        )
        Trainer.__init__(self, model, build_optimizer(model), (), scheduler=None,
                         augmenter=self.randomness.augmenter, cfg=cfg)
        self.history_path = self.results_dir / "training_history.json"
        self.history = []
        if resume_from is None:
            self.results_dir.mkdir(parents=True, exist_ok=False)
            _write_json(self.results_dir / "run.json", self.run_record)


def training_source(experiment: Path, data: GlobalData) -> dict[str, object]:
    """Validate the final raw weights and their fixed M11 reference without refitting."""
    reference = source_record(experiment, data, "M11")
    result, weights = training_paths(experiment.resolve())
    run, completion = _read_json(result / "run.json"), _read_json(result / "completion.json")
    expected = {**training_fields(experiment, data.train_sample_ids,
                                 data.train_pixel_count, reference),
                "schema_version": 2, "mode": "training", "smoke_id": None,
                "smoke_batches_per_epoch": None}
    _require(matches_run(run, expected), "M11-75 training source contract differs")
    runtime = runtime_record(torch.device("cpu"))
    _require(all(run["contract"]["runtime"][key] == runtime[key]
                 for key in ("torch", "chemomae")), "Source library versions differ")
    _require(completion.get("status") == "training_completed"
             and completion.get("completed_epochs") == global_config()["training"]["epochs"]
             and completion.get("attempted_updates") == expected["planned_production_updates"],
             "M11-75 training is incomplete")
    _require(0 < completion["nonzero_lr_updates"] <= completion["optimizer_updates"]
             <= completion["attempted_updates"]
             and completion["amp_skips"] == completion["attempted_updates"]
             - completion["optimizer_updates"], "Invalid training update counters")
    path = weights / "last_model.pt"
    _require(Path(completion["weights_file"]).resolve() == path
             and completion["weights_sha256"] == _digest(path), "M11-75 weights changed")
    boundary = _read_json(result / "checkpoint.json")
    _require(all(boundary.get(key) == completion[key] for key in (
        "completed_epochs", "attempted_updates", "optimizer_updates")),
        "Checkpoint/completion counters differ")
    return {"condition": CONDITION, "run_sha256": _digest(result / "run.json"),
            "completion_sha256": _digest(result / "completion.json"),
            "weights_sha256": completion["weights_sha256"]}
