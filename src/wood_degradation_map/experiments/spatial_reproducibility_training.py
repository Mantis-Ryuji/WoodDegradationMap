"""Isolated M11 encoder repeats for the spatial-map reproducibility supplement."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import torch
from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.trainer import Trainer, TrainerConfig

from .cluster_pipeline import NeuralRepresentation
from .clustering import Representation
from .config import ADOPTED_SAMPLE_IDS, ROOT_SEED, _seed
from .global_clustering import load_global_representation, source_record
from .global_manifest import GlobalData, global_config, global_seed_plan
from .global_seed_plan import global_kmeans_plan
from .global_training import GlobalTrainer, GlobalTrainingData, global_code_hashes
from .manifests import _digest, _read_json, _require, _write_json
from .neural import TorchRandomStream, TrainingRandomness, build_augmenter, build_optimizer
from .neural import neural_condition
from .records import make_run_record, matches_run
from .training import runtime_record

CONDITION = "M11"
ENCODER_REPEATS = (1, 2, 3)
TRAINING_PURPOSES = ("model_init", "pixel_order", "mask", "train_aug")


def encoder_seed(purpose: str, encoder_repeat: int) -> int:
    """Derive an encoder stream while preserving the existing repeat-1 seed."""
    _require(purpose in TRAINING_PURPOSES, "Invalid encoder seed purpose")
    _require(type(encoder_repeat) is int and encoder_repeat in ENCODER_REPEATS,
             "Encoder repeat must be 1, 2, or 3")
    return _seed(purpose, "global", encoder_repeat)


def seed_plan() -> dict[str, object]:
    """Record unique random streams; KMeans and sampling are shared across encoders."""
    legacy = global_seed_plan(ADOPTED_SAMPLE_IDS)["records"]
    fixed = [row for row in legacy if row["purpose"] not in (*TRAINING_PURPOSES, "kmeans")]
    encoder = [
        {"purpose": purpose, "encoder_repeat": repeat, "seed": encoder_seed(purpose, repeat)}
        for repeat in ENCODER_REPEATS for purpose in TRAINING_PURPOSES
    ]
    kmeans = [{"purpose": "kmeans", "kmeans_repeat": row["repeat"],
               "K": row["K"], "seed": row["seed"]} for row in global_kmeans_plan()]
    records = [*fixed, *encoder, *kmeans]
    _require(len({row["seed"] for row in records}) == len(records),
             "Spatial reproducibility seed collision")
    return {
        "root_seed": ROOT_SEED, "algorithm": "sha256-json-v1-first-32-bits",
        "scope": "global_spatial_reproducibility", "records": records,
        "collision_check": "unique purpose/identity streams; intentional reuse is listed once",
        "sampling": "reuse the saved source manifest for every encoder",
        "kmeans": "the same three seeds for every encoder",
    }


def training_paths(experiment: Path, encoder_repeat: int) -> tuple[Path, Path]:
    """Return the supplement-only run directories; encoder 1 remains at its source."""
    _require(type(encoder_repeat) is int and encoder_repeat in (2, 3),
             "Only encoder repeats 2 and 3 have supplementary training outputs")
    suffix = f"neural/{CONDITION}/encoder_{encoder_repeat}"
    return experiment / "results" / suffix, experiment / "checkpoints" / suffix


def _separate_roots(experiment: Path, source_experiment: Path) -> tuple[Path, Path]:
    experiment, source_experiment = experiment.resolve(), source_experiment.resolve()
    _require(not experiment.is_relative_to(source_experiment)
             and not source_experiment.is_relative_to(experiment),
             "Supplement output and source experiment must be disjoint directories")
    return experiment, source_experiment


def supplement_config(encoder_repeat: int) -> dict[str, object]:
    """Retain the global numerical recipe with explicit supplement run identity."""
    encoder_seed("model_init", encoder_repeat)
    config = global_config()
    config.pop("repeat")
    config.update({
        "scope": "global_spatial_reproducibility_fit", "encoder_repeat": encoder_repeat,
        "protocol": "docs/design/spatial_map_reproducibility_plan.md",
        "conditions": [row for row in config["conditions"] if row["condition_id"] == CONDITION],
        "seeds": {"root_seed": ROOT_SEED, "algorithm": "sha256-json-v1-first-32-bits",
                  "context": "encoder (purpose, global, e); source sampling is unchanged"},
    })
    return config


def _training_fields(
    source_experiment: Path, sample_ids: tuple[str, ...], pixels: int,
    encoder_repeat: int, reference: dict[str, object],
) -> dict[str, object]:
    config = supplement_config(encoder_repeat)
    steps = pixels // config["training"]["batch_size"]
    return {
        "scope": "global_spatial_reproducibility_fit", "condition": CONDITION,
        "encoder_repeat": encoder_repeat, "source_experiment": str(source_experiment.resolve()),
        "train_sample_ids": list(sample_ids), "train_pixels": pixels,
        "batch_size": config["training"]["batch_size"], "steps_per_full_epoch": steps,
        "planned_production_updates": steps * config["training"]["epochs"],
        "config": config,
        "code_sha256": {**global_code_hashes(), Path(__file__).name: _digest(Path(__file__)),
                        "global_seed_plan.py": _digest(Path(__file__).with_name(
                            "global_seed_plan.py"))},
        "manifest_artifact_sha256": _read_json(
            source_experiment / "manifests/complete.json")["artifact_sha256"],
        "source_manifest_sha256": _digest(source_experiment / "manifests/complete.json"),
        "reference_encoder": reference, "seed_plan": seed_plan(),
        "seeds": {purpose: encoder_seed(purpose, encoder_repeat) for purpose in TRAINING_PURPOSES},
        "interpretation": "descriptive full fit; shared source pixels; not OOF",
    }


def build_model(encoder_repeat: int) -> ChemoMAE:
    """Initialize M11 from the specified isolated CPU random stream."""
    seed = encoder_seed("model_init", encoder_repeat)
    settings = dict(global_config()["chemomae"])
    _require(version("chemomae") == settings.pop("version"), "ChemoMAE version differs")
    settings.pop("initialization")
    _require(torch.get_default_dtype() == torch.float32, "Model initialization requires FP32")
    with TorchRandomStream(seed, torch.device("cpu")).scope(), torch.device("cpu"):
        return ChemoMAE(**settings, n_mask=neural_condition(CONDITION).n_mask)


class SpatialRandomness(TrainingRandomness):
    """Keep the existing training loop with encoder-specific independent streams."""

    def __init__(self, encoder_repeat: int, device: torch.device) -> None:
        self.condition = neural_condition(CONDITION)
        self.mask = TorchRandomStream(encoder_seed("mask", encoder_repeat), device)
        self.augmentation = TorchRandomStream(encoder_seed("train_aug", encoder_repeat), device)
        self.pixel_order = torch.Generator(device="cpu").manual_seed(
            encoder_seed("pixel_order", encoder_repeat))
        self.augmenter = build_augmenter(CONDITION).train()


class SpatialTrainer(GlobalTrainer):
    """Train only the two new encoders using the established epoch-boundary resume loop.

    This separate constructor preserves the hash-bound contracts of the completed
    global fit. There is no model selection, smoke output, or changed fit selection.
    """

    def __init__(
        self, train: GlobalTrainingData, experiment: Path, source_experiment: Path,
        data: GlobalData, encoder_repeat: int, *, device: torch.device,
        resume_from: Path | None = None,
    ) -> None:
        experiment, source_experiment = _separate_roots(experiment, source_experiment)
        self.results_dir, weights = training_paths(experiment, encoder_repeat)
        self.recipe = global_config()["training"]
        self.is_smoke = False
        _require(device.type == "cuda", "Production training requires a single CUDA device")
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        _require(train.spectra.dtype == torch.float32 and train.spectra.device.type == "cpu"
                 and train.spectra.ndim == 2 and train.spectra.shape[1] == 256,
                 "Expected CPU FP32 spectra with 256 columns")
        _require(train.sample_ids == data.train_sample_ids
                 and len(train.spectra) == data.train_pixel_count,
                 "Training matrix differs from the shared source selection")
        self.steps_per_epoch = len(train.spectra) // self.recipe["batch_size"]
        _require(self.steps_per_epoch > 0, "Not enough fit pixels for one batch")
        self.target_epochs = self.recipe["epochs"]
        self.batches_per_epoch = self.steps_per_epoch
        reference = source_record(source_experiment, data, CONDITION)
        self.run_record = make_run_record({
            **_training_fields(source_experiment, train.sample_ids, len(train.spectra),
                               encoder_repeat, reference),
            "mode": "training", "smoke_id": None, "smoke_batches_per_epoch": None,
            "runtime": runtime_record(device),
            "resume_boundary": "completed epoch; interrupted epoch is replayed",
        })
        if resume_from is None:
            if self.results_dir.exists() or weights.exists():
                raise FileExistsError("Spatial encoder output exists; resume explicitly")
        else:
            resume_from = resume_from.resolve()
            _require(resume_from.parent == weights / "checkpoints" and resume_from.is_file(),
                     "Resume checkpoint must be inside this spatial encoder run")
            _require(_read_json(self.results_dir / "run.json") == self.run_record,
                     "Spatial encoder run/config/manifest/code/runtime mismatch; cannot resume")
        self.randomness = SpatialRandomness(encoder_repeat, device)
        self.train = train
        self.completed_epochs = self.attempted_updates = self.optimizer_updates = 0
        self.nonzero_lr_updates = 0
        self._epoch_in_progress = self._failed = self._fit_called = False
        self._epoch_stats: dict[str, object] = {}
        self.trace: list[dict[str, object]] = []
        model = build_model(encoder_repeat).to(device)
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


def training_source(
    experiment: Path, source_experiment: Path, data: GlobalData, encoder_repeat: int,
) -> dict[str, object]:
    """Validate completed raw weights against the fixed source and run contract."""
    encoder_seed("model_init", encoder_repeat)
    experiment, source_experiment = _separate_roots(experiment, source_experiment)
    reference = source_record(source_experiment, data, CONDITION)
    if encoder_repeat == 1:
        return reference
    result, weights = training_paths(experiment, encoder_repeat)
    run, completion = _read_json(result / "run.json"), _read_json(result / "completion.json")
    expected = {**_training_fields(source_experiment, data.train_sample_ids,
                                  data.train_pixel_count, encoder_repeat, reference),
                "schema_version": 2, "mode": "training", "smoke_id": None,
                "smoke_batches_per_epoch": None}
    _require(matches_run(run, expected), "Spatial encoder source contract differs")
    runtime = runtime_record(torch.device("cpu"))
    _require(all(run["contract"]["runtime"][key] == runtime[key]
                 for key in ("torch", "chemomae")), "Source library versions differ")
    _require(completion.get("status") == "training_completed"
             and completion.get("completed_epochs") == global_config()["training"]["epochs"]
             and completion.get("attempted_updates") == expected["planned_production_updates"],
             "Spatial encoder training is incomplete")
    _require(0 < completion["nonzero_lr_updates"] <= completion["optimizer_updates"]
             <= completion["attempted_updates"]
             and completion["amp_skips"] == completion["attempted_updates"]
             - completion["optimizer_updates"], "Invalid training update counters")
    path, checkpoint = weights / "last_model.pt", weights / "checkpoints/last.pt"
    _require(Path(completion["weights_file"]).resolve() == path
             and completion["weights_sha256"] == _digest(path), "Spatial encoder weights changed")
    _require(Path(completion["checkpoint_file"]).resolve() == checkpoint,
             "Spatial encoder checkpoint path changed")
    boundary = _read_json(result / "checkpoint.json")
    _require(all(boundary.get(key) == completion[key] for key in (
        "completed_epochs", "attempted_updates", "optimizer_updates", "amp_skips"))
        and boundary.get("checkpoint_sha256") == _digest(checkpoint),
        "Spatial encoder checkpoint/completion differs")
    return {
        "kind": CONDITION, "encoder_repeat": encoder_repeat,
        "run_sha256": _digest(result / "run.json"),
        "completion_sha256": _digest(result / "completion.json"),
        "weights_sha256": completion["weights_sha256"],
        "completed_epochs": completion["completed_epochs"],
        "source_experiment": str(source_experiment), "reference_encoder": reference,
    }


def load_representation(
    experiment: Path, source_experiment: Path, data: GlobalData, encoder_repeat: int,
    *, device: torch.device,
) -> tuple[Representation, dict[str, object]]:
    """Load a final raw M11 encoder for the existing all-visible FP32 extraction path."""
    source = training_source(experiment, source_experiment, data, encoder_repeat)
    if encoder_repeat == 1:
        representation, legacy = load_global_representation(
            source_experiment, data, CONDITION, device=device)
        _require(legacy == source, "Original encoder source changed during load")
        return representation, source
    _, weights = training_paths(experiment.resolve(), encoder_repeat)
    state = torch.load(weights / "last_model.pt", map_location="cpu", weights_only=True)
    _require(isinstance(state, dict) and bool(state) and all(
        isinstance(value, torch.Tensor) and value.dtype == torch.float32
        and bool(torch.isfinite(value).all()) for value in state.values()),
        "Expected finite raw FP32 spatial encoder weights")
    model = build_model(encoder_repeat)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return NeuralRepresentation(model, device), source
