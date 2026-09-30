"""M11-selected five-fold Bacteria-ID fine-tuning and Table 5 evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol

import numpy as np
import torch
from scipy.stats import t
from sklearn.model_selection import StratifiedKFold
from torch import nn

from .bacteria_config import CONDITIONS, RUN_SEEDS, purpose_seed
from .bacteria_data import PreparedBacteria
from .bacteria_evaluation import TABLE5_REPORTED
from .bacteria_training import (
    BacteriaClassifier, _accuracy, _augmentation, _check_batch, load_pretrained,
)
from .manifests import _digest, _read_json
from .neural import TorchRandomStream
from .records import make_run_record
from .training import runtime_record

if TYPE_CHECKING:
    import optuna


HEAD_LRS = tuple(i / 10_000 for i in range(1, 11))
ENCODER_LRS = tuple(i / 100_000 for i in range(1, 11))
BATCH_SIZES = (8, 16, 32, 64, 128)
AUG_MODES = ((False, False), (True, False), (False, True), (True, True))
WEIGHT_DECAYS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
TOTAL_TRIALS = 100
RANDOM_TRIALS = 30
EPOCHS = 50
HEAD_ONLY_EPOCHS = 5
FOLDS = 5
SPLIT_SEED = 42
SEARCH_SEED = 20260929
SEARCH_RUN_SEED = 0
STUDY_NAME = "bacteria_id_table5_m11_cv_v1"
# Only this pre-cutoff implementation is compatible with early selection.
# Its training, search space, scoring and split logic are unchanged.
EARLY_SELECTION_COMPATIBLE_CODE = (
    "ae71071d7fc9e8d0ce3dfa92443b3223e527eedeb53483533320cb7a56acc74e"
)


class IntegerTrial(Protocol):
    def suggest_int(self, name: str, low: int, high: int) -> int: ...


@dataclass(frozen=True)
class FineTuneSetting:
    k: int
    head_lr: float
    encoder_lr: float | None
    batch_size: int
    tgn: bool
    fs: bool
    weight_decay: float

    def __post_init__(self) -> None:
        if (self.k not in range(9) or self.head_lr not in HEAD_LRS
                or self.batch_size not in BATCH_SIZES
                or (self.tgn, self.fs) not in AUG_MODES
                or self.weight_decay not in WEIGHT_DECAYS
                or (self.encoder_lr is None) != (self.k == 0)
                or (self.encoder_lr is not None and self.encoder_lr not in ENCODER_LRS)):
            raise ValueError("Setting is outside the Bacteria-ID search space")


def suggest_setting(trial: IntegerTrial) -> FineTuneSetting:
    """Suggest ordered candidate indices; encoder LR is inactive at k=0."""
    k = trial.suggest_int("k", 0, 8)
    head_lr = HEAD_LRS[trial.suggest_int("head_lr_index", 0, len(HEAD_LRS) - 1)]
    encoder_lr = (ENCODER_LRS[trial.suggest_int(
        "encoder_lr_index", 0, len(ENCODER_LRS) - 1,
    )] if k else None)
    batch_size = BATCH_SIZES[trial.suggest_int("batch_index", 0, len(BATCH_SIZES) - 1)]
    tgn, fs = AUG_MODES[trial.suggest_int("aug_index", 0, len(AUG_MODES) - 1)]
    weight_decay = WEIGHT_DECAYS[trial.suggest_int(
        "weight_decay_index", 0, len(WEIGHT_DECAYS) - 1,
    )]
    return FineTuneSetting(k, head_lr, encoder_lr, batch_size, tgn, fs, weight_decay)


def setting_from_record(value: object) -> FineTuneSetting:
    if not isinstance(value, dict) or set(value) != set(FineTuneSetting.__dataclass_fields__):
        raise ValueError("Stored HPO setting is incomplete")
    if (type(value["k"]) is not int or type(value["batch_size"]) is not int
            or type(value["tgn"]) is not bool or type(value["fs"]) is not bool):
        raise ValueError("Stored HPO setting has invalid types")
    for name in ("head_lr", "weight_decay"):
        if type(value[name]) not in (int, float):
            raise ValueError(f"Stored HPO {name} has invalid type")
    if value["encoder_lr"] is not None and type(value["encoder_lr"]) not in (int, float):
        raise ValueError("Stored HPO encoder_lr has invalid type")
    return FineTuneSetting(**value)


def _setting_indices(setting: FineTuneSetting) -> dict[str, int]:
    params = {
        "k": setting.k,
        "head_lr_index": HEAD_LRS.index(setting.head_lr),
        "batch_index": BATCH_SIZES.index(setting.batch_size),
        "aug_index": AUG_MODES.index((setting.tgn, setting.fs)),
        "weight_decay_index": WEIGHT_DECAYS.index(setting.weight_decay),
    }
    if setting.encoder_lr is not None:
        params["encoder_lr_index"] = ENCODER_LRS.index(setting.encoder_lr)
    return params


def _unseen_random_setting(seen: set[FineTuneSetting], seed: int) -> FineTuneSetting:
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        k = int(rng.integers(0, 9))
        setting = FineTuneSetting(
            k, HEAD_LRS[int(rng.integers(len(HEAD_LRS)))],
            ENCODER_LRS[int(rng.integers(len(ENCODER_LRS)))] if k else None,
            BATCH_SIZES[int(rng.integers(len(BATCH_SIZES)))],
            *AUG_MODES[int(rng.integers(len(AUG_MODES)))],
            WEIGHT_DECAYS[int(rng.integers(len(WEIGHT_DECAYS)))],
        )
        if setting not in seen:
            return setting
    raise RuntimeError("Could not find an unused Bacteria-ID HPO setting")


@dataclass(frozen=True)
class FineTuneData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    validation_x: torch.Tensor | None = None
    validation_y: torch.Tensor | None = None
    train_rows: tuple[int, ...] | None = None
    validation_rows: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (len(self.train_x) != len(self.train_y)
                or (self.validation_x is None) != (self.validation_y is None)
                or (self.validation_x is not None
                    and len(self.validation_x) != len(self.validation_y))):
            raise ValueError("Fine-tuning spectra and labels do not match")


def _finetune_tensors(data: PreparedBacteria) -> tuple[torch.Tensor, torch.Tensor]:
    spectra = torch.from_numpy(np.array(data.spectra["finetune"], dtype=np.float32, copy=True))
    labels = torch.from_numpy(np.array(data.labels["finetune"], dtype=np.int64, copy=True))
    if spectra.shape != (3000, 1000) or labels.shape != (3000,):
        raise ValueError("Bacteria-ID finetune subset must contain 3000 spectra")
    return spectra, labels


def cv_folds(data: PreparedBacteria) -> tuple[FineTuneData, ...]:
    """Use all five stratified folds of the 3000-spectrum finetune subset."""
    spectra, labels = _finetune_tensors(data)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SPLIT_SEED)
    folds: list[FineTuneData] = []
    for train_rows, validation_rows in splitter.split(
        np.zeros(len(labels)), labels.numpy(),
    ):
        train_idx = torch.from_numpy(train_rows.copy())
        validation_idx = torch.from_numpy(validation_rows.copy())
        if len(train_idx) != 2400 or len(validation_idx) != 600:
            raise ValueError("Unexpected Bacteria-ID five-fold split size")
        folds.append(FineTuneData(
            spectra[train_idx], labels[train_idx],
            spectra[validation_idx], labels[validation_idx],
            tuple(int(row) for row in train_rows),
            tuple(int(row) for row in validation_rows),
        ))
    if len(folds) != FOLDS:
        raise ValueError("Incomplete Bacteria-ID five-fold split")
    return tuple(folds)


def full_finetune_data(data: PreparedBacteria) -> FineTuneData:
    spectra, labels = _finetune_tensors(data)
    return FineTuneData(spectra, labels)


def set_encoder_updates(
    model: BacteriaClassifier, k: int, *, enabled: bool,
) -> None:
    """Keep stem/projection frozen and update only the final k blocks after warmup."""
    if k not in range(9):
        raise ValueError("k must be in 0..8")
    model.encoder.requires_grad_(False)
    for layer in list(model.encoder.encoder.layers)[8 - k:]:
        layer.requires_grad_(enabled)


def classifier_and_optimizer(
    pretrained: nn.Module, run_seed: int, setting: FineTuneSetting,
    *, device: torch.device = torch.device("cpu"),
) -> tuple[BacteriaClassifier, torch.optim.AdamW]:
    model = BacteriaClassifier(pretrained, run_seed).to(device)
    layers = list(model.encoder.encoder.layers)
    if len(layers) != 8:
        raise ValueError("Bacteria-ID partial fine-tuning requires eight encoder blocks")
    set_encoder_updates(model, setting.k, enabled=False)
    head_params = list(model.norm.parameters()) + list(model.head.parameters())
    groups: list[dict[str, object]] = [{"params": head_params, "lr": setting.head_lr}]
    if setting.k:
        groups.append({
            "params": [parameter for layer in layers[8 - setting.k:]
                       for parameter in layer.parameters()],
            "lr": setting.encoder_lr,
        })
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=setting.weight_decay,
    )
    return model, optimizer


@dataclass
class FitResult:
    epoch50_validation_accuracy: float | None
    history: list[dict[str, float | int]]
    final_weights: dict[str, torch.Tensor] | None


def fit_classifier(
    pretrained: nn.Module, train: FineTuneData, setting: FineTuneSetting, run_seed: int,
    device: torch.device, *, epochs: int = EPOCHS, save_final_weights: bool = False,
) -> FitResult:
    """Train fixed epochs; the first five update only the CLS classification head."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model, optimizer = classifier_and_optimizer(pretrained, run_seed, setting, device=device)
    order_stream = torch.Generator(device="cpu").manual_seed(
        purpose_seed("reference30", run_seed, "finetune_order"),
    )
    augmentation_stream = TorchRandomStream(
        purpose_seed("reference30", run_seed, "finetune_augmentation"), device,
    )
    augmenter = _augmentation(0.5 if setting.tgn else 0.0, 0.5 if setting.fs else 0.0)
    history: list[dict[str, float | int]] = []
    old_matmul = torch.backends.cuda.matmul.fp32_precision
    old_cudnn = torch.backends.cudnn.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        for epoch in range(1, epochs + 1):
            if epoch == HEAD_ONLY_EPOCHS + 1:
                set_encoder_updates(model, setting.k, enabled=True)
            started = time.perf_counter()
            model.train()
            order = torch.randperm(len(train.train_x), generator=order_stream)
            loss_sum = 0.0
            for start in range(0, len(order), setting.batch_size):
                rows = order[start:start + setting.batch_size]
                clean = train.train_x[rows].to(device)
                with augmentation_stream.scope(), torch.autocast(device.type, enabled=False):
                    augmented = augmenter(clean.clone())
                _check_batch(augmented)
                targets = train.train_y[rows].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=False):
                    loss = nn.functional.cross_entropy(model(augmented), targets)
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"Nonfinite Bacteria-ID fine-tune loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(rows)
            row: dict[str, float | int] = {
                "epoch": epoch, "train_loss": loss_sum / len(train.train_x),
                "epoch_seconds": time.perf_counter() - started,
            }
            if train.validation_x is not None and train.validation_y is not None:
                row["validation_accuracy"] = _accuracy(
                    model, train.validation_x, train.validation_y, device,
                )
                print(f"[Bacteria-ID HPO] epoch={epoch} "
                      f"val={row['validation_accuracy']:.6f}", flush=True)
            else:
                print(f"[Bacteria-ID HPO] epoch={epoch} "
                      f"train={row['train_loss']:.6f}", flush=True)
            history.append(row)
    finally:
        torch.backends.cuda.matmul.fp32_precision = old_matmul
        torch.backends.cudnn.fp32_precision = old_cudnn
    final_weights = ({key: value.detach().cpu().clone()
                      for key, value in model.state_dict().items()}
                     if save_final_weights else None)
    accuracy = (float(history[-1]["validation_accuracy"])
                if train.validation_x is not None else None)
    return FitResult(accuracy, history, final_weights)


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    """Handle a short-lived Windows reader of an existing generated file."""
    for attempt in range(5):
        try:
            temporary.replace(destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) != 5 or attempt == 4:
                raise
            time.sleep(min(0.5 * 2**attempt, 4.0))


def _replace_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    _replace_with_retry(temporary, path)


def _save_weights(path: Path, weights: dict[str, torch.Tensor]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(weights, temporary)
    _replace_with_retry(temporary, path)


def _study_contract(data: PreparedBacteria, pretrain_dir: Path) -> dict[str, object]:
    weights = pretrain_dir / "checkpoints/pretrain/reference30/M11/seed_0/last_model.pt"
    done = _read_json(pretrain_dir / "results/pretrain/reference30/M11/seed_0/completion.json")
    if (done.get("status") != "training_completed" or done.get("completed_epochs") != 800
            or done.get("weights_sha256") != _digest(weights)):
        raise ValueError("Missing or changed M11 seed-0 reference30 pretraining")
    return {
        "schema_version": 1, "study": STUDY_NAME,
        "processed_manifest_sha256": _digest(data.root / "manifest.json"),
        "m11_pretraining_seed_0_sha256": done["weights_sha256"],
        "code_sha256": _digest(Path(__file__)),
        "torch_version": version("torch"), "chemomae_version": version("chemomae"),
        "optuna_version": version("optuna"),
        "folds": FOLDS, "split_seed": SPLIT_SEED,
        "head_lrs": list(HEAD_LRS), "encoder_lrs": list(ENCODER_LRS),
        "batch_sizes": list(BATCH_SIZES),
        "aug_modes": [list(mode) for mode in AUG_MODES],
        "weight_decays": list(WEIGHT_DECAYS), "k": list(range(9)),
        "trials": TOTAL_TRIALS, "random_trials": RANDOM_TRIALS,
        "epochs": EPOCHS, "head_only_epochs": HEAD_ONLY_EPOCHS,
        "search_seed": SEARCH_SEED, "search_run_seed": SEARCH_RUN_SEED,
        "objective": "M11 seed-0 mean of five validation accuracies at epoch 50",
        "final_training": "all 3000 finetune spectra, same setting and 50 epochs, M00/M11 x five seeds",
        "scheduler": None, "early_stopping": None, "pruner": None,
    }


def _contract_hash(contract: Mapping[str, object]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_directory(
    output_dir: Path, stage: str, trial_number: int, condition: str,
    run_seed: int, fold_index: int | None,
) -> Path:
    if stage == "cv":
        if condition != "M11" or run_seed != SEARCH_RUN_SEED or fold_index not in range(FOLDS):
            raise ValueError("Invalid Bacteria-ID CV run identity")
        return output_dir / "cv" / f"trial_{trial_number:03d}" / f"fold_{fold_index}"
    if stage == "final":
        if condition not in CONDITIONS or run_seed not in RUN_SEEDS or fold_index is not None:
            raise ValueError("Invalid Bacteria-ID final run identity")
        return output_dir / "final" / condition / f"seed_{run_seed}"
    raise ValueError("Unknown Bacteria-ID HPO stage")


def _expected_run(
    data: PreparedBacteria, pretrain_dir: Path, contract: Mapping[str, object],
    stage: str, trial_number: int, condition: str, run_seed: int,
    fold_index: int | None, setting: FineTuneSetting, device: torch.device,
) -> dict[str, object]:
    results = pretrain_dir / "results/pretrain/reference30" / condition / f"seed_{run_seed}"
    weights = pretrain_dir / "checkpoints/pretrain/reference30" / condition / (
        f"seed_{run_seed}/last_model.pt"
    )
    done = _read_json(results / "completion.json")
    if (done.get("status") != "training_completed" or done.get("completed_epochs") != 800
            or done.get("weights_sha256") != _digest(weights)):
        raise ValueError(f"Missing complete reference30 pretraining: {condition} seed {run_seed}")
    return make_run_record({
        "mode": "bacteria_id_cv_finetune", "stage": stage, "trial_number": trial_number,
        "condition": condition, "run_seed": run_seed, "fold_index": fold_index,
        "processed_manifest_sha256": _digest(data.root / "manifest.json"),
        "pretrain_run_sha256": _digest(results / "run.json"),
        "pretrain_weights_sha256": done["weights_sha256"],
        "study_contract_sha256": _contract_hash(contract),
        "seeds": {purpose: purpose_seed("reference30", run_seed, purpose) for purpose in
                  ("finetune_head", "finetune_order", "finetune_augmentation")},
        "config": {
            "setting": asdict(setting), "epochs": EPOCHS,
            "head_only_epochs": HEAD_ONLY_EPOCHS,
            "train_count": 2400 if stage == "cv" else 3000,
            "validation_count": 600 if stage == "cv" else 0,
            "scheduler": None, "early_stopping": None,
            "optimizer": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8,
            "augmentation_strength": "WoodDegradationMap defaults",
            "augmentation_order": "SNV then TGN/FS on train only",
        },
        "runtime": runtime_record(device),
        "code_sha256": {"bacteria_hpo.py": _digest(Path(__file__))},
    })


def run_finetune(
    data: PreparedBacteria, train: FineTuneData, pretrain_dir: Path, output_dir: Path,
    contract: Mapping[str, object], stage: str, trial_number: int, condition: str,
    run_seed: int, fold_index: int | None, setting: FineTuneSetting,
    device: torch.device,
) -> dict[str, object]:
    """Skip a matching completed fold/run; restart incomplete training at epoch one."""
    directory = _run_directory(
        output_dir, stage, trial_number, condition, run_seed, fold_index,
    )
    expected = _expected_run(
        data, pretrain_dir, contract, stage, trial_number, condition, run_seed,
        fold_index, setting, device,
    )
    record_path, metrics_path = directory / "run.json", directory / "metrics.json"
    if record_path.exists():
        if _read_json(record_path) != expected:
            raise ValueError(f"Existing Bacteria-ID run has a different contract: {record_path}")
    else:
        if (directory.exists()
                and {item.name for item in directory.iterdir()} - {"run.json.tmp"}):
            raise FileExistsError(f"Bacteria-ID run exists without contract: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        _replace_json(record_path, expected)
    if metrics_path.exists():
        metrics = _read_json(metrics_path)
        if (metrics.get("status") != "completed" or metrics.get("stage") != stage
                or metrics.get("trial_number") != trial_number
                or metrics.get("condition") != condition or metrics.get("run_seed") != run_seed
                or metrics.get("fold_index") != fold_index
                or metrics.get("completed_epochs") != EPOCHS
                or not (directory / "training_history.json").is_file()):
            raise ValueError(f"Invalid completed Bacteria-ID run: {metrics_path}")
        if stage == "cv":
            score = metrics.get("epoch50_validation_accuracy")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid fold accuracy: {metrics_path}")
        else:
            if metrics.get("final_weights_sha256") != _digest(directory / "last_model.pt"):
                raise ValueError(f"Invalid final checkpoint: {metrics_path}")
        return metrics
    print(f"[Bacteria-ID HPO] {stage} trial={trial_number} {condition} "
          f"seed={run_seed} fold={fold_index} k={setting.k}", flush=True)
    pretrained = load_pretrained(
        pretrain_dir, data, "reference30", condition, run_seed, torch.device("cpu"),
    )
    fit = fit_classifier(
        pretrained, train, setting, run_seed, device, save_final_weights=stage == "final",
    )
    if len(fit.history) != EPOCHS:
        raise RuntimeError("Bacteria-ID fine-tuning did not finish 50 epochs")
    weights_hash: str | None = None
    if fit.final_weights is not None:
        weights_path = directory / "last_model.pt"
        _save_weights(weights_path, fit.final_weights)
        weights_hash = _digest(weights_path)
    _replace_json(directory / "training_history.json", fit.history)
    metrics = {
        "status": "completed", "stage": stage, "trial_number": trial_number,
        "condition": condition, "run_seed": run_seed, "fold_index": fold_index,
        "completed_epochs": EPOCHS,
        "epoch50_validation_accuracy": fit.epoch50_validation_accuracy,
        "final_weights_sha256": weights_hash,
    }
    _replace_json(metrics_path, metrics)
    return metrics


def _study_storage(output_dir: Path) -> str:
    return "sqlite:///" + (output_dir / "study.sqlite3").resolve().as_posix()


def _open_study(
    output_dir: Path, contract: Mapping[str, object], *, create: bool,
    allow_legacy_selection: bool = False,
) -> optuna.study.Study:
    import optuna

    database = output_dir / "study.sqlite3"
    if not create and not database.is_file():
        raise FileNotFoundError(f"Bacteria-ID HPO study is missing: {database}")
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url=_study_storage(output_dir), engine_kwargs={"connect_args": {"timeout": 60}},
    )
    sampler = optuna.samplers.TPESampler(
        seed=SEARCH_SEED, n_startup_trials=RANDOM_TRIALS, multivariate=False,
    )
    if create:
        study = optuna.create_study(
            study_name=STUDY_NAME, direction="maximize", storage=storage,
            load_if_exists=True, sampler=sampler, pruner=optuna.pruners.NopPruner(),
        )
    else:
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage, sampler=sampler)
    existing = study.user_attrs.get("protocol")
    if existing is None:
        if not create or study.trials:
            raise ValueError("Existing HPO study has no matching protocol")
        rng = np.random.default_rng(SEARCH_SEED)
        startup_k = [k for k in range(9) for _ in range(3)]
        startup_k.extend(int(k) for k in rng.choice(9, size=3, replace=False))
        rng.shuffle(startup_k)
        study.set_user_attr("protocol", dict(contract))
        startup_seen: set[FineTuneSetting] = set()
        for k in startup_k:
            while True:
                setting = FineTuneSetting(
                    k, HEAD_LRS[int(rng.integers(len(HEAD_LRS)))],
                    ENCODER_LRS[int(rng.integers(len(ENCODER_LRS)))] if k else None,
                    BATCH_SIZES[int(rng.integers(len(BATCH_SIZES)))],
                    *AUG_MODES[int(rng.integers(len(AUG_MODES)))],
                    WEIGHT_DECAYS[int(rng.integers(len(WEIGHT_DECAYS)))],
                )
                if setting not in startup_seen:
                    break
            startup_seen.add(setting)
            study.enqueue_trial(_setting_indices(setting))
    elif not _matching_study_protocol(
        existing, contract, allow_legacy_selection=allow_legacy_selection,
    ):
        raise ValueError("HPO study data, code, version or protocol changed")
    return study


def _matching_study_protocol(
    existing: object, contract: Mapping[str, object], *, allow_legacy_selection: bool,
) -> bool:
    if existing == dict(contract):
        return True
    return (
        allow_legacy_selection and isinstance(existing, dict)
        and existing.get("code_sha256") == EARLY_SELECTION_COMPATIBLE_CODE
        and {**existing, "code_sha256": contract["code_sha256"]} == dict(contract)
    )


def _completed_trials(study: optuna.study.Study) -> list[optuna.trial.FrozenTrial]:
    import optuna

    completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    seen: set[FineTuneSetting] = set()
    for trial in completed:
        scores = trial.user_attrs.get("fold_accuracies")
        setting = setting_from_record(trial.user_attrs.get("setting"))
        if (setting in seen or not isinstance(scores, list) or len(scores) != FOLDS
                or any(type(score) not in (int, float) or not math.isfinite(score)
                       or not 0 <= score <= 1 for score in scores)
                or trial.value is None or not math.isfinite(trial.value)
                or not math.isclose(trial.value, sum(scores) / FOLDS,
                                    rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"HPO trial {trial.number} has an invalid result")
        seen.add(setting)
    return completed


def _selection_trials(
    study: optuna.study.Study, completed_trials: int | None = None,
) -> list[optuna.trial.FrozenTrial]:
    completed = sorted(_completed_trials(study), key=lambda trial: trial.number)
    if completed_trials is None:
        if len(completed) != TOTAL_TRIALS:
            raise ValueError(
                f"Selection requires {TOTAL_TRIALS} unique completed trials, "
                f"found {len(completed)}",
            )
        return completed
    if type(completed_trials) is not int or not RANDOM_TRIALS <= completed_trials <= TOTAL_TRIALS:
        raise ValueError(f"Selection count must be between {RANDOM_TRIALS} and {TOTAL_TRIALS}")
    if len(completed) < completed_trials:
        raise ValueError(
            f"Selection requires at least {completed_trials} unique completed trials, "
            f"found {len(completed)}",
        )
    return completed[:completed_trials]


def _selection_record(
    study: optuna.study.Study, contract: Mapping[str, object],
    completed_trials: int | None = None,
) -> dict[str, object]:
    trials = _selection_trials(study, completed_trials)
    best = min(trials, key=lambda trial: (-float(trial.value), trial.number))
    selected: dict[str, object] = {
        "schema_version": 1, "study_contract_sha256": _contract_hash(contract),
        "selected_trial_number": best.number, "setting": best.user_attrs["setting"],
        "cv_fold_accuracies": best.user_attrs["fold_accuracies"],
        "cv_mean_accuracy": best.value,
    }
    if completed_trials is not None:
        selected.update(
            schema_version=2,
            selection_code_sha256=_digest(Path(__file__)),
            selection_policy={
                "method": "first_completed_settings",
                "reason": "user_requested_cutoff",
                "planned_settings": TOTAL_TRIALS,
                "completed_settings": completed_trials,
                "trial_numbers": [trial.number for trial in trials],
            },
        )
    return selected


def _save_selection(output_dir: Path, selected: dict[str, object]) -> None:
    path = output_dir / "selected.json"
    if path.exists():
        if _read_json(path) != selected:
            raise ValueError("Existing HPO selection differs; keep the original selection")
    else:
        _replace_json(path, selected)
    print(f"[Bacteria-ID HPO] selected trial {selected['selected_trial_number']}, "
          f"M11 five-fold Accuracy={selected['cv_mean_accuracy']:.6f}", flush=True)


def select_completed(
    data: PreparedBacteria, pretrain_dir: Path, output_dir: Path, *, completed_trials: int,
) -> dict[str, object]:
    """Finalize the first N completed settings without running more CV training."""
    contract = _study_contract(data, pretrain_dir)
    study = _open_study(
        output_dir, contract, create=False, allow_legacy_selection=True,
    )
    # Keep the original study provenance; record the cutoff separately in selected.json.
    contract = study.user_attrs["protocol"]
    selected = _selection_record(study, contract, completed_trials)
    _save_selection(output_dir, selected)
    return selected


def search(
    data: PreparedBacteria, pretrain_dir: Path, output_dir: Path, device: torch.device,
) -> dict[str, object]:
    """Search 100 unique settings by M11 seed-0 epoch-50 five-fold mean Accuracy."""
    import optuna

    selection_path = output_dir / "selected.json"
    if selection_path.exists() and _read_json(selection_path).get("schema_version") == 2:
        raise ValueError("HPO was finalized at a requested cutoff; run train, not search")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Production Bacteria-ID HPO requires one CUDA device")
    contract = _study_contract(data, pretrain_dir)
    study = _open_study(output_dir, contract, create=True)
    for stale in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(stale.number, state=optuna.trial.TrialState.FAIL)
    folds = cv_folds(data)
    _write_once_or_same(
        output_dir / "cv_splits.json",
        json.dumps([{"fold": index, "train": fold.train_rows,
                     "validation": fold.validation_rows}
                    for index, fold in enumerate(folds)],
                   ensure_ascii=False, separators=(",", ":")) + "\n",
    )
    duplicate_streak = 0
    while True:
        completed = _completed_trials(study)
        if len(completed) == TOTAL_TRIALS:
            break
        if len(completed) > TOTAL_TRIALS:
            raise ValueError("HPO study has more than 100 completed settings")
        seen = {setting_from_record(row.user_attrs["setting"]) for row in completed}
        print(f"[Bacteria-ID HPO] unique settings {len(seen)}/{TOTAL_TRIALS}", flush=True)
        trial = study.ask()
        setting = suggest_setting(trial)
        if setting in seen:
            trial.set_user_attr("duplicate_of_setting", asdict(setting))
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            duplicate_streak += 1
            if duplicate_streak >= 10:
                alternative = _unseen_random_setting(
                    seen, SEARCH_SEED + trial.number,
                )
                study.enqueue_trial(_setting_indices(alternative))
                duplicate_streak = 0
            continue
        duplicate_streak = 0
        trial.set_user_attr("setting", asdict(setting))
        scores: list[float] = []
        for fold_index, fold in enumerate(folds):
            metrics = run_finetune(
                data, fold, pretrain_dir, output_dir, contract, "cv", trial.number,
                "M11", SEARCH_RUN_SEED, fold_index, setting, device,
            )
            scores.append(float(metrics["epoch50_validation_accuracy"]))
        trial.set_user_attr("fold_accuracies", scores)
        study.tell(trial, sum(scores) / FOLDS)
    selected = _selection_record(study, contract)
    _save_selection(output_dir, selected)
    return selected


def _selected(
    output_dir: Path, contract: Mapping[str, object],
) -> tuple[dict[str, object], FineTuneSetting, dict[str, object]]:
    selected = _read_json(output_dir / "selected.json")
    cutoff = selected.get("schema_version") == 2
    completed_trials = None
    if cutoff:
        policy = selected.get("selection_policy")
        if not isinstance(policy, dict) or type(policy.get("completed_settings")) is not int:
            raise ValueError("Invalid HPO selection cutoff")
        completed_trials = policy["completed_settings"]
    study = _open_study(
        output_dir, contract, create=False, allow_legacy_selection=cutoff,
    )
    contract = study.user_attrs["protocol"]
    if selected != _selection_record(study, contract, completed_trials):
        raise ValueError("Selected HPO setting differs from the completed M11 study")
    return selected, setting_from_record(selected["setting"]), contract


def train_selected(
    data: PreparedBacteria, pretrain_dir: Path, output_dir: Path, device: torch.device,
) -> None:
    """Fit the shared setting on all 3000 finetune spectra for both conditions."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Production Bacteria-ID final training requires one CUDA device")
    contract = _study_contract(data, pretrain_dir)
    selected, setting, contract = _selected(output_dir, contract)
    train = full_finetune_data(data)
    for condition in CONDITIONS:
        for run_seed in RUN_SEEDS:
            run_finetune(
                data, train, pretrain_dir, output_dir, contract, "final",
                selected["selected_trial_number"], condition, run_seed, None,
                setting, device,
            )


def evaluate_selected(
    data: PreparedBacteria, pretrain_dir: Path, output_dir: Path, device: torch.device,
) -> None:
    """Evaluate test only after all ten fixed-epoch final runs are complete."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Production Bacteria-ID test evaluation requires one CUDA device")
    contract = _study_contract(data, pretrain_dir)
    selected, setting, contract = _selected(output_dir, contract)
    trial_number = selected["selected_trial_number"]
    checkpoints: list[tuple[str, int, Path, str]] = []
    for condition in CONDITIONS:
        for run_seed in RUN_SEEDS:
            directory = _run_directory(
                output_dir, "final", trial_number, condition, run_seed, None,
            )
            expected = _expected_run(
                data, pretrain_dir, contract, "final", trial_number, condition,
                run_seed, None, setting, device,
            )
            if _read_json(directory / "run.json") != expected:
                raise ValueError(f"Final run differs from selected protocol: {directory}")
            metrics = _read_json(directory / "metrics.json")
            weights_path = directory / "last_model.pt"
            weights_hash = _digest(weights_path)
            if (metrics.get("status") != "completed" or metrics.get("stage") != "final"
                    or metrics.get("condition") != condition
                    or metrics.get("run_seed") != run_seed
                    or metrics.get("trial_number") != trial_number
                    or metrics.get("completed_epochs") != EPOCHS
                    or metrics.get("final_weights_sha256") != weights_hash):
                raise ValueError(f"Incomplete final run: {directory}")
            checkpoints.append((condition, run_seed, weights_path, weights_hash))
    test_x, test_y = data.rows("test", "test_all")
    spectra = torch.from_numpy(test_x)
    labels = torch.from_numpy(np.array(test_y, dtype=np.int64))
    for condition, run_seed, weights_path, weights_hash in checkpoints:
        result_dir = output_dir / "evaluation" / condition / f"seed_{run_seed}"
        path = result_dir / "metrics.json"
        if path.exists():
            previous = _read_json(path)
            accuracy = previous.get("test_accuracy")
            percent = previous.get("test_accuracy_percent")
            if (previous.get("status") != "completed"
                    or previous.get("condition") != condition
                    or previous.get("run_seed") != run_seed
                    or previous.get("selected_trial_number") != trial_number
                    or previous.get("final_weights_sha256") != weights_hash
                    or previous.get("study_contract_sha256") != _contract_hash(contract)
                    or previous.get("test_count") != len(spectra)
                    or type(accuracy) not in (int, float)
                    or not math.isfinite(accuracy) or not 0 <= accuracy <= 1
                    or type(percent) not in (int, float)
                    or not math.isclose(percent, 100 * accuracy,
                                        rel_tol=0, abs_tol=1e-10)):
                raise ValueError(f"Existing test result has different provenance: {path}")
            continue
        pretrained = load_pretrained(
            pretrain_dir, data, "reference30", condition, run_seed,
            torch.device("cpu"),
        )
        model, _ = classifier_and_optimizer(pretrained, run_seed, setting, device=device)
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True), strict=True,
        )
        accuracy = _accuracy(model, spectra, labels, device, batch_size=setting.batch_size)
        result_dir.mkdir(parents=True, exist_ok=True)
        _replace_json(path, {
            "status": "completed", "selected_trial_number": trial_number,
            "condition": condition, "run_seed": run_seed,
            "study_contract_sha256": _contract_hash(contract),
            "final_weights_sha256": weights_hash,
            "test_count": len(spectra), "test_accuracy": accuracy,
            "test_accuracy_percent": 100.0 * accuracy,
        })
        print(f"[Bacteria-ID Table 5] {condition} seed {run_seed} "
              f"test Accuracy={100.0 * accuracy:.2f}%", flush=True)


def _write_once_or_same(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Existing report differs; inspect before replacing: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    _replace_with_retry(temporary, path)


def render_table5(
    data: PreparedBacteria, pretrain_dir: Path, output_dir: Path,
) -> tuple[Path, Path]:
    """Append the selected ChemoMAE rows to published SMAE Table 5."""
    contract = _study_contract(data, pretrain_dir)
    selected, setting, contract = _selected(output_dir, contract)
    study = _open_study(output_dir, contract, create=False)
    search_trials = _completed_trials(study)
    table5_details: dict[str, object] = {}
    details: dict[str, object] = {
        "schema_version": 1, "study_contract_sha256": _contract_hash(contract),
        "selected_trial_number": selected["selected_trial_number"],
        "setting": asdict(setting), "selected_m11_cv_mean": selected["cv_mean_accuracy"],
        "selected_m11_cv_folds": selected["cv_fold_accuracies"],
        "search_trials": [{
            "trial_number": trial.number, "setting": trial.user_attrs["setting"],
            "cv_fold_accuracies": trial.user_attrs["fold_accuracies"],
            "cv_mean_accuracy": trial.value,
        } for trial in sorted(search_trials, key=lambda row: row.number)],
        "table5": table5_details,
    }
    selection_policy = selected.get("selection_policy")
    selection_description = (
        "One hundred unique settings were ranked by M11 seed-0 five-fold mean validation "
        "Accuracy at epoch 50. "
    )
    if isinstance(selection_policy, dict):
        selection_count = selection_policy["completed_settings"]
        selection_description = (
            f"The first {selection_count} of {len(search_trials)} unique completed settings "
            "were ranked by M11 seed-0 five-fold mean validation Accuracy at epoch 50. "
            f"The planned {TOTAL_TRIALS}-setting search was stopped early at user request; "
            "later settings and incomplete trials were excluded from selection. "
        )
        details.update(
            schema_version=2, selection_policy=selection_policy,
            selection_code_sha256=selected["selection_code_sha256"],
        )
    lines = [
        "# SMAE Table 5 + ChemoMAE (Bacteria-ID)", "",
        "| Method | Supervised learning Accuracy | w/o pretraining Accuracy | w/ pretraining Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, supervised, without, with_pretrain in TABLE5_REPORTED:
        lines.append(f"| {name} | {supervised} | {without} | {with_pretrain} |")
    for condition in CONDITIONS:
        values: list[float] = []
        for run_seed in RUN_SEEDS:
            row = _read_json(output_dir / "evaluation" / condition / f"seed_{run_seed}/metrics.json")
            weights_path = output_dir / "final" / condition / f"seed_{run_seed}/last_model.pt"
            accuracy = row.get("test_accuracy")
            if (row.get("status") != "completed" or row.get("condition") != condition
                    or row.get("run_seed") != run_seed
                    or row.get("selected_trial_number") != selected["selected_trial_number"]
                    or row.get("study_contract_sha256") != _contract_hash(contract)
                    or row.get("final_weights_sha256") != _digest(weights_path)
                    or row.get("test_count") != 3000
                    or type(accuracy) not in (int, float)
                    or not math.isfinite(accuracy) or not 0 <= accuracy <= 1
                    or type(row.get("test_accuracy_percent")) not in (int, float)
                    or not math.isfinite(row["test_accuracy_percent"])
                    or not math.isclose(row["test_accuracy_percent"], 100 * accuracy,
                                        rel_tol=0, abs_tol=1e-10)):
                raise ValueError(f"Incomplete Table 5 test result: {condition} seed {run_seed}")
            values.append(float(row["test_accuracy_percent"]))
        scores = np.asarray(values, dtype=np.float64)
        mean = float(scores.mean())
        sample_sd = float(scores.std(ddof=1))
        halfwidth = float(t.ppf(0.975, df=4) * sample_sd / math.sqrt(5))
        table5_details[condition] = {
            "run_seeds": list(RUN_SEEDS), "accuracy_percent": values,
            "mean_percent": mean, "sample_sd_percent": sample_sd,
            "ci95_halfwidth_percent": halfwidth,
        }
        lines.append(f"| ChemoMAE({condition}) | — | — | {mean:.2f} ± {halfwidth:.1f} |")
    lines.extend([
        "",
        f"ChemoMAE uses the M11-selected shared setting: k={setting.k}, "
        f"head lr={setting.head_lr:g}, encoder lr="
        f"{setting.encoder_lr if setting.encoder_lr is not None else 'unused'}, "
        f"batch={setting.batch_size}, TGN={'on' if setting.tgn else 'off'}, "
        f"FS={'on' if setting.fs else 'off'}, AdamW weight decay={setting.weight_decay:g}. "
        f"{selection_description}For each condition and five pretraining seeds, the selected "
        "setting was independently trained on all 3000 finetune spectra for 50 epochs; "
        "epochs 1–5 updated only the head and epochs 6–50 also updated the final k blocks. "
        "There was no LR scheduler or early stopping. ChemoMAE Accuracy is five-seed "
        "mean ± t(0.975,4) × sample SD / sqrt(5) in percent. Other rows are values "
        "reported in [published SMAE Table 5]"
        "(https://doi.org/10.1016/j.eswa.2025.128576); their training settings and "
        "interval calculation are not asserted identical.", "",
    ])
    table_path = output_dir / "table5.md"
    statistics_path = output_dir / "table5_statistics.json"
    _write_once_or_same(table_path, "\n".join(lines))
    _write_once_or_same(
        statistics_path,
        json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return table_path, statistics_path
