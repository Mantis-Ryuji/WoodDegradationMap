"""CPU-only integration fixtures for the inherited fit loop and explicit resume."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.trainer import Trainer

from wood_degradation_map.experiments import neural, training
from wood_degradation_map.experiments.config import experiment_config, run_seed
from wood_degradation_map.experiments.training import ExperimentTrainer, TrainingData


@pytest.fixture
def experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, TrainingData]:
    # Only this fixture reduces batch size and model width/depth. Production CLI
    # exposes neither override, and the real GPU smoke retains batch size 1024.
    def fixture_config() -> dict[str, object]:
        config = experiment_config()
        config["training"]["batch_size"] = 4
        return config

    def tiny_builder(condition_id: str, fold: int, repeat: int) -> ChemoMAE:
        stream = neural.TorchRandomStream(run_seed("model_init", fold, repeat), torch.device("cpu"))
        with stream.scope():
            return ChemoMAE(
                seq_len=256, n_patches=16, d_model=8, nhead=2, num_layers=1,
                dim_feedforward=16, dropout=0.0, latent_dim=16,
                latent_normalize=True, decoder_num_layers=1,
                n_mask=neural.neural_condition(condition_id).n_mask,
            )

    monkeypatch.setattr(training, "experiment_config", fixture_config)
    monkeypatch.setattr(neural, "experiment_config", fixture_config)
    monkeypatch.setattr(training, "build_model", tiny_builder)
    directory = tmp_path / "experiment"
    (directory / "manifests").mkdir(parents=True)
    (directory / "manifests/complete.json").write_text(
        json.dumps({"artifact_sha256": {"train_fixture": "fixture-manifest-hash"}}),
        encoding="utf-8",
    )
    spectra = torch.randn(13, 256, generator=torch.Generator().manual_seed(291))
    spectra -= spectra.mean(dim=1, keepdim=True)
    spectra /= spectra.std(dim=1, keepdim=True)
    return directory, TrainingData(1, ("train-only-fixture",), spectra)


def _trainer(
    experiment: tuple[Path, TrainingData], *, resume: Path | None = None,
    condition: str = "M11", repeat: int = 1,
) -> ExperimentTrainer:
    directory, train = experiment
    return ExperimentTrainer(train, directory, condition, repeat, device=torch.device("cpu"),
                             smoke_batches=2, smoke_id="fixture", resume_from=resume)


def _assert_nested_equal(expected: object, actual: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert expected == actual


@pytest.mark.parametrize("condition", ["A0", "M11"])
def test_inherited_fit_recipe_and_resume_reproduce_trajectory(
    experiment: tuple[Path, TrainingData], condition: str,
) -> None:
    trainer = _trainer(experiment, condition=condition)
    clean = experiment[1].spectra.clone()
    assert isinstance(trainer, Trainer)
    assert trainer.cfg.amp_dtype == "fp16" and trainer.cfg.grad_clip is None
    assert trainer.ema is None and trainer.scheduler is None
    assert trainer.cfg.resume_from is None
    result = trainer.fit()
    assert result["scope"] == "smoke" and result["final_model"] == "smoke_model.pt"
    assert trainer.completed_epochs == 2
    assert trainer.attempted_updates == trainer.optimizer_updates == 4
    assert trainer.nonzero_lr_updates == 3  # The first optimizer step has lr=0.
    assert torch.equal(clean, experiment[1].spectra)
    assert not (trainer.out_dir / "last_model.pt").exists()
    expected_model = copy.deepcopy(trainer.model.state_dict())
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    expected_trace = trainer.trace[2:]
    expected_rng = trainer.randomness.pixel_order.get_state()
    assert trainer.trace[0]["lr"] == 0
    assert trainer.trace[1]["lr"] == pytest.approx(6e-4 / (40 * 3))

    # A history write can precede a checkpoint write in Trainer.fit(). The saved
    # checkpoint must win over any incomplete/newer external history on resume.
    trainer.history_path.write_text("unfinished history", encoding="utf-8")
    restored = _trainer(experiment, resume=trainer.ckpt_dir / "epoch_1.pt", condition=condition)
    restored.fit()
    assert restored.trace == expected_trace
    _assert_nested_equal(expected_model, restored.model.state_dict())
    _assert_nested_equal(expected_optimizer, restored.optimizer.state_dict())
    assert torch.equal(expected_rng, restored.randomness.pixel_order.get_state())
    assert [row["epoch"] for row in json.loads(restored.history_path.read_text())] == [1, 2]
    assert [row["cumulative_attempted_updates"] for row in restored.history] == [2, 4]
    assert all(row["amp_skips"] == 0 for row in restored.history)


def test_new_run_never_automatically_resumes_or_overwrites(
    experiment: tuple[Path, TrainingData],
) -> None:
    trainer = _trainer(experiment)
    before = (trainer.results_dir / "run.json").read_bytes()
    with pytest.raises(FileExistsError, match="resume explicitly"):
        _trainer(experiment)
    assert (trainer.results_dir / "run.json").read_bytes() == before
    with pytest.raises(ValueError, match="fixed budget"):
        trainer.fit(1)
    with pytest.raises(RuntimeError, match="complete fixed"):
        trainer.save_weights_only()


def test_partial_epoch_cannot_be_saved_and_resumes_from_last_boundary(
    experiment: tuple[Path, TrainingData], monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(experiment)
    forward = trainer.model.forward
    calls = 0

    def interrupted_forward(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return forward(*args, **kwargs)

    monkeypatch.setattr(trainer.model, "forward", interrupted_forward)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        trainer.fit()
    assert trainer.attempted_updates == 1 and trainer.completed_epochs == 0
    checkpoint = trainer.ckpt_dir / "last.pt"
    state = torch.load(checkpoint, weights_only=True)
    assert state["epoch"] == 0 and state["progress"]["attempted_updates"] == 0
    with pytest.raises(RuntimeError, match="completed epoch"):
        trainer.save_checkpoint(0)
    restored = _trainer(experiment, resume=checkpoint)
    restored.fit()
    assert restored.attempted_updates == restored.optimizer_updates == 4


@pytest.mark.parametrize("change", ["condition", "repeat", "manifest", "code"])
def test_resume_rejects_incompatible_run_before_training(
    experiment: tuple[Path, TrainingData], change: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(experiment)
    trainer.save_checkpoint(0)
    checkpoint = trainer.ckpt_dir / "last.pt"
    if change == "manifest":
        (experiment[0] / "manifests/complete.json").write_text(
            json.dumps({"artifact_sha256": {"train_fixture": "changed"}}), encoding="utf-8",
        )
    elif change == "code":
        monkeypatch.setattr(training, "_code_hashes", lambda: {"training.py": "changed"})
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="run|mismatch"):
        _trainer(experiment, resume=checkpoint,
                 condition="M00" if change == "condition" else "M11",
                 repeat=2 if change == "repeat" else 1)
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("damage", ["progress", "scaler", "weights", "optimizer", "contract"])
def test_incompatible_checkpoint_payload_fails_without_overwrite(
    experiment: tuple[Path, TrainingData], damage: str,
) -> None:
    trainer = _trainer(experiment)
    trainer.save_checkpoint(0)
    checkpoint = trainer.ckpt_dir / "last.pt"
    state = torch.load(checkpoint, weights_only=True)
    if damage == "progress":
        state["progress"]["attempted_updates"] = 1
    elif damage == "scaler":
        state["scaler"] = {"scale": 65536.0}  # CPU fixture must have no CUDA scaler state.
    elif damage == "weights":
        key = next(iter(state["model"]))
        state["model"][key] = state["model"][key].half()
    elif damage == "optimizer":
        state["optimizer"]["param_groups"][0]["weight_decay"] = 0.0
    else:
        state["experiment"]["condition"] = "M00"
    corrupt = trainer.ckpt_dir / "corrupt.pt"
    torch.save(state, corrupt)
    before = checkpoint.read_bytes()
    restored = _trainer(experiment, resume=corrupt)
    with pytest.raises(ValueError):
        restored.fit()
    assert checkpoint.read_bytes() == before


def test_scaler_restore_error_is_not_swallowed(
    experiment: tuple[Path, TrainingData],
) -> None:
    trainer = _trainer(experiment)
    trainer.save_checkpoint(0)
    checkpoint = trainer.ckpt_dir / "last.pt"
    state = torch.load(checkpoint, weights_only=True)
    state["scaler"] = {"scale": 65536.0}
    corrupt = trainer.ckpt_dir / "scaler.pt"
    torch.save(state, corrupt)
    restored = _trainer(experiment, resume=corrupt)

    class FailingScaler:
        def is_enabled(self) -> bool:
            return True

        def load_state_dict(self, state_dict: dict) -> None:
            raise ValueError("synthetic scaler restore failure")

    restored.scaler = FailingScaler()
    with pytest.raises(ValueError, match="synthetic scaler restore failure"):
        restored.fit()


def test_actual_optimizer_hook_counts_skips(
    experiment: tuple[Path, TrainingData],
) -> None:
    trainer = _trainer(experiment)

    class SkippingScaler:
        def is_enabled(self) -> bool:
            return True

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            pass  # Simulate an AMP overflow without calling optimizer.step().

        def update(self) -> None:
            pass

        def get_scale(self) -> float:
            return 1.0

    trainer.scaler = SkippingScaler()
    trainer.train_one_epoch()
    assert trainer.attempted_updates == 2
    assert trainer.optimizer_updates == trainer.nonzero_lr_updates == 0
    assert trainer._epoch_stats["amp_skips"] == 2


def test_production_cannot_fall_back_to_cpu(experiment: tuple[Path, TrainingData]) -> None:
    directory, train = experiment
    with pytest.raises(ValueError, match="CUDA"):
        ExperimentTrainer(train, directory, "A0", 1, device=torch.device("cpu"))
