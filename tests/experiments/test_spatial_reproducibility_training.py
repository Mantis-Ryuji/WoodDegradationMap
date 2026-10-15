"""Small synthetic contract checks for the supplementary encoder repetitions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from wood_degradation_map.experiments import spatial_reproducibility_training as training
from wood_degradation_map.experiments.global_manifest import GlobalData, global_config, global_seed
from wood_degradation_map.experiments.global_seed_plan import global_kmeans_seed
from wood_degradation_map.experiments.global_training import GlobalRandomness, GlobalTrainingData
from wood_degradation_map.experiments.manifests import _digest, _read_json, _write_json
from wood_degradation_map.experiments.records import make_run_record


def test_seeds_preserve_the_original_run_and_share_kmeans_across_encoders() -> None:
    plan = training.seed_plan()
    records = plan["records"]
    assert len({row["seed"] for row in records}) == len(records)
    encoder = [row for row in records if "encoder_repeat" in row]
    assert len(encoder) == 12
    for row in encoder:
        assert row["seed"] == training.encoder_seed(row["purpose"], row["encoder_repeat"])
        if row["encoder_repeat"] == 1:
            assert row["seed"] == global_seed(row["purpose"])
    kmeans = [row for row in records if row["purpose"] == "kmeans"]
    assert [row["seed"] for row in kmeans] == [global_kmeans_seed(k) for k in (1, 2, 3)]
    assert all("encoder_repeat" not in row for row in kmeans)
    assert all(row["seed"] == global_seed("sampling", row["sample_id"])
               for row in records if row["purpose"] == "sampling")


@pytest.mark.parametrize("repeat", [0, 4, True, 2.0])
def test_encoder_repeat_requires_an_explicit_planned_integer(repeat: object) -> None:
    with pytest.raises(ValueError, match="Encoder repeat"):
        training.encoder_seed("model_init", repeat)


def test_seed_plan_rejects_a_new_encoder_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="seed purpose"):
        training.encoder_seed("sampling", 2)
    monkeypatch.setattr(training, "encoder_seed", lambda purpose, repeat: 17)
    with pytest.raises(ValueError, match="seed collision"):
        training.seed_plan()


def test_repeat_one_streams_match_legacy_and_new_streams_are_reproducible() -> None:
    clean = torch.randn(8, 256, generator=torch.Generator().manual_seed(15))
    clean -= clean.mean(dim=1, keepdim=True)
    clean /= clean.std(dim=1, keepdim=True)
    legacy = GlobalRandomness("M11", torch.device("cpu"))
    first = training.SpatialRandomness(1, torch.device("cpu"))
    second = training.SpatialRandomness(2, torch.device("cpu"))
    replay = training.SpatialRandomness(2, torch.device("cpu"))
    expected_augmented, expected_visible = legacy.prepare(clean)
    augmented, visible = first.prepare(clean)
    assert torch.equal(augmented, expected_augmented)
    assert torch.equal(visible, expected_visible)
    second_augmented, second_visible = second.prepare(clean)
    replay_augmented, replay_visible = replay.prepare(clean)
    assert torch.equal(second_augmented, replay_augmented)
    assert torch.equal(second_visible, replay_visible)
    assert not torch.equal(second_visible, visible)
    assert torch.all(second_visible.sum(dim=1) == 128)
    assert torch.equal(legacy.pixel_order.get_state(), first.pixel_order.get_state())
    assert not torch.equal(first.pixel_order.get_state(), second.pixel_order.get_state())


def test_model_initialization_is_repeatable_and_preserves_the_caller_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def small_config() -> dict[str, object]:
        config = global_config()
        config["chemomae"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
        return config

    monkeypatch.setattr(training, "global_config", small_config)
    before = torch.get_rng_state().clone()
    first, replay, second = (training.build_model(repeat) for repeat in (2, 2, 3))
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(value, replay.state_dict()[key])
               for key, value in first.state_dict().items())
    assert any(not torch.equal(value, second.state_dict()[key])
               for key, value in first.state_dict().items())
    assert first.n_mask == second.n_mask == 8


def test_training_preserves_the_numeric_recipe_and_cannot_replace_encoder_one(
    tmp_path: Path,
) -> None:
    original = global_config()
    config = training.supplement_config(2)
    for key in ("training", "chemomae", "augmentation", "sampling", "input", "extraction"):
        assert config[key] == original[key]
    assert "repeat" not in config and config["encoder_repeat"] == 2
    assert [row["condition_id"] for row in config["conditions"]] == ["M11"]
    assert global_config() == original
    with pytest.raises(ValueError, match="Only encoder repeats 2 and 3"):
        training.training_paths(tmp_path, 1)
    for output, source in ((tmp_path, tmp_path), (tmp_path / "child", tmp_path),
                           (tmp_path, tmp_path / "child")):
        with pytest.raises(ValueError, match="disjoint"):
            training._separate_roots(output, source)
    data = cast(GlobalData, SimpleNamespace(train_sample_ids=("fixture",), train_pixel_count=1))
    with pytest.raises(ValueError, match="single CUDA"):
        training.SpatialTrainer(
            GlobalTrainingData(("fixture",), torch.zeros((1, 256))), tmp_path / "new",
            tmp_path / "source", data, 2, device=torch.device("cpu"))
    assert not (tmp_path / "new").exists()


def _completed_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, GlobalData]:
    experiment, source = tmp_path / "supplement", tmp_path / "source"
    (source / "manifests").mkdir(parents=True)
    _write_json(source / "manifests/complete.json", {"artifact_sha256": {"pixels": "fixed"}})
    reference = {"kind": "M11", "weights_sha256": "original", "completed_epochs": 800}
    monkeypatch.setattr(training, "source_record", lambda *args: reference.copy())
    runtime = {"torch": "fixture", "chemomae": "fixture"}
    monkeypatch.setattr(training, "runtime_record", lambda device: runtime.copy())
    data = cast(GlobalData, SimpleNamespace(train_sample_ids=("fixture",), train_pixel_count=2048))
    result, weights = training.training_paths(experiment, 2)
    result.mkdir(parents=True)
    (weights / "checkpoints").mkdir(parents=True)
    (weights / "last_model.pt").write_bytes(b"raw weights fixture")
    (weights / "checkpoints/last.pt").write_bytes(b"checkpoint fixture")
    run = make_run_record({
        **training._training_fields(source, data.train_sample_ids, data.train_pixel_count,
                                    2, reference),
        "mode": "training", "smoke_id": None, "smoke_batches_per_epoch": None,
        "runtime": runtime,
    })
    _write_json(result / "run.json", run)
    completion = {
        "status": "training_completed", "completed_epochs": 800,
        "attempted_updates": 1600, "optimizer_updates": 1599, "nonzero_lr_updates": 1598,
        "amp_skips": 1, "weights_file": str(weights / "last_model.pt"),
        "checkpoint_file": str(weights / "checkpoints/last.pt"),
        "weights_sha256": _digest(weights / "last_model.pt"),
    }
    _write_json(result / "completion.json", completion)
    _write_json(result / "checkpoint.json", {
        **{key: completion[key] for key in ("completed_epochs", "attempted_updates",
                                           "optimizer_updates", "amp_skips")},
        "checkpoint_sha256": _digest(weights / "checkpoints/last.pt"),
    })
    return experiment, source, data


def test_training_source_requires_complete_final_weights_and_unchanged_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, source, data = _completed_fixture(tmp_path, monkeypatch)
    record = training.training_source(experiment, source, data, 2)
    assert record["encoder_repeat"] == 2 and record["completed_epochs"] == 800
    assert training.training_source(experiment, source, data, 1) == record["reference_encoder"]
    _, weights = training.training_paths(experiment, 2)
    (weights / "checkpoints/last.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint/completion differs"):
        training.training_source(experiment, source, data, 2)


@pytest.mark.parametrize("change", ["manifest", "seeds", "epochs", "weights"])
def test_training_source_rejects_changed_provenance_or_incomplete_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    experiment, source, data = _completed_fixture(tmp_path, monkeypatch)
    result, weights = training.training_paths(experiment, 2)
    if change == "manifest":
        (source / "manifests/complete.json").write_text(
            json.dumps({"artifact_sha256": {"pixels": "new"}}), encoding="utf-8")
    elif change == "seeds":
        run = _read_json(result / "run.json")
        run["seeds"]["mask"] += 1
        (result / "run.json").write_text(json.dumps(run), encoding="utf-8")
    elif change == "epochs":
        completion = _read_json(result / "completion.json")
        completion["completed_epochs"] = 799
        (result / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    else:
        (weights / "last_model.pt").write_bytes(b"changed weights")
    with pytest.raises(ValueError):
        training.training_source(experiment, source, data, 2)
