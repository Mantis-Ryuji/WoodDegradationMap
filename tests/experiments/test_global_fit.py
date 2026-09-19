"""Synthetic global manifests, PCA roundtrip, exact resume and batch orchestration."""

from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from chemomae.models.chemo_mae import ChemoMAE

from wood_degradation_map.experiments import global_manifest as gm
from wood_degradation_map.experiments import global_baselines as gb
from wood_degradation_map.experiments import global_training as gt
from wood_degradation_map.experiments import neural
from wood_degradation_map.experiments.config import ADOPTED_SAMPLE_IDS, RUN_PURPOSES, run_seed
from wood_degradation_map.experiments.global_baselines import GlobalPCA, global_baselines
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput


def inventory_fixture(tmp_path: Path, *, pixels: int = 37) -> InputInventory:
    processed = tmp_path / "production_v1"
    (processed / "samples").mkdir(parents=True)
    rng = np.random.default_rng(937)
    coords = np.column_stack((np.arange(pixels) // 320, np.arange(pixels) % 320)).astype(np.int32)
    height = (pixels + 319) // 320
    mask = np.zeros((height, 320), dtype=np.uint8)
    mask[coords[:, 0], coords[:, 1]] = 1
    samples = []
    for sample_id in ADOPTED_SAMPLE_IDS[:3]:
        path = processed / "samples" / f"{sample_id}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("pixel_row_col", data=coords)
            handle.create_dataset("valid_spectrum_mask", data=mask)
            if pixels < 100:
                spectra = rng.standard_normal((pixels, 256), dtype=np.float32)
                spectra -= spectra.mean(axis=1, keepdims=True)
                spectra /= spectra.std(axis=1, ddof=1, keepdims=True)
                handle.create_dataset("snv", data=spectra)
            handle.attrs.update(sample_id=sample_id, saved_pixel_count=pixels, schema_version=2)
        samples.append(SampleInput(sample_id, path, height, 320, pixels))
    return InputInventory("production_v1", tuple(samples), (), 900.0, 2300.0)


def test_global_seeds_are_fixed_shared_and_separate_from_cv() -> None:
    plan = gm.global_seed_plan(ADOPTED_SAMPLE_IDS)
    assert len(plan["records"]) == 55
    assert len({row["seed"] for row in plan["records"]}) == 55
    assert all("condition" not in row and "fold" not in row for row in plan["records"])
    for purpose in RUN_PURPOSES:
        assert gm.global_seed(purpose) not in {run_seed(purpose, fold, repeat)
                                              for fold in range(1, 6) for repeat in range(1, 4)}
    config = gm.global_config()
    assert "split" not in config and "evaluation" not in config
    assert [row["condition_id"] for row in config["conditions"]] == list(gm.GLOBAL_CONDITIONS)
    assert config["training"]["epochs"] == 800 and config["sampling"]["q"] == 8192
    with pytest.raises(ValueError):
        gm.global_seed("split")


def test_global_selection_is_canonical_and_loader_uses_only_selected_rows(tmp_path: Path) -> None:
    inventory = inventory_fixture(tmp_path)
    plan = gm.create_global_manifest(inventory, q=12)
    other = gm.create_global_manifest(replace(inventory, samples=inventory.samples[::-1]), q=12)
    pd.testing.assert_frame_equal(plan.fit_pixels, other.fit_pixels)
    assert len(plan.fit_pixels) == 36
    for sample in inventory.samples:
        selected = plan.fit_pixels.loc[plan.fit_pixels.sample_id == sample.sample_id, "hdf5_row"]
        with h5py.File(sample.path, "r+") as handle:
            unused = np.setdiff1d(np.arange(37), selected)
            handle["snv"][unused] = np.full((len(unused), 256), np.nan, dtype=np.float32)
    data = gm.GlobalData(inventory, plan)
    assert not hasattr(data, "fold") and data.test_sample_ids == ()
    assert data.train_matrix(chunk_pixels=7).shape == (36, 256)
    assert np.isfinite(data.train_matrix()).all()
    broken = replace(plan, fit_pixels=plan.fit_pixels.copy())
    broken.fit_pixels.loc[0, "hdf5_row"] = broken.fit_pixels.loc[1, "hdf5_row"]
    with pytest.raises(ValueError, match="rows differ"):
        gm.validate_global_manifest(broken, inventory)


def test_global_bundle_roundtrip_and_tamper_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = inventory_fixture(tmp_path, pixels=8192)
    monkeypatch.setattr(gm, "ADOPTED_SAMPLE_IDS", tuple(s.sample_id for s in inventory.samples))
    processed = inventory.samples[0].path.parents[1]
    for name in ("config.json", "manifest.parquet", "sample_quality.parquet"):
        (processed / name).write_bytes(b"synthetic provenance")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("synthetic metadata", encoding="utf-8")
    output = tmp_path / "global"
    saved = gm.create_global_bundle(output, inventory, processed, metadata)
    loaded = gm.load_global_bundle(output, inventory, processed, metadata)
    pd.testing.assert_frame_equal(saved.fit_pixels, loaded.fit_pixels)
    assert "fold" not in loaded.fit_pixels
    with pytest.raises(FileExistsError):
        gm.create_global_bundle(output, inventory, processed, metadata)
    path = output / "manifests/fit_pixels.parquet"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        gm.load_global_bundle(output, inventory, processed, metadata)


def test_global_bundle_rejects_partial_sample_set(tmp_path: Path) -> None:
    inventory = inventory_fixture(tmp_path)
    with pytest.raises(ValueError, match="49 samples"):
        gm.create_global_bundle(tmp_path / "global", inventory, tmp_path, tmp_path / "metadata")
    assert not (tmp_path / "global").exists()


def test_global_pca_roundtrip_provenance_and_existing_output_guard(tmp_path: Path) -> None:
    inventory = inventory_fixture(tmp_path)
    data = gm.GlobalData(inventory, gm.create_global_manifest(inventory, q=12))
    experiment = tmp_path / "global"
    (experiment / "manifests").mkdir(parents=True)
    (experiment / "manifests/complete.json").write_text(
        json.dumps({"artifact_sha256": {"fixture": "manifest"}}), encoding="utf-8",
    )
    before = np.random.get_state()
    report = global_baselines(experiment, data, fit=True)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]
    assert report["pca_fit"]["train_pixel_count"] == 36
    assert "fold" not in report["pca_fit"]
    global_baselines(experiment, data, fit=False)
    with pytest.raises(FileExistsError):
        global_baselines(experiment, data, fit=True)
    pca = GlobalPCA.load(experiment / "checkpoints/baselines/pca.npz")
    np.testing.assert_allclose(pca.estimator.mean_, data.train_matrix().mean(axis=0), atol=1e-7)
    (experiment / "results/baselines/b0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        global_baselines(experiment, data, fit=False)


def test_global_covariance_pca_preserves_layout_and_transform_on_reload(tmp_path: Path) -> None:
    # A tall 3000 x 256 fixture selects covariance_eigh just like production;
    # the existing 36-row fixture exercises a different solver/layout.
    inventory = inventory_fixture(tmp_path, pixels=1000)
    rng = np.random.default_rng(317)
    for sample in inventory.samples:
        spectra = rng.standard_normal((1000, 256), dtype=np.float32)
        spectra -= spectra.mean(axis=1, keepdims=True)
        spectra /= spectra.std(axis=1, ddof=1, keepdims=True)
        with h5py.File(sample.path, "r+") as handle:
            handle.create_dataset("snv", data=spectra)
    data = gm.GlobalData(inventory, gm.create_global_manifest(inventory, q=1000))
    fitted = gb.fit_global_pca(data)
    assert fitted.record.solver == "covariance_eigh"
    assert fitted.estimator.components_.flags.f_contiguous
    fitted.save(tmp_path / "pca.npz")
    restored = GlobalPCA.load(tmp_path / "pca.npz")
    assert restored.estimator.components_.strides == fitted.estimator.components_.strides
    probe = data.train_matrix()[:264]
    np.testing.assert_array_equal(restored.estimator.transform(probe),
                                  fitted.estimator.transform(probe))
    np.testing.assert_array_equal(restored.transform(probe).values, fitted.transform(probe).values)


def test_failed_global_pca_roundtrip_leaves_output_paths_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = inventory_fixture(tmp_path)
    data = gm.GlobalData(inventory, gm.create_global_manifest(inventory, q=12))
    experiment = tmp_path / "global"
    (experiment / "manifests").mkdir(parents=True)
    (experiment / "manifests/complete.json").write_text(
        json.dumps({"artifact_sha256": {"fixture": "manifest"}}), encoding="utf-8",
    )
    original_load = GlobalPCA.load

    def broken_load(path: Path) -> GlobalPCA:
        restored = original_load(path)
        restored.estimator.components_[0] *= -1
        return restored

    with monkeypatch.context() as patch:
        patch.setattr(GlobalPCA, "load", broken_load)
        with pytest.raises(ValueError, match="roundtrip mismatch: maximum absolute error"):
            global_baselines(experiment, data, fit=True)
    assert not (experiment / "results/baselines").exists()
    assert not (experiment / "checkpoints/baselines").exists()
    assert not list(experiment.glob(".baseline-fit-*"))
    global_baselines(experiment, data, fit=True)
    global_baselines(experiment, data, fit=False)


@pytest.fixture
def training_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, gt.GlobalTrainingData]:
    original = gm.global_config

    def fixture_config() -> dict:
        config = original()
        config["training"]["batch_size"] = 4
        return config

    def tiny_model(condition_id: str) -> ChemoMAE:
        with neural.TorchRandomStream(gm.global_seed("model_init"), torch.device("cpu")).scope():
            return ChemoMAE(seq_len=256, n_patches=16, d_model=8, nhead=2, num_layers=1,
                            dim_feedforward=16, dropout=0.0, latent_dim=16, latent_normalize=True,
                            decoder_num_layers=1, n_mask=neural.neural_condition(condition_id).n_mask)

    monkeypatch.setattr(gt, "global_config", fixture_config)
    monkeypatch.setattr(neural, "experiment_config", fixture_config)
    monkeypatch.setattr(gt, "build_global_model", tiny_model)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/complete.json").write_text(
        json.dumps({"artifact_sha256": {"fixture": "manifest"}}), encoding="utf-8",
    )
    spectra = torch.randn(13, 256, generator=torch.Generator().manual_seed(201))
    spectra -= spectra.mean(dim=1, keepdim=True)
    spectra /= spectra.std(dim=1, keepdim=True)
    return tmp_path, gt.GlobalTrainingData(("fixture",), spectra)


@pytest.mark.parametrize("condition", gm.NEURAL_CONDITIONS)
def test_global_training_resume_reproduces_weights_and_random_streams(
    training_fixture: tuple[Path, gt.GlobalTrainingData], condition: str,
) -> None:
    experiment, train = training_fixture
    kwargs = {"device": torch.device("cpu"), "smoke_batches": 2, "smoke_id": "fixture"}
    trainer = gt.GlobalTrainer(train, experiment, condition, **kwargs)
    assert "fold" not in trainer.run_record
    assert trainer.run_record["seeds"]["model_init"] == gm.global_seed("model_init")
    trainer.fit()
    reference = copy.deepcopy(trainer.model.state_dict())
    expected_trace = trainer.trace[2:]
    restored = gt.GlobalTrainer(train, experiment, condition,
                                resume_from=trainer.ckpt_dir / "epoch_1.pt", **kwargs)
    restored.fit()
    assert restored.trace == expected_trace
    assert all(torch.equal(reference[key], value) for key, value in restored.model.state_dict().items())
    assert restored.completed_epochs == 2 and restored.attempted_updates == 4
    with pytest.raises(FileExistsError):
        gt.GlobalTrainer(train, experiment, condition, **kwargs)
    with pytest.raises(ValueError, match="CUDA"):
        gt.GlobalTrainer(train, experiment, condition, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="neural condition"):
        gt.GlobalTrainer(train, experiment, "M10", **kwargs)


def test_global_completion_requires_matching_checkpoint_and_final_weights(
    training_fixture: tuple[Path, gt.GlobalTrainingData],
) -> None:
    experiment, train = training_fixture
    trainer = gt.GlobalTrainer(train, experiment, "M00", device=torch.device("cpu"),
                               smoke_batches=2, smoke_id="fixture")
    # Keep the synthetic two-epoch budget; exercise production exports/checks on CPU.
    trainer.is_smoke = False
    trainer.fit()
    gt.check_global_training(trainer)
    completion = trainer.results_dir / "completion.json"
    record = json.loads(completion.read_text())
    record["completed_epochs"] = 1
    completion.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        gt.check_global_training(trainer)


def cli_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/experiments/global_fit.py"
    spec = importlib.util.spec_from_file_location("global_fit_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_smoke_checks_raw_weights_extraction_and_replayed_epoch(
    training_fixture: tuple[Path, gt.GlobalTrainingData],
) -> None:
    experiment, train = training_fixture
    report = cli_module().smoke_run(train, experiment, "M11", torch.device("cpu"), batches=2)
    assert report["checks_passed"] and report["raw_weights_save_load_exact"]
    assert report["executed_batches_including_replay"] == 6


@pytest.mark.parametrize("change", ["manifest", "code"])
def test_global_resume_rejects_changed_provenance(
    training_fixture: tuple[Path, gt.GlobalTrainingData], monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    experiment, train = training_fixture
    kwargs = {"device": torch.device("cpu"), "smoke_batches": 2, "smoke_id": "fixture"}
    trainer = gt.GlobalTrainer(train, experiment, "M11", **kwargs)
    trainer.save_checkpoint(0)
    checkpoint = trainer.ckpt_dir / "last.pt"
    original = checkpoint.read_bytes()
    if change == "code":
        monkeypatch.setattr(gt, "global_code_hashes", lambda: {"changed": "code"})
    else:
        (experiment / "manifests/complete.json").write_text(
            json.dumps({"artifact_sha256": {"fixture": "changed"}}), encoding="utf-8",
        )
    with pytest.raises(ValueError, match="mismatch"):
        gt.GlobalTrainer(train, experiment, "M11", resume_from=checkpoint, **kwargs)
    assert checkpoint.read_bytes() == original


def test_batch_resume_checks_completed_and_resumes_only_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = cli_module()
    calls, fits, checks = [], [], []
    for condition in ("A0", "M00"):
        checkpoint = tmp_path / f"checkpoints/neural/{condition}/repeat_1/checkpoints/last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture")
    completion = tmp_path / "results/neural/A0/repeat_1/completion.json"
    completion.parent.mkdir(parents=True)
    completion.write_text("{}", encoding="utf-8")

    def factory(train: object, experiment: Path, condition: str, **kwargs: object) -> str:
        calls.append((condition, kwargs["resume_from"]))
        return condition

    def check(trainer: str) -> dict[str, str]:
        checks.append(trainer)
        return {"status": "checked"}

    monkeypatch.setattr(cli, "GlobalTrainer", factory)
    monkeypatch.setattr(cli, "_fit_recorded", fits.append)
    monkeypatch.setattr(cli, "check_global_training", check)
    cli.train_runs(None, tmp_path, ["A0", "M00", "M11"], torch.device("cpu"), resume=True)
    assert fits == ["M00", "M11"] and checks == ["A0", "M00", "M11"]
    assert [path is not None for _, path in calls] == [True, True, False]


def test_cli_defaults_and_batch_failure_stops_next_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = cli_module()
    assert cli.parse_args(["train"]).conditions == ["A0", "M00", "M11"]
    for argv in (["create", "--resume"], ["train", "--smoke-batches", "2"],
                 ["train", "--conditions", "M10"], ["train", "--conditions", "A0", "A0"]):
        with pytest.raises(SystemExit):
            cli.parse_args(argv)
    calls = []

    def factory(train: object, experiment: Path, condition: str, **kwargs: object) -> object:
        calls.append(condition)
        if condition == "M00":
            raise RuntimeError("synthetic failure")
        return object()

    monkeypatch.setattr(cli, "GlobalTrainer", factory)
    monkeypatch.setattr(cli, "_fit_recorded", lambda trainer: None)
    monkeypatch.setattr(cli, "check_global_training", lambda trainer: {"status": "completed"})
    with pytest.raises(RuntimeError, match="synthetic failure"):
        cli.train_runs(None, tmp_path, ["A0", "M00", "M11"], torch.device("cpu"), resume=False)
    assert calls == ["A0", "M00"]
