"""Synthetic checks for the supplementary global mask-rate comparison."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from chemomae.models.chemo_mae import ChemoMAE
from PIL import Image
from sklearn.metrics import silhouette_samples

from wood_degradation_map.experiments import global_mask_rate_pipeline as pipeline
from wood_degradation_map.experiments import global_mask_rate_reporting as reporting
from wood_degradation_map.experiments import global_mask_rate_training as training
from wood_degradation_map.experiments import neural
from wood_degradation_map.experiments.baselines import NormalizedRepresentation
from wood_degradation_map.experiments.clustering import _diagnose
from wood_degradation_map.experiments.global_clustering import read_label_map
from wood_degradation_map.experiments.global_manifest import GlobalData, create_global_manifest
from wood_degradation_map.experiments.global_manifest import global_config, global_seed
from wood_degradation_map.experiments.global_seed_plan import global_kmeans_plan
from wood_degradation_map.experiments.global_training import (
    GlobalRandomness, GlobalTrainingData, build_global_model,
)
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput
from wood_degradation_map.experiments.manifests import _read_json, _write_json
from wood_degradation_map.experiments.oof_sanity import REPRESENTATIVES


def test_condition_seeds_and_initial_weights_preserve_the_main_experiment() -> None:
    original = global_config()
    config = training.supplement_config()
    assert config.pop("conditions")[:-1] == original["conditions"]
    assert config == {key: value for key, value in original.items() if key != "conditions"}
    assert global_config() == original
    condition = training.supplement_config()["conditions"][-1]
    assert condition["condition_id"] == "M11-75" and condition["n_mask"] == 12
    assert [row["seed"] for row in global_kmeans_plan()] == [1960870825, 322674335, 2051198256]
    reference, target = build_global_model("M11"), training.build_model()
    assert reference.n_mask == 8 and target.n_mask == 12
    assert reference.state_dict().keys() == target.state_dict().keys()
    assert all(torch.equal(value, target.state_dict()[key])
               for key, value in reference.state_dict().items())
    assert original["training"]["epochs"] == 800 and original["sampling"]["q"] == 8192


def test_mask_rate_changes_only_visibility_with_shared_random_streams() -> None:
    clean = torch.randn(8, 256, generator=torch.Generator().manual_seed(8))
    clean -= clean.mean(dim=1, keepdim=True)
    clean /= clean.std(dim=1, keepdim=True)
    reference = GlobalRandomness("M11", torch.device("cpu"))
    target = training.MaskRateRandomness(torch.device("cpu"))
    augmented_50, visible_50 = reference.prepare(clean)
    augmented_75, visible_75 = target.prepare(clean)
    assert torch.equal(augmented_50, augmented_75)
    assert torch.all(visible_50.sum(dim=1) == 128)
    assert torch.all(visible_75.sum(dim=1) == 64)
    assert not (visible_75 & ~visible_50).any()
    assert torch.equal(reference.pixel_order.get_state(), target.pixel_order.get_state())


def test_supplement_smoke_resume_keeps_masks_and_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def small_config() -> dict[str, object]:
        config = global_config()
        config["training"]["batch_size"] = 4
        return config

    def tiny_model() -> ChemoMAE:
        with neural.TorchRandomStream(global_seed("model_init"), torch.device("cpu")).scope():
            return ChemoMAE(seq_len=256, n_patches=16, d_model=8, nhead=2, num_layers=1,
                            dim_feedforward=16, dropout=0, latent_dim=16, latent_normalize=True,
                            decoder_num_layers=1, n_mask=12)

    monkeypatch.setattr(training, "global_config", small_config)
    monkeypatch.setattr(neural, "experiment_config", small_config)
    monkeypatch.setattr(training, "build_model", tiny_model)
    (tmp_path / "manifests").mkdir()
    _write_json(tmp_path / "manifests/complete.json", {"artifact_sha256": {"fixture": "fixed"}})
    spectra = torch.randn(12, 256, generator=torch.Generator().manual_seed(77))
    spectra -= spectra.mean(dim=1, keepdim=True)
    spectra /= spectra.std(dim=1, keepdim=True)
    train = GlobalTrainingData(("fixture",), spectra)
    trainer = training.MaskRateTrainer(train, tmp_path, {"fixture": "M11"},
        device=torch.device("cpu"), smoke_batches=2, smoke_id="fixture")
    trainer.fit()
    reference = {key: value.detach().clone() for key, value in trainer.model.state_dict().items()}
    restored = training.MaskRateTrainer(train, tmp_path, {"fixture": "M11"},
        device=torch.device("cpu"), smoke_batches=2, smoke_id="fixture",
        resume_from=trainer.ckpt_dir / "epoch_1.pt")
    restored.fit()
    for left, right in zip(trainer.trace[2:], restored.trace, strict=True):
        for key in ("train_rows_sha256", "augmented_sha256", "visible_mask_sha256", "lr"):
            assert left[key] == right[key]
    assert all(torch.allclose(value, restored.model.state_dict()[key], atol=1e-6, rtol=0)
               for key, value in reference.items())
    assert restored.completed_epochs == 2 and restored.nonzero_lr_updates > 0


class FixtureRepresentation:
    def transform(self, spectra: np.ndarray) -> NormalizedRepresentation:
        values = spectra[:, :16].copy()
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return NormalizedRepresentation(values, _diagnose(values, 16))


def test_three_seeds_evaluation_roundtrip_and_resume_with_synthetic_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(31)
    samples = []
    coords = np.column_stack((np.arange(64) // 8, np.arange(64) % 8)).astype(np.int32)
    for sample_id in ("KYOw02752", "KYOw02772"):
        path = tmp_path / f"{sample_id}.h5"
        snv = rng.normal(size=(64, 256)).astype(np.float32)
        snv -= snv.mean(axis=1, keepdims=True)
        snv /= snv.std(axis=1, ddof=1, keepdims=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("snv", data=snv)
            handle.create_dataset("pixel_row_col", data=coords)
            handle.create_dataset("valid_spectrum_mask", data=np.ones((8, 8), dtype=np.uint8))
            handle.attrs.update(sample_id=sample_id, saved_pixel_count=64, schema_version=2)
        samples.append(SampleInput(sample_id, path, 8, 8, 64))
    inventory = InputInventory("fixture", tuple(samples), (), 900.0, 2300.0)
    data = GlobalData(inventory, create_global_manifest(inventory, q=32))
    (tmp_path / "manifests").mkdir()
    _write_json(tmp_path / "manifests/complete.json", {"fixture": "fixed"})
    monkeypatch.setattr(pipeline, "training_source", lambda *_: {"fixture": "frozen"})
    monkeypatch.setattr(pipeline, "load_representation", lambda *_: FixtureRepresentation())
    pipeline.cluster(tmp_path, data, inventory, device=torch.device("cpu"), chunk_pixels=17)
    pipeline.cluster(tmp_path, data, inventory, device=torch.device("cpu"), resume=True)
    assert not (tmp_path / "results/global_k8_3seed_v1").exists()
    with pytest.raises(ValueError, match="identity differs"):
        pipeline.load_centers(pipeline.cluster_paths(tmp_path, 1)[1] / "centers_k8.npz",
                              2, torch.device("cpu"))
    pipeline.evaluate(tmp_path, data, inventory, device=torch.device("cpu"), chunk_pixels=19)
    pipeline.check_evaluation(tmp_path, data, inventory)
    rows = _read_json(pipeline.evaluation_path(tmp_path) / "samples.json")["samples"]
    assert len(rows) == 6
    values = []
    for sample in samples:
        with h5py.File(sample.path, "r") as handle:
            values.append(FixtureRepresentation().transform(handle["snv"][:]).values)
    for repeat in (1, 2, 3):
        maps = [read_label_map(pipeline.cluster_paths(tmp_path, repeat)[0]
                              / "maps" / f"{sample.sample_id}.npz", sample) for sample in samples]
        expected = silhouette_samples(np.concatenate(values),
                                      np.concatenate([labels.ravel() for labels in maps]),
                                      metric="cosine")
        for index, sample in enumerate(samples):
            row = next(row for row in rows if row["sample_id"] == sample.sample_id
                       and row["repeat"] == repeat)
            assert row["silhouette"] == pytest.approx(expected[index * 64:(index + 1) * 64].mean(),
                                                       abs=2e-6)
            assert row["used_clusters"] == len(np.unique(maps[index]))
            assert row["maximum_occupancy"] == pytest.approx(max(row["occupancy"]))
    path = pipeline.cluster_paths(tmp_path, 1)[0] / "maps" / f"{samples[0].sample_id}.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        pipeline.check_cluster(tmp_path, data, inventory, 1)


def test_paired_summaries_exclude_undefined_samples_in_both_conditions() -> None:
    rows = {condition: [{"sample_id": sample_id, "repeat": repeat, "undefined_reasons": {},
                        **{metric: repeat / 10 + offset for metric in pipeline.METRICS}}
                       for sample_id in ("a", "b") for repeat in (1, 2, 3)]
            for condition, offset in (("M11", 0.0), ("M11-75", 0.2))}
    rows["M11-75"][-1]["silhouette"] = None
    rows["M11-75"][-1]["undefined_reasons"]["silhouette"] = "single_cluster"
    original = copy.deepcopy(rows)
    table, detail = reporting.comparison_tables(rows)
    selected = table.loc[table.metric == "silhouette"]
    assert selected.common_samples.tolist() == [1, 1, 1]
    assert selected["mean"].tolist() == pytest.approx([0.2, 0.4, 0.2])
    assert detail["silhouette"]["paired_summaries"]["M11"]["excluded_sample_ids"] == ["b"]
    assert rows == original
    with pytest.raises(ValueError, match="sample sets differ"):
        reporting.comparison_tables({**rows, "M11-75": rows["M11-75"][:-1]})


def test_snv_assignment_recovers_permutation_and_handles_empty_clusters() -> None:
    reference = np.eye(8, 256)
    permutation = np.array([4, 2, 1, 0, 3, 6, 7, 5])
    means = np.stack([reference[permutation], reference[permutation]])
    mapping, similarity = reporting.spectral_mapping(reference, means)
    np.testing.assert_array_equal(mapping, np.r_[0, permutation + 1])
    np.testing.assert_allclose(similarity[mapping[1:] - 1, np.arange(8)], 1)
    means[:, 3] = np.nan
    mapping, similarity = reporting.spectral_mapping(reference, means)
    assert sorted(mapping.tolist()) == list(range(9))
    assert np.isnan(similarity[:, 3]).all()


def test_figure_has_two_rows_fixed_columns_and_rejects_different_masks(tmp_path: Path) -> None:
    ids = [sample_id for _, sample_id in REPRESENTATIVES]
    maps = {condition: {sample_id: np.ones((8, 8), dtype=np.uint8) for sample_id in ids}
            for condition in reporting.CONDITIONS}
    path = tmp_path / reporting.FIGURE
    reporting.plot_representatives(maps, path, 120)
    with Image.open(path) as image:
        assert image.size == (210 + 7 * 360 + 8 * 12, 2 * 360 + 3 * 12 + 64 + 180)
        assert image.info["dpi"] == pytest.approx((120, 120), abs=0.1)
    maps["M11-75"][ids[0]][0, 0] = 0
    with pytest.raises(ValueError, match="coverage differs"):
        reporting.plot_representatives(maps, tmp_path / "bad.png", 120)
    assert not (tmp_path / "bad.png").exists()


def test_cli_exposes_only_the_supplement_and_validates_execution_options() -> None:
    path = Path(__file__).resolve().parents[2] / "scripts/experiments/global_mask_rate.py"
    spec = importlib.util.spec_from_file_location("global_mask_rate_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse_args(["cluster", "--resume"]).resume
    assert module.parse_args(["report"]).dpi == 240
    for args in (["train", "--conditions", "M11"], ["check", "--resume"],
                 ["report", "--dpi", "0"], ["cluster", "--chunk-pixels", "0"]):
        with pytest.raises(SystemExit):
            module.parse_args(args)
