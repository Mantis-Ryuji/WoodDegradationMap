"""Small CPU fixtures for Bacteria-ID invariants; no real dataset or training runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from chemomae.models.chemo_mae import ChemoMAE
from chemomae.models.losses import masked_mse

from wood_degradation_map.experiments.bacteria_config import bacteria_config, purpose_seed, seed_manifest
from wood_degradation_map.experiments.bacteria_data import _checked_labels, _table4_split, snv_rows
from wood_degradation_map.experiments.bacteria_evaluation import cluster_metrics, extract_latents
from wood_degradation_map.experiments.bacteria_training import BacteriaClassifier, BacteriaRandomness


def _spectra(count: int = 4) -> torch.Tensor:
    values = torch.randn(count, 1000, generator=torch.Generator().manual_seed(19))
    return (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True)


def _tiny_model() -> ChemoMAE:
    return ChemoMAE(
        seq_len=1000, n_patches=20, d_model=256, nhead=8, num_layers=1,
        dim_feedforward=512, dropout=0.0, latent_dim=128,
        latent_normalize=True, decoder_num_layers=1, n_mask=10,
    )


def test_fixed_model_recipe_and_pairing() -> None:
    model = bacteria_config()["model"]
    assert (model["seq_len"], model["n_patches"], model["latent_dim"], model["n_mask"]) == (
        1000, 20, 128, 10,
    )
    seeds = seed_manifest()
    assert len({value for corpus in seeds.values() for run in corpus.values()
                for value in run.values()}) == 3 * 5 * 8
    assert seeds["bacteria4"]["0"]["mask"] == purpose_seed("bacteria4", 0, "mask")


def test_snv_uses_sample_standard_deviation_and_rejects_bad_rows() -> None:
    raw = np.stack((np.arange(1000, dtype=np.float64),
                    np.arange(1000, dtype=np.float64) * 2 + 8))
    transformed = snv_rows(raw, subset="fixture", start=7)
    assert transformed.dtype == np.float32
    np.testing.assert_allclose(transformed.mean(axis=1), 0, atol=1e-6)
    np.testing.assert_allclose(transformed.std(axis=1, ddof=1), 1, atol=1e-6)
    with pytest.raises(ValueError, match="source row 8"):
        snv_rows(np.vstack((raw[:1], np.ones((1, 1000)))), subset="fixture", start=7)


def test_float64_source_labels_must_be_integral(tmp_path: Path) -> None:
    path = tmp_path / "labels.npy"
    labels = np.repeat(np.arange(30), 100).astype(np.float64)
    np.save(path, labels)
    np.testing.assert_array_equal(_checked_labels(path, 3000, 100), labels.astype(np.int64))
    labels[0] = 0.25
    np.save(path, labels)
    with pytest.raises(ValueError, match="exact integers"):
        _checked_labels(path, 3000, 100)


def test_table4_split_is_disjoint_and_per_class_fixed() -> None:
    labels = np.repeat(np.arange(4, dtype=np.int64), 2000)
    train, validation, test = _table4_split(labels, (0, 1, 2, 3))
    assert (len(train), len(validation), len(test)) == (4800, 1600, 1600)
    np.testing.assert_array_equal(np.sort(np.concatenate((train, validation, test))), np.arange(8000))
    for label in range(4):
        assert [np.count_nonzero(labels[part] == label) for part in (train, validation, test)] == [1200, 400, 400]


@pytest.mark.parametrize("condition", ["M00", "M11"])
def test_pretraining_mask_target_and_full_visible_latent(condition: str) -> None:
    clean = _spectra()
    original = clean.clone()
    stream = BacteriaRandomness("bacteria4", condition, 0, torch.device("cpu"))
    before = torch.get_rng_state().clone()
    augmented, visible = stream.prepare(clean)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(clean, original)
    assert augmented.data_ptr() != clean.data_ptr()
    patches = visible.reshape(len(clean), 20, 50)
    assert torch.equal(patches, patches[:, :, :1].expand_as(patches))
    assert bool(((~patches[:, :, 0]).sum(dim=1) == 10).all())
    if condition == "M00":
        assert torch.equal(augmented, clean)
    else:
        torch.testing.assert_close(augmented.mean(dim=1), torch.zeros(len(clean)), atol=1e-6, rtol=0)
        torch.testing.assert_close(augmented.norm(dim=1), clean.norm(dim=1), atol=1e-5, rtol=1e-5)
    model = _tiny_model()
    reconstruction, _, used_visible = model(augmented, visible_mask=visible)
    assert torch.equal(used_visible, visible)
    reference_loss = masked_mse(reconstruction, clean, ~visible, reduction="mean")
    assert bool(torch.isfinite(reference_loss))
    features = extract_latents(model, clean.numpy(), device=torch.device("cpu"), batch_size=3)
    assert features.shape == (len(clean), 128)
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)


def test_classifier_updates_cls_encoder_and_not_projection() -> None:
    model = BacteriaClassifier(_tiny_model(), run_seed=0)
    before_patch = model.encoder.patch_proj.weight.detach().clone()
    before_projection = model.encoder.to_latent.weight.detach().clone()
    assert all(parameter.requires_grad for name, parameter in model.named_parameters()
               if not name.startswith("encoder.to_latent"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    logits = model(_spectra(2))
    assert logits.shape == (2, 30)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before_patch, model.encoder.patch_proj.weight)
    assert torch.equal(before_projection, model.encoder.to_latent.weight)


def test_cluster_metrics_use_hungarian_matching_and_named_normalizers() -> None:
    truth = np.array([0, 0, 1, 1], dtype=np.int64)
    predicted = np.array([1, 1, 0, 0], dtype=np.int64)
    scores = cluster_metrics(truth, predicted, 2)
    assert scores["ACC"] == 100.0
    assert scores["NMI"] == pytest.approx(100.0)
    assert scores["AMI"] == pytest.approx(100.0)
