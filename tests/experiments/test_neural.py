"""CPU fixtures for the neural recipe; no real data, CUDA training, or file output."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch
from chemomae.models.chemo_mae import ChemoMAE

from wood_degradation_map.experiments.baselines import RepresentationError
from wood_degradation_map.experiments.config import CONDITIONS, Condition, experiment_config
from wood_degradation_map.experiments.neural import (
    TorchRandomStream,
    TrainingRandomness,
    build_augmenter,
    build_model,
    build_optimizer,
    extract_full_visible,
    fp32_inference,
    learning_rate,
    neural_condition,
    reconstruction_loss,
)


@pytest.fixture
def spectra() -> torch.Tensor:
    values = torch.randn(8, 256, generator=torch.Generator().manual_seed(41))
    values -= values.mean(dim=1, keepdim=True)
    return values / values.std(dim=1, keepdim=True)


@pytest.fixture
def tiny_model() -> ChemoMAE:
    # Reduced transformer width/depth only in this test fixture, never in the
    # production builder. Keep the real input, patch, latent and decoder contract.
    with TorchRandomStream(11, torch.device("cpu")).scope():
        return ChemoMAE(
            seq_len=256, n_patches=16, d_model=16, nhead=4, num_layers=1,
            dim_feedforward=32, dropout=0.0, latent_dim=16,
            latent_normalize=True, decoder_num_layers=1, n_mask=8,
        )


def test_fixed_model_initialization_is_shared_and_rng_is_restored() -> None:
    before = torch.get_rng_state().clone()
    mae = build_model("M11", 1, 1)
    ae = build_model("A0", 1, 1)
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in mae.parameters()) == 6_335_504
    assert mae.n_mask == 8 and ae.n_mask == 0
    assert isinstance(mae.decoder.net, torch.nn.Linear)
    assert len(mae.encoder.encoder.layers) == 8
    assert mae.encoder.encoder.layers[0].self_attn.num_heads == 8
    for key, value in mae.state_dict().items():
        assert value.dtype == torch.float32
        assert torch.equal(value, ae.state_dict()[key])
    layers = mae.encoder.encoder.layers
    assert torch.equal(layers[0].linear1.weight, layers[1].linear1.weight)
    assert layers[0].linear1.weight is not layers[1].linear1.weight
    assert all(m.p == 0 for m in mae.modules() if isinstance(m, torch.nn.Dropout))


@pytest.mark.parametrize("condition_id", ["B0", "B1", "unknown"])
def test_non_neural_conditions_are_rejected(condition_id: str) -> None:
    with pytest.raises(ValueError, match="neural condition"):
        neural_condition(condition_id)


@pytest.mark.parametrize("condition", [c for c in CONDITIONS if c.n_mask is not None])
def test_condition_masks_augmentation_and_clean_target(
    condition: Condition, spectra: torch.Tensor,
) -> None:
    settings = asdict(build_augmenter(condition.condition_id).config)
    assert settings["noise_prob"] == condition.noise_prob
    assert settings["shift_prob"] == condition.shift_prob
    for key, expected in experiment_config()["augmentation"].items():
        assert settings[key] == (tuple(expected) if isinstance(expected, list) else expected)
    stream = TrainingRandomness(condition.condition_id, 1, 1, torch.device("cpu"))
    original = spectra.clone()
    before = torch.get_rng_state().clone()
    augmented, visible = stream.prepare(spectra)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(spectra, original)
    assert augmented.data_ptr() != spectra.data_ptr()
    assert augmented.dtype == torch.float32
    assert visible.dtype == torch.bool
    patches = visible.reshape(len(spectra), 16, 16)
    assert torch.equal(patches, patches[:, :, :1].expand_as(patches))
    assert bool(((~patches[:, :, 0]).sum(dim=1) == condition.n_mask).all())
    if condition.noise_prob == condition.shift_prob == 0:
        assert torch.equal(augmented, spectra)
    else:
        torch.testing.assert_close(augmented.mean(dim=1), torch.zeros(8), atol=1e-6, rtol=0)
        torch.testing.assert_close(augmented.norm(dim=1), spectra.norm(dim=1))


def test_mask_and_pixel_order_do_not_depend_on_aug_consumption(spectra: torch.Tensor) -> None:
    left = TrainingRandomness("M00", 1, 1, torch.device("cpu"))
    right = TrainingRandomness("M11", 1, 1, torch.device("cpu"))
    for _ in range(2):
        with right.augmentation.scope():
            torch.rand(97)
        assert torch.equal(left.prepare(spectra)[1], right.prepare(spectra)[1])
        left_batches = list(left.epoch_batches(2 * 1024 + 19))
        right_batches = list(right.epoch_batches(2 * 1024 + 19))
        assert len(left_batches) == 2
        assert all(len(batch) == 1024 for batch in left_batches)
        assert all(torch.equal(a, b) for a, b in zip(left_batches, right_batches, strict=True))
        order = torch.cat(left_batches)
        assert len(order.unique()) == 2048 and int(order.max()) < 2067
    with pytest.raises(ValueError, match="at least one"):
        list(left.epoch_batches(1023))


def test_stream_replay_and_exception_restore() -> None:
    stream = TorchRandomStream(13, torch.device("cpu"))
    snapshot = stream.state()
    outside = torch.get_rng_state().clone()
    with stream.scope():
        expected = torch.rand(5)
    stream.restore(snapshot)
    with pytest.raises(RuntimeError, match="intentional"):
        with stream.scope():
            assert torch.equal(torch.rand(5), expected)
            raise RuntimeError("intentional")
    assert torch.equal(outside, torch.get_rng_state())
    assert not torch.equal(snapshot, stream.state())


def test_all_three_rng_states_can_replay_next_epoch_and_batch(spectra: torch.Tensor) -> None:
    stream = TrainingRandomness("M11", 2, 3, torch.device("cpu"))
    stream.prepare(spectra)
    list(stream.epoch_batches(2067))
    mask_state, aug_state = stream.mask.state(), stream.augmentation.state()
    pixel_state = stream.pixel_order.get_state()
    expected_order = list(stream.epoch_batches(2067))
    expected_input, expected_mask = stream.prepare(spectra)
    stream.mask.restore(mask_state)
    stream.augmentation.restore(aug_state)
    stream.pixel_order.set_state(pixel_state)
    actual_order = list(stream.epoch_batches(2067))
    actual_input, actual_mask = stream.prepare(spectra)
    assert all(torch.equal(a, b) for a, b in zip(expected_order, actual_order, strict=True))
    assert torch.equal(expected_input, actual_input)
    assert torch.equal(expected_mask, actual_mask)


@pytest.mark.parametrize("condition_id", ["A0", "M00", "M11-25", "M11-75"])
def test_loss_uses_clean_target_and_correct_region(condition_id: str, spectra: torch.Tensor) -> None:
    stream = TrainingRandomness(condition_id, 1, 1, torch.device("cpu"))
    _, visible = stream.prepare(spectra)
    reconstruction = (spectra + torch.where(visible, 7.0, 2.0)).detach().requires_grad_()
    loss = reconstruction_loss(condition_id, reconstruction, spectra, visible)
    assert float(loss.detach()) == pytest.approx(49.0 if condition_id == "A0" else 4.0)
    loss.backward()
    if condition_id != "A0":
        assert bool((reconstruction.grad[visible] == 0).all())
        assert bool((reconstruction.grad[~visible] != 0).all())


def test_invalid_loss_masks_fail(spectra: torch.Tensor) -> None:
    visible = TrainingRandomness("M00", 1, 1, torch.device("cpu")).prepare(spectra)[1]
    broken = visible.clone()
    broken[0, 0] = ~broken[0, 0]
    with pytest.raises(ValueError, match="patch aligned"):
        reconstruction_loss("M00", spectra, spectra, broken)
    with pytest.raises(ValueError, match="mask count"):
        reconstruction_loss("A0", spectra, spectra, visible)


def test_optimizer_excludes_bias_norm_but_decays_embeddings(tiny_model: ChemoMAE) -> None:
    optimizer = build_optimizer(tiny_model)
    decay_by_id = {id(p): group["weight_decay"] for group in optimizer.param_groups
                   for p in group["params"]}
    assert len(decay_by_id) == len(list(tiny_model.parameters()))
    for name, parameter in tiny_model.named_parameters():
        expected = 0.0 if name.endswith("bias") or ".norm" in name else 0.05
        assert decay_by_id[id(parameter)] == expected, name
    assert decay_by_id[id(tiny_model.encoder.cls_token)] == 0.05
    assert decay_by_id[id(tiny_model.encoder.pos_embed)] == 0.05
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["amsgrad"] is False
    assert all(group["lr"] == 0 for group in optimizer.param_groups)


def test_small_cpu_forward_backward_preserves_clean_target(
    tiny_model: ChemoMAE, spectra: torch.Tensor,
) -> None:
    streams = TrainingRandomness("M11", 1, 1, torch.device("cpu"))
    optimizer = build_optimizer(tiny_model)
    original = spectra.clone()
    initial = tiny_model.decoder.net.weight.detach().clone()
    for batch in range(2):
        # This only exercises component wiring on a tiny CPU fixture. It does
        # not validate the production FP16 GradScaler loop or its GPU budget.
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(0, batch, 312)
        augmented, visible = streams.prepare(spectra)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, _, actual_visible = tiny_model(augmented, visible_mask=visible)
        loss = reconstruction_loss("M11", reconstruction, spectra, actual_visible)
        loss.backward()
        optimizer.step()
        assert torch.equal(spectra, original)
        if batch == 0:
            assert torch.equal(initial, tiny_model.decoder.net.weight)
    assert not torch.equal(initial, tiny_model.decoder.net.weight)


@pytest.mark.parametrize("steps", [312, 320])
def test_lr_uses_fractional_epoch_before_batch(steps: int) -> None:
    assert learning_rate(0, 0, steps) == 0
    assert learning_rate(0, 1, steps) == pytest.approx(6e-4 / (40 * steps))
    assert learning_rate(20, 0, steps) == pytest.approx(3e-4)
    assert learning_rate(40, 0, steps) == pytest.approx(6e-4)
    assert learning_rate(420, 0, steps) == pytest.approx(3e-4)
    assert 0 < learning_rate(799, steps - 1, steps) < learning_rate(799, 0, steps)
    for args in [(800, 0, steps), (-1, 0, steps), (0, steps, steps), (0, 0, 0)]:
        with pytest.raises(ValueError):
            learning_rate(*args)


def test_full_visible_fp32_no_rng_or_decoder(
    tiny_model: ChemoMAE, spectra: torch.Tensor, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Random mask generation and decoder must not run during extraction")

    monkeypatch.setattr(tiny_model, "forward", forbidden)
    monkeypatch.setattr(tiny_model, "make_visible", forbidden)
    monkeypatch.setattr(tiny_model.decoder, "forward", forbidden)
    encoder_forward = tiny_model.encoder.forward

    def checked_forward(values: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        assert values.dtype == torch.float32 and bool(visible.all())
        assert not torch.is_autocast_enabled("cpu")
        assert not tiny_model.training
        return encoder_forward(values, visible)

    monkeypatch.setattr(tiny_model.encoder, "forward", checked_forward)
    tiny_model.train()
    tiny_model.decoder.eval()
    before = torch.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        first = extract_full_visible(tiny_model, spectra)
    second = extract_full_visible(tiny_model, spectra)
    assert torch.equal(before, torch.get_rng_state())
    assert tiny_model.training and not tiny_model.decoder.training
    assert first.values.shape == (8, 16) and first.values.dtype == np.float32
    np.testing.assert_array_equal(first.values, second.values)
    assert first.diagnostics.unit_norm_absolute_error_max < 1e-6


@pytest.mark.parametrize("value,field", [
    (0.0, "zero_norm_rows"), (1e-15, "epsilon_clamped_rows"),
    (float("nan"), "nonfinite_rows"), (1e30, "nonfinite_norm_rows"),
])
def test_latent_checked_before_normalization_and_cleanup_on_failure(
    value: float, field: str, tiny_model: ChemoMAE, spectra: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_latent(values: torch.Tensor) -> torch.Tensor:
        return torch.full((len(values), 16), value, dtype=torch.float32)

    monkeypatch.setattr(tiny_model.encoder.to_latent, "forward", bad_latent)
    tiny_model.train()
    with pytest.raises(RepresentationError) as error:
        extract_full_visible(tiny_model, spectra)
    assert getattr(error.value.diagnostics, field) == tuple(range(8))
    assert tiny_model.training
    assert len(tiny_model.encoder.to_latent._forward_hooks) == 0


def test_precision_context_restores_settings_even_on_failure() -> None:
    matmul = torch.backends.cuda.matmul.fp32_precision
    cudnn = torch.backends.cudnn.fp32_precision
    with pytest.raises(RuntimeError, match="intentional"):
        with fp32_inference(torch.device("cpu")):
            assert torch.backends.cuda.matmul.fp32_precision == "ieee"
            assert torch.backends.cudnn.fp32_precision == "ieee"
            raise RuntimeError("intentional")
    assert torch.backends.cuda.matmul.fp32_precision == matmul
    assert torch.backends.cudnn.fp32_precision == cudnn
