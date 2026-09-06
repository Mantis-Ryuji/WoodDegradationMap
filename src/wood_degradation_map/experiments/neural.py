"""Fixed ChemoMAE components; no automatic training, resume, or data selection."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version

import torch
from chemomae.models.chemo_mae import ChemoMAE, make_patch_mask
from chemomae.models.losses import masked_mse
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig
from torch import nn

from .baselines import NormalizationDiagnostics, NormalizedRepresentation, RepresentationError
from .config import CONDITIONS, Condition, experiment_config, run_seed


def neural_condition(condition_id: str) -> Condition:
    for condition in CONDITIONS:
        if condition.condition_id == condition_id and condition.representation == "chemomae":
            return condition
    raise ValueError(f"Not a fixed neural condition: {condition_id!r}")


class TorchRandomStream:
    """Isolate reference functions that do not accept a torch.Generator.

    Scopes use global RNG state temporarily and must run serially, never from
    competing threads. State snapshots belong to this device and RNG purpose.
    """

    def __init__(self, seed: int, device: torch.device) -> None:
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and single-device CUDA streams are supported")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self._state = torch.Generator(device=device).manual_seed(seed).get_state()
        self._active = False

    def state(self) -> torch.Tensor:
        return self._state.clone()

    def restore(self, state: torch.Tensor) -> None:
        if self._active:
            raise RuntimeError("Cannot restore an active RNG stream")
        generator = torch.Generator(device=self.device)
        generator.set_state(state)
        self._state = generator.get_state()

    @contextmanager
    def scope(self) -> Iterator[None]:
        if self._active:
            raise RuntimeError("RNG stream scopes must not be reentrant")
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            self._active = True
            try:
                if self.device.type == "cuda":
                    torch.cuda.set_rng_state(self._state, self.device)
                else:
                    torch.set_rng_state(self._state)
                yield
            finally:
                self._state = (torch.cuda.get_rng_state(self.device)
                               if self.device.type == "cuda" else torch.get_rng_state())
                self._active = False


def build_model(condition_id: str, fold: int, repeat: int) -> ChemoMAE:
    """Construct FP32 weights on CPU using unmodified reference initialization."""
    condition = neural_condition(condition_id)
    settings = dict(experiment_config()["chemomae"])
    required_version = settings.pop("version")
    if version("chemomae") != required_version:
        raise ValueError(f"The fixed protocol requires ChemoMAE {required_version}")
    settings.pop("initialization")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Model initialization requires the FP32 default dtype")
    stream = TorchRandomStream(run_seed("model_init", fold, repeat), torch.device("cpu"))
    with stream.scope(), torch.device("cpu"):
        return ChemoMAE(**settings, n_mask=condition.n_mask)


def build_augmenter(condition_id: str) -> SpectraAugmenter:
    condition = neural_condition(condition_id)
    settings = dict(experiment_config()["augmentation"])
    for key in ("noise_angle_deg_range", "shift_delta_range"):
        settings[key] = tuple(settings[key])
    return SpectraAugmenter(SpectraAugmenterConfig(
        noise_prob=condition.noise_prob, shift_prob=condition.shift_prob, **settings,
    ))


def _check_spectra(values: torch.Tensor) -> None:
    if (values.dtype != torch.float32 or values.ndim != 2 or values.shape[0] == 0
            or values.shape[1] != 256 or values.device.type not in ("cpu", "cuda")):
        raise ValueError("Expected nonempty CPU/CUDA FP32 spectra with 256 columns")
    norms = torch.linalg.vector_norm(values, dim=1)
    if not bool((torch.isfinite(values).all(dim=1) & torch.isfinite(norms) & (norms > 0)).all()):
        raise ValueError("Spectra contain nonfinite values/norms or zero-norm rows")


class TrainingRandomness:
    """Independent pixel-order, mask and augmentation streams for one run.

    Feed only the shared train matrix from FoldData. This class neither chooses
    samples nor reads test data. Epoch order includes a fresh shuffle before
    dropping its tail at the fixed batch size.
    """

    def __init__(self, condition_id: str, fold: int, repeat: int, device: torch.device) -> None:
        self.condition = neural_condition(condition_id)
        self.mask = TorchRandomStream(run_seed("mask", fold, repeat), device)
        self.augmentation = TorchRandomStream(run_seed("train_aug", fold, repeat), device)
        self.pixel_order = torch.Generator(device="cpu").manual_seed(
            run_seed("pixel_order", fold, repeat),
        )
        self.augmenter = build_augmenter(condition_id).train()

    def epoch_batches(self, train_pixel_count: int) -> Iterator[torch.Tensor]:
        batch_size = experiment_config()["training"]["batch_size"]
        if train_pixel_count < batch_size:
            raise ValueError("The train set must contain at least one fixed-size batch")
        order = torch.randperm(train_pixel_count, generator=self.pixel_order, device="cpu")
        for offset in range(0, train_pixel_count - batch_size + 1, batch_size):
            yield order[offset:offset + batch_size]

    def prepare(self, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return augmented FP32 input and patch-aligned True=visible mask.

        The caller retains clean as the target and passes the explicit mask to
        model.forward, preventing any additional reference-model mask draw.
        """
        _check_spectra(clean)
        if clean.device != self.mask.device:
            raise ValueError("Spectra and run RNG streams must use the same device")
        with torch.autocast(device_type=clean.device.type, enabled=False):
            with self.augmentation.scope():
                augmented = self.augmenter(clean.clone())
            _check_spectra(augmented)
            with self.mask.scope():
                visible = ~make_patch_mask(
                    batch_size=len(clean), seq_len=256, n_patches=16,
                    n_mask=self.condition.n_mask, device=clean.device,
                )
        return augmented, visible


def reconstruction_loss(
    condition_id: str, reconstruction: torch.Tensor, clean: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    """Reference mean squared error against clean SNV in the fixed loss region."""
    condition = neural_condition(condition_id)
    _check_spectra(clean)
    if (visible.dtype != torch.bool or visible.shape != clean.shape
            or reconstruction.shape != clean.shape or visible.device != clean.device
            or reconstruction.device != clean.device):
        raise ValueError("Reconstruction, clean target and bool visibility must align")
    patches = visible.reshape(len(clean), 16, 16)
    if not torch.equal(patches, patches[:, :, :1].expand_as(patches)):
        raise ValueError("Visibility must be patch aligned")
    if not bool(((~patches[:, :, 0]).sum(dim=1) == condition.n_mask).all()):
        raise ValueError("Visibility does not match the condition's mask count")
    mask = torch.ones_like(visible) if condition.loss_region == "all" else ~visible
    # Same operation as Trainer(loss_type="mse", reduction="mean").
    loss = masked_mse(reconstruction, clean, mask, reduction="mean")
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite reconstruction loss")
    return loss


def build_optimizer(model: ChemoMAE) -> torch.optim.AdamW:
    """Decay CLS/position embeddings; exclude only biases and norm parameters."""
    no_decay_ids = {
        id(parameter)
        for module in model.modules() if isinstance(module, nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype != torch.float32:
            raise ValueError("Training weights must be FP32")
        target = no_decay if name.endswith("bias") or id(parameter) in no_decay_ids else decay
        target.append(parameter)
    settings = experiment_config()["training"]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=0.0, betas=tuple(settings["betas"]), eps=settings["eps"],
        amsgrad=settings["amsgrad"],
    )


def learning_rate(epoch: int, batch: int, steps_per_epoch: int) -> float:
    """Return the rate to set BEFORE the zero-based batch's forward pass."""
    settings = experiment_config()["training"]
    if (not 0 <= epoch < settings["epochs"] or steps_per_epoch < 1
            or not 0 <= batch < steps_per_epoch):
        raise ValueError("Invalid epoch, batch or steps_per_epoch")
    progress = epoch + batch / steps_per_epoch
    warmup = settings["warmup_epochs"]
    peak = settings["peak_lr"]
    if progress < warmup:
        return peak * progress / warmup
    return peak * 0.5 * (1.0 + math.cos(
        math.pi * (progress - warmup) / (settings["epochs"] - warmup),
    ))


@contextmanager
def fp32_inference(device: torch.device) -> Iterator[None]:
    """Disable autocast/TF32 temporarily; do not run concurrently with training."""
    matmul = torch.backends.cuda.matmul.fp32_precision
    cudnn = torch.backends.cudnn.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = matmul
        torch.backends.cudnn.fp32_precision = cudnn


def _latent_diagnostics(values: torch.Tensor) -> NormalizationDiagnostics:
    if values.dtype != torch.float32 or values.ndim != 2 or values.shape[1] != 16:
        raise ValueError("Expected FP32 latent vectors with 16 columns")
    norms = torch.linalg.vector_norm(values, dim=1)
    finite = torch.isfinite(values).all(dim=1)
    finite_norm = torch.isfinite(norms)

    def rows(selected: torch.Tensor) -> tuple[int, ...]:
        return tuple(torch.nonzero(selected, as_tuple=True)[0].cpu().tolist())

    diagnostics = NormalizationDiagnostics(
        len(values), 16, rows(~finite), rows(finite & ~finite_norm),
        rows(finite & (norms == 0)),
        rows(finite & finite_norm & (norms > 0) & (norms < 1e-12)), None,
    )
    if (diagnostics.nonfinite_rows or diagnostics.nonfinite_norm_rows
            or diagnostics.zero_norm_rows or diagnostics.epsilon_clamped_rows):
        raise RepresentationError(diagnostics)
    return NormalizationDiagnostics(
        len(values), 16, (), (), (), (), float((norms - 1).abs().max()),
    )


def extract_full_visible(model: ChemoMAE, spectra: torch.Tensor) -> NormalizedRepresentation:
    """Extract a chunk with no augmentation/mask draw; return FP32 CPU values.

    Inspect the linear latent output before the library's eps=1e-12
    normalization so its division guard cannot hide invalid or tiny norms.
    Caller keeps the SpectrumBatch's coordinates and persists diagnostics.
    """
    _check_spectra(spectra)
    for parameter in model.parameters():
        if parameter.dtype != torch.float32 or parameter.device != spectra.device:
            raise ValueError("Model parameters must be FP32 on the spectra device")
    if not model.encoder.latent_normalize:
        raise ValueError("The fixed encoder requires latent_normalize=True")
    modes = [(module, module.training) for module in model.modules()]
    inspected = False

    def inspect_latent(module: nn.Module, inputs: tuple[object, ...], output: torch.Tensor) -> None:
        nonlocal inspected
        _latent_diagnostics(output)
        inspected = True

    handle = model.encoder.to_latent.register_forward_hook(inspect_latent)
    try:
        model.eval()
        with fp32_inference(spectra.device):
            latent = model.encoder(spectra, torch.ones_like(spectra, dtype=torch.bool))
            if not inspected:
                raise RuntimeError("The pre-normalization latent check did not run")
            diagnostics = _latent_diagnostics(latent)
            return NormalizedRepresentation(latent.cpu().numpy(), diagnostics)
    finally:
        handle.remove()
        for module, training in modes:
            module.training = training
