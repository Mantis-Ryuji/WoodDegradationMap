"""Canonical, bounded shared SNV perturbations for the fixed LFR draws."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import torch
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig

from .config import PERTURBATIONS, Perturbation, experiment_config, perturbation_seed
from .data import SpectrumBatch, SpectrumInputError
from .neural import TorchRandomStream, fp32_inference

# Generation boundaries are independent of loader chunks and model inference.
# Changing this value changes the realized Monte Carlo inputs, not just memory use.
GENERATION_BATCH_PIXELS = 1024
DRAW_KEYS = tuple((kind, draw) for kind in PERTURBATIONS for draw in range(1, 6))


def evaluation_augmenter(kind: Perturbation) -> SpectraAugmenter:
    if kind not in PERTURBATIONS:
        raise ValueError("Unknown evaluation perturbation")
    required_version = experiment_config()["chemomae"]["version"]
    if version("chemomae") != required_version:
        raise ValueError(f"Evaluation perturbations require ChemoMAE {required_version}")
    settings = dict(experiment_config()["augmentation"])
    for key in ("noise_angle_deg_range", "shift_delta_range"):
        settings[key] = tuple(settings[key])
    return SpectraAugmenter(SpectraAugmenterConfig(
        noise_prob=float(kind in ("noise", "both")),
        shift_prob=float(kind in ("shift", "both")), **settings,
    )).train()


def _readonly(array: np.ndarray) -> np.ndarray:
    result = array.copy()
    result.flags.writeable = False
    return result


def _validate_batch(batch: SpectrumBatch) -> None:
    if (not isinstance(batch.snv, np.ndarray) or batch.snv.dtype != np.float32
            or batch.snv.ndim != 2 or batch.snv.shape[1] != 256 or len(batch.snv) == 0):
        raise ValueError("Expected nonempty FP32 SNV with 256 columns")
    n = len(batch.snv)
    if (batch.hdf5_rows.shape != (n,) or batch.hdf5_rows.dtype.kind not in "iu"
            or batch.pixel_row_col.shape != (n, 2) or batch.pixel_row_col.dtype.kind not in "iu"
            or np.any(batch.pixel_row_col < 0)
            or np.any(batch.pixel_row_col > np.iinfo(np.int64).max)):
        raise ValueError("Invalid HDF5 row/coordinate identity")


def _check_spectra(batch: SpectrumBatch, values: torch.Tensor) -> None:
    norms = torch.linalg.vector_norm(values, dim=1)
    bad = (~torch.isfinite(values).all(dim=1) | ~torch.isfinite(norms)
           | (norms <= experiment_config()["augmentation"]["eps"]))
    if bool(bad.any()):
        raise SpectrumInputError(batch.sample_id, batch.hdf5_rows[bad.cpu().numpy()],
                                 "Nonfinite or zero/epsilon-clamped evaluation spectrum")


@dataclass(frozen=True)
class PerturbedSpectra:
    kind: Perturbation
    draw: int
    seed: int
    batch: SpectrumBatch


@dataclass(frozen=True)
class SharedPerturbationBlock:
    clean: SpectrumBatch
    perturbed: tuple[PerturbedSpectra, ...]


class SharedPerturbations:
    """Replay a sample's 15 independent RNG streams on canonical generation batches.

    Source batches must contain all saved rows of ONE test sample in HDF5 order.
    FoldData validates the source mask/coordinates before they enter this class.
    A fresh batches() traversal restarts all streams. Replay requires the same
    input, generation width, device and software; CPU/CUDA bit equality is not
    promised. Use each returned block across all conditions/K/repeats before
    discarding it. Read-only CPU arrays protect shared inputs from normal writes.
    Never collect an entire sample's blocks into a list in production.
    """

    def __init__(self, sample_id: str, pixel_count: int, *, device: torch.device) -> None:
        if not sample_id or type(pixel_count) is not int or pixel_count < 1:
            raise ValueError("Expected a sample ID and positive full saved pixel count")
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and single-device CUDA are supported")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.sample_id, self.pixel_count, self.device = sample_id, pixel_count, device

    def record(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id, "pixels": self.pixel_count,
            "generation_batch_pixels": GENERATION_BATCH_PIXELS,
            "row_order": "all saved HDF5 rows, ascending; final partial batch retained",
            "device": str(self.device), "torch": str(torch.__version__),
            "chemomae": version("chemomae"), "cuda_runtime": torch.version.cuda,
            "dtype": "float32", "tf32": False,
            "augmentation": experiment_config()["augmentation"],
            "draws": [{"kind": kind, "draw": draw,
                       "seed": perturbation_seed(self.sample_id, kind, draw),
                       "noise_prob": int(kind in ("noise", "both")),
                       "shift_prob": int(kind in ("shift", "both"))} for kind, draw in DRAW_KEYS],
        }

    def batches(self, source: Iterable[SpectrumBatch]) -> Iterator[SharedPerturbationBlock]:
        streams = {key: TorchRandomStream(perturbation_seed(self.sample_id, *key), self.device)
                   for key in DRAW_KEYS}
        augmenters = {kind: evaluation_augmenter(kind) for kind in PERTURBATIONS}
        spectra = np.empty((GENERATION_BATCH_PIXELS, 256), dtype=np.float32)
        coordinates = np.empty((GENERATION_BATCH_PIXELS, 2), dtype=np.int64)
        received, filled = 0, 0

        def generate(start: int, size: int) -> SharedPerturbationBlock:
            clean = SpectrumBatch(self.sample_id, _readonly(np.arange(start, start + size)),
                                  _readonly(coordinates[:size]), _readonly(spectra[:size]))
            tensor = torch.from_numpy(clean.snv.copy()).to(self.device)
            variants = []
            with fp32_inference(self.device):
                _check_spectra(clean, tensor)
                for kind, draw in DRAW_KEYS:
                    with streams[kind, draw].scope():
                        augmented = augmenters[kind].train()(tensor.clone())
                    _check_spectra(clean, augmented)
                    batch = SpectrumBatch(clean.sample_id, clean.hdf5_rows, clean.pixel_row_col,
                                          _readonly(augmented.cpu().numpy()))
                    variants.append(PerturbedSpectra(
                        kind, draw, perturbation_seed(self.sample_id, kind, draw), batch,
                    ))
            # Never yield from within the global RNG/precision scopes.
            return SharedPerturbationBlock(clean, tuple(variants))

        for batch in source:
            _validate_batch(batch)
            if (batch.sample_id != self.sample_id or received + len(batch.snv) > self.pixel_count
                    or not np.array_equal(batch.hdf5_rows, np.arange(received, received + len(batch.snv)))):
                raise ValueError("Expected the same sample's complete, ordered, nonduplicated HDF5 rows")
            offset = 0
            while offset < len(batch.snv):
                take = min(len(batch.snv) - offset, GENERATION_BATCH_PIXELS - filled)
                spectra[filled:filled + take] = batch.snv[offset:offset + take]
                coordinates[filled:filled + take] = batch.pixel_row_col[offset:offset + take]
                filled += take
                received += take
                offset += take
                if filled == GENERATION_BATCH_PIXELS:
                    yield generate(received - filled, filled)
                    filled = 0
        if received != self.pixel_count:
            raise ValueError("Missing saved test rows; no partial-sample LFR is allowed")
        if filled:
            yield generate(received - filled, filled)
