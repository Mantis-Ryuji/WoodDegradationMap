"""Chunked access to fixed train selections and complete held-out SNV samples."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import h5py
import numpy as np

from .config import FOLDS
from .input_validation import InputInventory, SampleInput
from .manifests import CVManifest, validate_cv_manifest


@dataclass(frozen=True)
class SpectrumBatch:
    """Clean SNV and its exact source rows; augmentation must work on a copy."""

    sample_id: str
    hdf5_rows: np.ndarray
    pixel_row_col: np.ndarray
    snv: np.ndarray


@dataclass(frozen=True)
class _Selection:
    sample: SampleInput
    rows: np.ndarray | None
    coordinates: np.ndarray | None


class SpectrumInputError(ValueError):
    """Preserve the affected source rows instead of dropping invalid spectra."""

    def __init__(self, sample_id: str, hdf5_rows: np.ndarray, reason: str) -> None:
        self.sample_id = sample_id
        self.hdf5_rows = tuple(int(row) for row in hdf5_rows)
        self.reason = reason
        super().__init__(f"{sample_id}: {reason}; HDF5 rows={self.hdf5_rows[:8]} "
                         f"({len(self.hdf5_rows)} affected)")


class FoldData:
    """Freeze selections from a validated manifest; never create another split.

    Train rows are shared by all downstream consumers. Test iteration includes
    every saved row, including the final partial chunk. The train matrix helper
    deliberately materializes only train spectra for ordinary sklearn PCA;
    it does not substitute IncrementalPCA for the fixed PCA definition.
    """

    def __init__(self, inventory: InputInventory, manifest: CVManifest, fold: int) -> None:
        if type(fold) is not int or fold not in FOLDS:
            raise ValueError("fold must be an integer in 1..5")
        validate_cv_manifest(manifest, inventory, check_coordinates=False)
        samples = {sample.sample_id: sample for sample in inventory.samples}
        pixels = manifest.train_pixels[fold]
        self.fold = fold
        self._train = tuple(
            _Selection(samples[sample_id], group["hdf5_row"].to_numpy(copy=True),
                       group[["pixel_row", "pixel_col"]].to_numpy(copy=True))
            for sample_id, group in pixels.groupby("sample_id", sort=True)
        )
        test_ids = manifest.folds.loc[manifest.folds["test_fold"] == fold, "sample_id"]
        self._test = tuple(_Selection(samples[sample_id], None, None)
                           for sample_id in sorted(test_ids))
        self.train_sample_ids = tuple(selection.sample.sample_id for selection in self._train)
        self.test_sample_ids = tuple(selection.sample.sample_id for selection in self._test)
        self.train_pixel_count = len(pixels)
        self.test_pixel_count = sum(selection.sample.saved_pixel_count for selection in self._test)

    def batches(
        self, split: Literal["train", "test"], *, chunk_pixels: int = 2048,
    ) -> Iterator[SpectrumBatch]:
        """Yield source-coordinate batches using bounded contiguous HDF5 reads.

        A compressed source chunk can contain unselected train rows. Such rows
        may be decompressed, but only manifest-selected rows leave this loader.
        Source windows with no selected train rows are skipped altogether.
        """
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        if type(chunk_pixels) is not int or chunk_pixels <= 0:
            raise ValueError("chunk_pixels must be a positive integer")
        for selection in self._train if split == "train" else self._test:
            sample = selection.sample
            with h5py.File(sample.path, "r") as handle:
                if (handle.attrs.get("sample_id") != sample.sample_id
                        or handle.attrs.get("saved_pixel_count") != sample.saved_pixel_count):
                    raise ValueError(f"{sample.sample_id}: HDF5 identity changed")
                for name, shape, kinds in (
                    ("snv", (sample.saved_pixel_count, 256), "f"),
                    ("pixel_row_col", (sample.saved_pixel_count, 2), "iu"),
                    ("valid_spectrum_mask", (sample.height, sample.width), "biu"),
                ):
                    if (name not in handle or not isinstance(handle[name], h5py.Dataset)
                            or handle[name].shape != shape or handle[name].dtype.kind not in kinds):
                        raise ValueError(f"{sample.sample_id}: invalid dataset {name}")
                if handle["snv"].dtype != np.dtype("float32"):
                    raise ValueError(f"{sample.sample_id}: saved SNV must be float32")
                starts = (
                    range(0, sample.saved_pixel_count, chunk_pixels)
                    if selection.rows is None else
                    np.unique(selection.rows // chunk_pixels) * chunk_pixels
                )
                for start in starts:
                    start = int(start)
                    stop = min(start + chunk_pixels, sample.saved_pixel_count)
                    if selection.rows is None:
                        rows = np.arange(start, stop, dtype=np.int64)
                        expected = None
                    else:
                        left, right = np.searchsorted(selection.rows, [start, stop])
                        rows = selection.rows[left:right]
                        expected = selection.coordinates[left:right]
                    local = rows - start
                    snv = handle["snv"][start:stop][local]
                    coordinates = handle["pixel_row_col"][start:stop][local]
                    if expected is not None and not np.array_equal(coordinates, expected):
                        raise SpectrumInputError(
                            sample.sample_id, rows, "manifest coordinate mismatch",
                        )
                    bounded = ((coordinates >= 0)
                               & (coordinates < [sample.height, sample.width])).all(axis=1)
                    if not bounded.all():
                        raise SpectrumInputError(
                            sample.sample_id, rows[~bounded], "coordinate bounds",
                        )
                    valid = np.ones(len(rows), dtype=bool)
                    for image_row in np.unique(coordinates[:, 0]):
                        selected = coordinates[:, 0] == image_row
                        mask_row = handle["valid_spectrum_mask"][int(image_row)]
                        valid[selected] = mask_row[coordinates[selected, 1]] == 1
                    if not valid.all():
                        raise SpectrumInputError(
                            sample.sample_id, rows[~valid], "invalid mask pixel",
                        )
                    finite = np.isfinite(snv).all(axis=1)
                    if not finite.all():
                        raise SpectrumInputError(sample.sample_id, rows[~finite], "non-finite SNV")
                    zero = ~(snv != 0).any(axis=1)
                    if zero.any():
                        raise SpectrumInputError(sample.sample_id, rows[zero], "zero SNV spectrum")
                    yield SpectrumBatch(sample.sample_id, rows.copy(), coordinates, snv)

    def train_matrix(self, *, chunk_pixels: int = 2048) -> np.ndarray:
        """Materialize N_train x 256 float32; input alone is 312–320 MiB in production."""
        matrix = np.empty((self.train_pixel_count, 256), dtype=np.float32)
        offset = 0
        for batch in self.batches("train", chunk_pixels=chunk_pixels):
            stop = offset + len(batch.snv)
            if stop > len(matrix):
                raise RuntimeError("Loader returned more train rows than the manifest")
            matrix[offset:stop] = batch.snv
            offset = stop
        if offset != len(matrix):
            raise RuntimeError("Loader returned fewer train rows than the manifest")
        return matrix
