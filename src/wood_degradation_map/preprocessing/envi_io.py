"""ENVI discovery and structural validation without loading sample cubes eagerly."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
import re
from typing import Mapping
import warnings

import numpy as np
from spectral.io import envi


_HSI_PATTERN = re.compile(r"^(?P<mode>030|200)hz_(?P<sample>KYOw[^.]+)\.hdr$", re.IGNORECASE)


class DatasetLayoutError(ValueError):
    """Raised when raw files or ENVI metadata violate the expected data contract."""


@dataclass(frozen=True)
class CubeDescriptor:
    """Metadata needed to validate and lazily open one ENVI cube."""

    mode: str
    sample_id: str | None
    hdr_path: Path
    raw_path: Path
    shape: tuple[int, int, int]
    dtype: str
    wavelengths_nm: tuple[float, ...]
    interleave: str
    fps: float | None
    x_start: int
    y_start: int
    header_offset: int
    expected_bytes: int
    actual_bytes: int


@dataclass(frozen=True)
class ReferenceSet:
    """White/dark data and metadata for one acquisition mode."""

    mode: str
    white: np.ndarray
    dark: np.ndarray
    wavelengths_nm: np.ndarray
    white_descriptor: CubeDescriptor
    dark_descriptor: CubeDescriptor
    dtype_max: float | None


def _metadata_value(metadata: Mapping[str, object], key: str) -> object | None:
    normalized_key = key.casefold()
    for candidate, value in metadata.items():
        if str(candidate).casefold() == normalized_key:
            return value
    return None


def _metadata_int(metadata: Mapping[str, object], key: str, default: int = 0) -> int:
    value = _metadata_value(metadata, key)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise DatasetLayoutError(f"Invalid integer metadata {key!r}: {value!r}") from exc


def _metadata_float(metadata: Mapping[str, object], key: str) -> float | None:
    value = _metadata_value(metadata, key)
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise DatasetLayoutError(f"Invalid numeric metadata {key!r}: {value!r}") from exc


def _metadata_wavelengths(metadata: Mapping[str, object], hdr_path: Path) -> tuple[float, ...]:
    raw_wavelengths = _metadata_value(metadata, "wavelength")
    if not isinstance(raw_wavelengths, (list, tuple)):
        raise DatasetLayoutError(f"Missing wavelength vector in {hdr_path}")
    try:
        return tuple(float(value) for value in raw_wavelengths)
    except (TypeError, ValueError) as exc:
        raise DatasetLayoutError(f"Invalid wavelength vector in {hdr_path}") from exc


def _open_descriptor(
    hdr_path: Path,
    raw_path: Path,
    *,
    mode: str,
    sample_id: str | None,
) -> CubeDescriptor:
    if not hdr_path.is_file():
        raise DatasetLayoutError(f"Missing ENVI header: {hdr_path}")
    if not raw_path.is_file():
        raise DatasetLayoutError(f"Missing ENVI raw file: {raw_path}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"spectral\.io\.envi")
        image = envi.open(str(hdr_path), str(raw_path))

    if len(image.shape) != 3:
        raise DatasetLayoutError(f"Expected a 3D ENVI cube in {hdr_path}, got {image.shape}")

    shape = tuple(int(size) for size in image.shape)
    dtype = np.dtype(image.dtype)
    metadata = image.metadata
    wavelengths = _metadata_wavelengths(metadata, hdr_path)
    if len(wavelengths) != shape[2]:
        raise DatasetLayoutError(
            f"Wavelength count does not match bands in {hdr_path}: "
            f"{len(wavelengths)} != {shape[2]}"
        )

    header_offset = _metadata_int(metadata, "header offset")
    expected_bytes = header_offset + prod(shape) * dtype.itemsize
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise DatasetLayoutError(
            f"Raw byte size mismatch for {raw_path}: {actual_bytes} != {expected_bytes}"
        )

    interleave = str(_metadata_value(metadata, "interleave") or "").strip().lower()
    fps = _metadata_float(metadata, "fps")
    expected_fps = {"030": 30.0, "200": 200.0}[mode]
    if fps is not None and not np.isclose(fps, expected_fps):
        raise DatasetLayoutError(
            f"FPS metadata does not match {mode} Hz filename in {hdr_path}: {fps}"
        )
    return CubeDescriptor(
        mode=mode,
        sample_id=sample_id,
        hdr_path=hdr_path,
        raw_path=raw_path,
        shape=shape,
        dtype=dtype.str,
        wavelengths_nm=wavelengths,
        interleave=interleave,
        fps=fps,
        x_start=_metadata_int(metadata, "x start"),
        y_start=_metadata_int(metadata, "y start"),
        header_offset=header_offset,
        expected_bytes=expected_bytes,
        actual_bytes=actual_bytes,
    )


def discover_sample_cubes(
    data_dir: Path,
    mode: str,
    *,
    expected_width: int = 320,
    expected_bands: int = 256,
) -> list[CubeDescriptor]:
    """Discover and validate sample cubes for one acquisition mode only."""

    if mode not in {"030", "200"}:
        raise ValueError(f"Unsupported acquisition mode: {mode!r}")
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise DatasetLayoutError(f"Raw data directory does not exist: {data_dir}")

    headers: dict[str, Path] = {}
    for hdr_path in data_dir.glob("*.hdr"):
        match = _HSI_PATTERN.match(hdr_path.name)
        if match is None or match.group("mode") != mode:
            continue
        sample_id = match.group("sample")
        if sample_id in headers:
            raise DatasetLayoutError(f"Duplicate {mode} Hz header for {sample_id}")
        headers[sample_id] = hdr_path
    if not headers:
        raise DatasetLayoutError(f"No {mode} Hz KYOw samples found in {data_dir}")

    cubes: list[CubeDescriptor] = []
    for sample_id in sorted(headers):
        hdr_path = headers[sample_id]
        cube = _open_descriptor(
            hdr_path,
            hdr_path.with_suffix(".raw"),
            mode=mode,
            sample_id=sample_id,
        )
        height, width, bands = cube.shape
        if height <= 0:
            raise DatasetLayoutError(f"Empty height for {cube.hdr_path}")
        if width != expected_width:
            raise DatasetLayoutError(
                f"Unexpected width for {cube.hdr_path}: {width} != {expected_width}"
            )
        if bands != expected_bands:
            raise DatasetLayoutError(
                f"Unexpected band count for {cube.hdr_path}: {bands} != {expected_bands}"
            )
        cubes.append(cube)
    return cubes


def load_reference_set(
    data_dir: Path,
    mode: str,
    *,
    expected_width: int = 320,
    expected_bands: int = 256,
) -> ReferenceSet:
    """Load and validate the small white/dark reference pair for one mode."""

    if mode not in {"030", "200"}:
        raise ValueError(f"Unsupported acquisition mode: {mode!r}")

    data_dir = data_dir.resolve()
    white_hdr = data_dir / f"{mode}hz_white.hdr"
    dark_hdr = data_dir / f"{mode}hz_dark.hdr"
    white_descriptor = _open_descriptor(
        white_hdr,
        white_hdr.with_suffix(".raw"),
        mode=mode,
        sample_id=None,
    )
    dark_descriptor = _open_descriptor(
        dark_hdr,
        dark_hdr.with_suffix(".raw"),
        mode=mode,
        sample_id=None,
    )

    if white_descriptor.shape != dark_descriptor.shape:
        raise DatasetLayoutError(
            f"White/dark shape mismatch for {mode} Hz: "
            f"{white_descriptor.shape} != {dark_descriptor.shape}"
        )
    reference_height, width, bands = white_descriptor.shape
    if reference_height != 1 or width != expected_width or bands != expected_bands:
        raise DatasetLayoutError(
            f"Unexpected {mode} Hz reference shape: {white_descriptor.shape}"
        )
    if white_descriptor.dtype != dark_descriptor.dtype:
        raise DatasetLayoutError(f"White/dark dtype mismatch for {mode} Hz")
    if white_descriptor.x_start != dark_descriptor.x_start:
        raise DatasetLayoutError(f"White/dark x start mismatch for {mode} Hz")
    if not np.allclose(
        white_descriptor.wavelengths_nm,
        dark_descriptor.wavelengths_nm,
        rtol=0.0,
        atol=1e-6,
    ):
        raise DatasetLayoutError(f"White/dark wavelength axes differ for {mode} Hz")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"spectral\.io\.envi")
        white_image = envi.open(str(white_descriptor.hdr_path), str(white_descriptor.raw_path))
        dark_image = envi.open(str(dark_descriptor.hdr_path), str(dark_descriptor.raw_path))
        white = np.asarray(white_image.load(), dtype=np.float64)
        dark = np.asarray(dark_image.load(), dtype=np.float64)

    source_dtype = np.dtype(white_descriptor.dtype)
    dtype_max = (
        float(np.iinfo(source_dtype).max)
        if np.issubdtype(source_dtype, np.integer)
        else None
    )
    return ReferenceSet(
        mode=mode,
        white=white,
        dark=dark,
        wavelengths_nm=np.asarray(white_descriptor.wavelengths_nm, dtype=np.float64),
        white_descriptor=white_descriptor,
        dark_descriptor=dark_descriptor,
        dtype_max=dtype_max,
    )


def validate_reference_against_cubes(
    cubes: list[CubeDescriptor],
    reference: ReferenceSet,
) -> None:
    """Ensure one reference uses the same columns and wavelengths as its cubes."""

    if reference.mode not in {"030", "200"}:
        raise ValueError(f"Unsupported acquisition mode: {reference.mode!r}")
    for cube in cubes:
        if cube.mode != reference.mode:
            raise DatasetLayoutError(
                f"Reference mode {reference.mode} does not match {cube.hdr_path}"
            )
        _validate_reference_against_cube(cube, reference)


def _validate_reference_against_cube(
    cube: CubeDescriptor,
    reference: ReferenceSet,
) -> None:
    sample_label = cube.sample_id or cube.hdr_path.name
    if cube.x_start != reference.white_descriptor.x_start:
        raise DatasetLayoutError(
            f"{reference.mode} Hz reference x start differs from {sample_label}"
        )
    if not np.allclose(
        cube.wavelengths_nm,
        reference.wavelengths_nm,
        rtol=0.0,
        atol=1e-6,
    ):
        raise DatasetLayoutError(
            f"{reference.mode} Hz reference wavelength axis differs from {sample_label}"
        )


def open_cube_memmap(descriptor: CubeDescriptor) -> np.ndarray:
    """Open one sample as a read-only BIP-shaped memory map."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"spectral\.io\.envi")
        image = envi.open(str(descriptor.hdr_path), str(descriptor.raw_path))
    return image.open_memmap(interleave="bip", writable=False)
