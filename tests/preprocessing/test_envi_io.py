from __future__ import annotations

from pathlib import Path

import pytest

from wood_degradation_map.preprocessing import envi_io


def _descriptor(path: Path, mode: str, sample_id: str, height: int) -> envi_io.CubeDescriptor:
    return envi_io.CubeDescriptor(
        mode=mode,
        sample_id=sample_id,
        hdr_path=path,
        raw_path=path.with_suffix(".raw"),
        shape=(height, 320, 256),
        dtype="<u2",
        wavelengths_nm=tuple(float(index) for index in range(256)),
        interleave="bil",
        fps=float(int(mode)),
        x_start=0,
        y_start=0,
        header_offset=0,
        expected_bytes=height * 320 * 256 * 2,
        actual_bytes=height * 320 * 256 * 2,
    )


def test_single_mode_discovery_does_not_require_030_or_rgb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_id = "KYOw00001"
    (tmp_path / f"200hz_{sample_id}.hdr").touch()

    def fake_open_descriptor(
        hdr_path: Path,
        raw_path: Path,
        *,
        mode: str,
        sample_id: str | None,
    ) -> envi_io.CubeDescriptor:
        del raw_path
        assert sample_id is not None
        return _descriptor(hdr_path, mode, sample_id, 666)

    monkeypatch.setattr(envi_io, "_open_descriptor", fake_open_descriptor)

    cubes = envi_io.discover_sample_cubes(tmp_path, "200")

    assert len(cubes) == 1
    assert cubes[0].sample_id == sample_id
    assert cubes[0].shape == (666, 320, 256)
