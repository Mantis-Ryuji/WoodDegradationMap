from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wood_degradation_map.preprocessing.spectral_grid import (
    build_linear_interpolation_plan,
    derive_terminal_snr_cutoff,
    interpolate_spectra_linear,
)


def _reference_quality(snr: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "band_index": np.arange(len(snr)),
            "wavelength_nm": np.arange(len(snr), dtype=np.float64) * 100.0 + 900.0,
            "snr_proxy": snr,
        }
    )


def test_terminal_low_snr_suffix_defines_first_excluded_band() -> None:
    decision = derive_terminal_snr_cutoff(
        _reference_quality([20.0, 15.0, 11.0, 9.0, 8.0]),
        snr_threshold=10.0,
    )

    assert decision.retained_source_bands == 3
    assert decision.last_retained_band == 2
    assert decision.first_excluded_band == 3
    assert decision.last_retained_wavelength_nm == 1100.0
    assert decision.first_excluded_wavelength_nm == 1200.0
    assert decision.cutoff_boundary_nm == 1150.0


def test_nonterminal_low_snr_band_rejects_single_cutoff_rule() -> None:
    with pytest.raises(ValueError, match="not a terminal suffix"):
        derive_terminal_snr_cutoff(
            _reference_quality([20.0, 9.0, 15.0, 14.0]),
            snr_threshold=10.0,
        )


def test_linear_interpolation_preserves_endpoints_without_extrapolation() -> None:
    plan = build_linear_interpolation_plan(
        np.array([900.0, 1000.0, 1100.0]),
        target_bands=5,
    )

    interpolated = interpolate_spectra_linear(
        np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
        plan,
        spectrum_chunk_size=1,
    )

    np.testing.assert_allclose(plan.target_wavelength_nm, [900, 950, 1000, 1050, 1100])
    np.testing.assert_allclose(interpolated, [[0.0, 0.5, 1.0, 1.5, 2.0]])
