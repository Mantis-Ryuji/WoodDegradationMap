from __future__ import annotations

import numpy as np

from wood_degradation_map.preprocessing.diagnostics import compute_reference_band_quality


def test_compute_reference_band_quality_matches_spatial_mean_over_std() -> None:
    dark = np.zeros((1, 2, 3), dtype=np.uint16)
    white = np.array([[[10, 0, 2], [10, 2, 4]]], dtype=np.uint16)

    result = compute_reference_band_quality(
        white,
        dark,
        np.array([1000.0, 1006.0, 1012.0]),
        mode="200",
        snr_threshold=2.0,
        eps=1e-12,
        dtype_max=float(np.iinfo(np.uint16).max),
    )

    np.testing.assert_allclose(result["denominator_mean"], [10.0, 1.0, 3.0])
    np.testing.assert_allclose(result["denominator_std"], [0.0, 1.0, 1.0])
    assert result["low_snr"].tolist() == [False, True, False]
    assert result["white_at_dtype_max_fraction"].tolist() == [0.0, 0.0, 0.0]
