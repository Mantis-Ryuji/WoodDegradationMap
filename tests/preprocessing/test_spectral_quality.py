from __future__ import annotations

import numpy as np

from wood_degradation_map.preprocessing.spectral_quality import (
    apply_snv_in_place,
    snv_pixel_metrics,
)


def test_snv_uses_sample_standard_deviation_and_marks_invalid_rows() -> None:
    spectra = np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
            [1.0, np.nan, 3.0],
        ],
        dtype=np.float32,
    )

    valid, input_mean, input_std, nonfinite_count = apply_snv_in_place(
        spectra,
        spectrum_chunk_size=2,
    )

    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_allclose(spectra[0], [-1.0, 0.0, 1.0])
    assert np.isnan(spectra[1:]).all()
    np.testing.assert_allclose(input_mean[:2], [2.0, 5.0])
    assert np.isnan(input_mean[2])
    np.testing.assert_allclose(input_std[:2], [1.0, 0.0])
    assert np.isnan(input_std[2])
    np.testing.assert_array_equal(nonfinite_count, [0, 0, 1])


def test_snv_metrics_recover_normalization_invariants() -> None:
    snv = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

    metrics = snv_pixel_metrics(
        snv,
        np.array([True]),
        np.array([False, False, True]),
        spectrum_chunk_size=1,
    )

    np.testing.assert_allclose(metrics["snv_mean"], [0.0])
    np.testing.assert_allclose(metrics["snv_sample_std"], [1.0])
    np.testing.assert_allclose(metrics["snv_l2_norm"], [np.sqrt(2.0)])
    np.testing.assert_allclose(metrics["first_difference_rms"], [1.0])
    np.testing.assert_allclose(metrics["max_abs_second_difference"], [0.0])
