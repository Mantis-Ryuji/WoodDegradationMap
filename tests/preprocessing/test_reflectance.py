from __future__ import annotations

import numpy as np

from wood_degradation_map.preprocessing.reflectance import (
    convert_to_reflectance,
    reflectance_l2_norm,
)


def test_convert_to_reflectance_broadcasts_column_references() -> None:
    intensity = np.array([[[6.0, 9.0], [12.0, 20.0]]])
    white = np.array([[[10.0, 10.0], [20.0, 30.0]]])
    dark = np.array([[[2.0, 1.0], [4.0, 10.0]]])

    reflectance = convert_to_reflectance(intensity, white, dark)

    np.testing.assert_allclose(reflectance, [[[0.5, 8.0 / 9.0], [0.5, 0.5]]])


def test_convert_to_reflectance_marks_nonpositive_denominators_invalid() -> None:
    intensity = np.array([[[2.0, 2.0]]])
    white = np.array([[[1.0, 2.0]]])
    dark = np.array([[[1.0, 3.0]]])

    reflectance = convert_to_reflectance(intensity, white, dark)

    assert np.isnan(reflectance).all()


def test_reflectance_l2_norm_requires_every_band_to_be_finite() -> None:
    reflectance = np.array(
        [
            [[3.0, 4.0], [1.0, np.nan]],
            [[0.0, 2.0], [5.0, 12.0]],
        ]
    )

    norm, finite_band_count = reflectance_l2_norm(reflectance)

    np.testing.assert_allclose(norm[[0, 1, 1], [0, 0, 1]], [5.0, 2.0, 13.0])
    assert np.isnan(norm[0, 1])
    np.testing.assert_array_equal(finite_band_count, [[2, 1], [2, 2]])
