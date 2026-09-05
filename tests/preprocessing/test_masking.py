from __future__ import annotations

import numpy as np

from wood_degradation_map.preprocessing.masking import (
    build_score_mask,
    connected_component_records,
    integrate_intensity,
)


def test_integrate_intensity_sums_bands_in_float64() -> None:
    cube = np.full((2, 3, 4), 65535, dtype=np.uint16)

    result = integrate_intensity(cube)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.full((2, 3), 4 * 65535, dtype=np.float64))


def test_build_score_mask_excludes_nonfinite_scores() -> None:
    score_map = np.array([[np.nan, 0.0], [1.0, 2.0]])

    result = build_score_mask(
        score_map,
        threshold=0.5,
        min_object_size=1,
        erosion_radius=0,
        connectivity=2,
    )

    np.testing.assert_array_equal(
        result.final_mask,
        [[False, False], [True, True]],
    )


def test_connected_component_records_reports_area_and_boundary_contact() -> None:
    mask = np.zeros((6, 7), dtype=bool)
    mask[0:2, 0:2] = True
    mask[3:5, 3:6] = True

    records = connected_component_records(mask, connectivity=2)

    assert [record["area_pixels"] for record in records] == [4, 6]
    assert [record["touches_image_boundary"] for record in records] == [True, False]
