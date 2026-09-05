from __future__ import annotations

import numpy as np

from wood_degradation_map.preprocessing.intensity_multiotsu_masking import (
    build_three_class_multiotsu_mask,
)


def test_three_class_multiotsu_combines_middle_and_high_classes() -> None:
    low = np.arange(0.0, 30.0)
    middle = np.arange(100.0, 130.0)
    high = np.arange(200.0, 230.0)
    intensity = np.concatenate((low, middle, high)).reshape(9, 10)

    result = build_three_class_multiotsu_mask(
        intensity,
        min_object_size=1,
        erosion_radius=0,
        connectivity=2,
    )

    assert set(np.unique(result.class_map)) == {0, 1, 2}
    np.testing.assert_array_equal(
        result.mask_result.final_mask,
        result.class_map >= 1,
    )
