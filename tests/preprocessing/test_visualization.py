from __future__ import annotations

import numpy as np

from wood_degradation_map.preprocessing.visualization import (
    shared_robust_display_limits,
)


def test_shared_display_limits_are_computed_from_all_samples() -> None:
    first = np.arange(100, dtype=np.float64).reshape(10, 10)
    second = np.arange(100, 200, dtype=np.float64).reshape(10, 10)

    limits = shared_robust_display_limits([first, second])

    expected = np.quantile(np.arange(200, dtype=np.float64), (0.01, 0.99))
    np.testing.assert_allclose(limits, expected)
