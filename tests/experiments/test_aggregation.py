"""Hand-calculated common-sample summaries; no models, real data or GPU execution."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

from wood_degradation_map.experiments.aggregation import (
    RepeatedSummary, ScoreRecord, aggregate_ari, aggregate_scores, paired_difference,
)
from wood_degradation_map.experiments.diagnostic_metrics import RepeatARI, repeat_ari
from wood_degradation_map.experiments.lfr import PixelLabels

PLAN = {"KYOw02702": 1, "KYOw02707": 1, "KYOw02708": 2}
IDS = tuple(PLAN)


def _records(
    matrix: list[list[float | None]], condition: str = "M11", metric: str = "lla_3",
) -> list[ScoreRecord]:
    return [ScoreRecord(sample, PLAN[sample], condition, 4, metric, repeat,
                        "undefined" if value is None else "defined", value,
                        "fixture_undefined" if value is None else None)
            for sample, values in zip(IDS, matrix, strict=True)
            for repeat, value in enumerate(values, start=1)]


def _summary(
    records: list[ScoreRecord], condition: str = "M11", metric: str = "lla_3",
) -> RepeatedSummary:
    return aggregate_scores(records, expected_test_folds=PLAN, condition_id=condition, metric=metric, k=4)


def test_sample_and_repeat_sd_are_distinct_and_folds_are_not_averaged() -> None:
    records = _records([[0, 0.1, 0.2], [0.4, 0.5, 0.6], [0.8, 0.9, 1]])
    result = _summary(records[::-1])
    assert result.total_samples == result.common_samples == 3
    assert result.common_sample_ids == IDS and result.excluded_sample_ids == ()
    assert [sample.mean for sample in result.samples] == pytest.approx([0.1, 0.5, 0.9])
    assert result.repeat_macro_means == pytest.approx([0.4, 0.5, 0.6])
    assert result.mean == pytest.approx(0.5)
    assert result.sample_sd == pytest.approx(0.4)  # ddof=1 on three sample means.
    assert result.repeat_sd == pytest.approx(0.1)  # ddof=1 on three macro means.
    assert result.mean != pytest.approx((0.3 + 0.9) / 2)  # Fold 1 has two samples, fold 2 one.
    assert all(item.defined_samples == 3 for item in result.availability)
    assert not result.has_failed_or_interrupted_sources
    assert result == _summary(records)


def test_three_repeat_intersection_and_original_availability_are_both_preserved() -> None:
    result = _summary(_records([[0, 0.1, 0.2], [0.4, 0.5, 0.6], [None, 0.9, 1]]))
    assert result.common_sample_ids == IDS[:2] and result.common_samples == 2
    assert result.excluded_sample_ids == IDS[2:]
    assert result.repeat_macro_means == pytest.approx([0.2, 0.3, 0.4])
    assert result.mean == pytest.approx(0.3)
    assert result.sample_sd == pytest.approx(math.sqrt(0.08))
    assert [item.defined_samples for item in result.availability] == [2, 3, 3]
    assert result.availability[0].unavailable[0].reason == "fixture_undefined"
    assert result.rows[-1].value == 1  # Do not discard the excluded sample's available repeats.


def test_no_common_sample_is_null_and_one_common_sample_has_only_repeat_sd() -> None:
    result = _summary(_records([[None, 0.1, 0.2], [0.4, None, 0.6], [0.8, 0.9, None]]))
    assert result.common_samples == 0 and result.samples == ()
    assert result.mean is result.sample_sd is result.repeat_sd is None
    assert result.repeat_macro_means == (None, None, None)
    assert result.mean_undefined_reason == "no_common_samples"
    assert all(item.defined_samples == 2 for item in result.availability)
    restored = json.loads(json.dumps(asdict(result), allow_nan=False))
    assert restored["mean"] is None
    result = _summary(_records([[0.2, 0.4, 0.6], [None, 0.5, 0.6], [0.8, None, 1]]))
    assert result.common_samples == 1 and result.mean == pytest.approx(0.4)
    assert result.sample_sd is None and result.repeat_sd == pytest.approx(0.2)
    assert result.sample_sd_undefined_reason == "fewer_than_two_common_samples"


def test_explicit_failure_and_interruption_are_visible_not_implicit_missingness() -> None:
    records = _records([[0.1] * 3, [0.5] * 3, [0.9] * 3])
    records[0] = replace(records[0], status="failed", value=None, reason="nonfinite_representation")
    records[4] = replace(records[4], status="interrupted", value=None, reason="user_interrupt")
    result = _summary(records)
    assert result.has_failed_or_interrupted_sources and result.common_sample_ids == IDS[2:]
    assert result.availability[0].unavailable[0].status == "failed"
    assert result.availability[1].unavailable[0].status == "interrupted"
    with pytest.raises(ValueError, match="Missing score"):
        _summary(records[1:])


def test_paired_differences_use_both_conditions_and_all_repeats() -> None:
    left = _records([[0.4, 0.5, 0.6], [None, 0.7, 0.7], [0.1, 0.1, 0.1]])
    right = _records([[0.1, 0.2, 0.3], [0.9, 0.9, 0.9], [0.8, None, 0.8]], "B0")
    result = paired_difference(left + right, expected_test_folds=PLAN, condition_id="M11",
                               reference_condition="B0", metric="lla_3", k=4)
    assert result.expression == "M11 - B0" and result.common_sample_ids == IDS[:1]
    assert result.mean == pytest.approx(0.3)
    assert result.mean != pytest.approx(_summary(left).mean - _summary(right, "B0").mean)
    assert result.repeat_macro_means == pytest.approx([0.3, 0.3, 0.3])
    assert len(result.availability) == 6
    assert result.rows[3].unavailable_sources[0].condition_id == "M11"
    assert result.rows[7].unavailable_sources[0].condition_id == "B0"
    assert result.rows[4].value == pytest.approx(-0.2)
    assert result.rows[7].value is None


def test_paired_sd_and_lfr_subtraction_direction() -> None:
    left = _records([[0.2, 0.4, 0.6], [0.4, 0.5, 0.6], [None] * 3], metric="lfr_noise")
    right = _records([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [None] * 3], "B0", "lfr_noise")
    result = paired_difference(left + right, expected_test_folds=PLAN, condition_id="M11",
                               reference_condition="B0", metric="lfr_noise", k=4)
    assert result.mean == pytest.approx(0.1)
    assert result.sample_sd == pytest.approx(math.sqrt(0.02))
    assert result.repeat_sd == pytest.approx(0.1)
    assert result.samples[1].repeat_values[0] == pytest.approx(-0.1)  # Improvement remains negative.


def test_negative_adjusted_lla_is_not_clipped() -> None:
    result = _summary(_records([[-2] * 3, [-0.5] * 3, [1] * 3], metric="adjusted_lla_3"),
                       metric="adjusted_lla_3")
    assert result.mean == -0.5 and result.sample_sd == 1.5


@pytest.mark.parametrize("changes", [
    {"value": float("nan")}, {"value": float("inf")}, {"value": True}, {"value": None},
    {"value": -0.1}, {"value": 1.1}, {"reason": "contradiction"}, {"status": "unknown"},
    {"status": "undefined", "value": None}, {"status": "failed", "reason": "failed"},
    {"status": "interrupted", "value": None, "reason": " "},
    {"sample_id": "unexpected"}, {"fold": 2}, {"fold": True}, {"repeat": 4}, {"repeat": True},
    {"k": 2}, {"metric": "lfr_noise"}, {"condition_id": "B0"},
])
def test_invalid_score_contract_or_identity_is_rejected(changes: dict[str, object]) -> None:
    records = _records([[0.1] * 3, [0.5] * 3, [0.9] * 3])
    records[0] = replace(records[0], **changes)
    with pytest.raises(ValueError):
        _summary(records)


def test_duplicates_missing_rows_and_ari_as_repeated_metric_are_rejected() -> None:
    records = _records([[0.1] * 3, [0.5] * 3, [0.9] * 3])
    with pytest.raises(ValueError, match="Duplicate"):
        _summary(records + [records[0]])
    with pytest.raises(ValueError, match="Missing"):
        _summary(records[:-1])
    with pytest.raises(ValueError, match="ARI"):
        _summary(records, metric="ari")
    with pytest.raises(ValueError, match="ARI"):
        paired_difference(records, expected_test_folds=PLAN, condition_id="M11",
                          reference_condition="B0", metric="ari", k=4)


def _ari(sample: str, partitions: list[list[int]]) -> RepeatARI:
    n = len(partitions[0])
    rows = np.arange(n)
    coordinates = np.column_stack((rows // 4, rows % 4))
    predictions = {repeat: PixelLabels(sample, rows, coordinates, np.array(labels))
                   for repeat, labels in enumerate(partitions, start=1)}
    return repeat_ari(predictions, expected_pixel_count=n, condition_id="M11", fold=PLAN[sample], k=4)


def test_ari_averages_sample_means_and_retains_degeneracy_without_repeat_sd() -> None:
    first = _ari(IDS[0], [[1, 1, 2, 2], [1, 2, 1, 2], [4, 4, 3, 3]])  # Pair mean 0.
    second = _ari(IDS[1], [[2, 2, 2, 2, 2]] * 3)  # Degenerate perfect match retained.
    third = _ari(IDS[2], [[1], [2], [3]])  # N<2 is undefined, not zero.
    result = aggregate_ari([third, second, first], expected_test_folds=PLAN, condition_id="M11", k=4)
    assert result.defined_samples == 2 and result.total_samples == 3
    assert result.defined_sample_ids == IDS[:2] and result.undefined_sample_ids == IDS[2:]
    assert result.mean == 0.5 and result.sample_sd == pytest.approx(math.sqrt(0.5))
    assert result.samples[1].pairs[0].degeneracy_flags
    assert result.samples[2].undefined_reason == "fewer_than_two_valid_pixels"
    assert not hasattr(result, "repeat_sd")
    json.dumps(asdict(result), allow_nan=False)


def test_ari_zero_or_one_defined_sample_has_explicit_sd_reason() -> None:
    records = [_ari(sample, [[1], [2], [3]]) for sample in IDS]
    result = aggregate_ari(records, expected_test_folds=PLAN, condition_id="M11", k=4)
    assert result.mean is result.sample_sd is None and result.defined_samples == 0
    records[0] = _ari(IDS[0], [[1, 1, 2, 2]] * 3)
    result = aggregate_ari(records, expected_test_folds=PLAN, condition_id="M11", k=4)
    assert result.mean == 1.0 and result.sample_sd is None and result.defined_samples == 1


@pytest.mark.parametrize("problem", ["missing_sample", "duplicate_sample", "missing_pair", "mean", "fold"])
def test_ari_incomplete_or_inconsistent_records_are_rejected(problem: str) -> None:
    records = [_ari(sample, [[1, 1, 2, 2]] * 3) for sample in IDS]
    if problem == "missing_sample":
        records.pop()
    elif problem == "duplicate_sample":
        records.append(records[0])
    elif problem == "missing_pair":
        records[0] = replace(records[0], pairs=records[0].pairs[:2])
    elif problem == "mean":
        records[0] = replace(records[0], mean=0.0)
    else:
        records[0] = replace(records[0], fold=2)
    with pytest.raises(ValueError):
        aggregate_ari(records, expected_test_folds=PLAN, condition_id="M11", k=4)
