"""Sample-macro summaries and paired differences under evaluation protocol section 8."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import CLUSTER_COUNTS, CONDITIONS, FOLDS, REPEATS, experiment_config
from .diagnostic_metrics import RepeatARI

ScoreStatus = Literal["defined", "undefined", "failed", "interrupted"]
REPEATED_METRICS = (
    "lla_3", "lla_5", "lla_9", "adjusted_lla_3", "adjusted_lla_5", "adjusted_lla_9",
    "lfr_noise", "lfr_shift", "lfr_both", "silhouette",
)


@dataclass(frozen=True)
class ScoreRecord:
    sample_id: str
    fold: int
    condition_id: str
    k: int
    metric: str
    repeat: int
    status: ScoreStatus
    value: float | None
    reason: str | None = None


@dataclass(frozen=True)
class UnavailableScore:
    sample_id: str
    fold: int
    condition_id: str
    repeat: int
    status: ScoreStatus
    reason: str


@dataclass(frozen=True)
class Availability:
    condition_id: str
    repeat: int
    total_samples: int
    defined_samples: int
    unavailable: tuple[UnavailableScore, ...]


@dataclass(frozen=True)
class SummaryRow:
    sample_id: str
    fold: int
    repeat: int
    value: float | None
    unavailable_sources: tuple[UnavailableScore, ...]


@dataclass(frozen=True)
class SampleMean:
    sample_id: str
    fold: int
    repeat_values: tuple[float, ...]
    mean: float


@dataclass(frozen=True)
class RepeatedSummary:
    expression: str  # Paired differences retain condition minus reference, including for LFR.
    metric: str
    k: int
    total_samples: int
    common_samples: int
    common_sample_ids: tuple[str, ...]
    excluded_sample_ids: tuple[str, ...]
    availability: tuple[Availability, ...]
    rows: tuple[SummaryRow, ...]  # Includes defined pairs even if another repeat excludes the sample.
    samples: tuple[SampleMean, ...]
    repeat_ids: tuple[int, ...]
    repeat_macro_means: tuple[float | None, ...]
    mean: float | None
    sample_sd: float | None
    repeat_sd: float | None
    mean_undefined_reason: str | None
    sample_sd_undefined_reason: str | None
    repeat_sd_undefined_reason: str | None
    has_failed_or_interrupted_sources: bool


def _plan(expected_test_folds: Mapping[str, int], conditions: tuple[str, ...], k: int) -> tuple[str, ...]:
    if (not expected_test_folds or any(not isinstance(sample, str) or not sample
                                      for sample in expected_test_folds)
            or any(type(fold) is not int or fold not in FOLDS for fold in expected_test_folds.values())):
        raise ValueError("Expected a nonempty sample-to-test-fold mapping from the saved manifest")
    known = {condition.condition_id for condition in CONDITIONS}
    if not conditions or any(condition not in known for condition in conditions):
        raise ValueError("Unknown experiment condition")
    if type(k) is not int or k not in CLUSTER_COUNTS:
        raise ValueError("Expected an integer K from the fixed plan")
    return tuple(sorted(expected_test_folds))


def _number(value: object) -> np.float32:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or abs(value) > np.finfo(np.float32).max):
        raise ValueError("A defined score must be a finite FP32-representable number")
    return np.float32(value)


def _mean(values: np.ndarray) -> float:
    result = float(values.mean(dtype=np.float32))
    if not np.isfinite(result):
        raise ValueError("Nonfinite aggregate mean")
    return result


def _sd(values: np.ndarray) -> float:
    result = float(values.std(ddof=1, dtype=np.float32))
    if not np.isfinite(result):
        raise ValueError("Nonfinite aggregate sample SD")
    return result


def _validated_records(
    records: Sequence[ScoreRecord], expected_test_folds: Mapping[str, int],
    conditions: tuple[str, ...], metric: str, k: int,
) -> tuple[dict[tuple[str, str, int], ScoreRecord], tuple[Availability, ...]]:
    ids = _plan(expected_test_folds, conditions, k)
    if metric not in REPEATED_METRICS:
        raise ValueError("Unknown repeated metric; ARI requires its separate sample-level summary")
    indexed = {}
    for record in records:
        if (record.sample_id not in expected_test_folds
                or type(record.fold) is not int or record.fold != expected_test_folds[record.sample_id]
                or record.condition_id not in conditions or type(record.k) is not int or record.k != k
                or record.metric != metric or type(record.repeat) is not int or record.repeat not in REPEATS):
            raise ValueError("Score sample/fold/condition/K/metric/repeat differs from the requested plan")
        key = (record.condition_id, record.sample_id, record.repeat)
        if key in indexed:
            raise ValueError("Duplicate score record; never average duplicate runs")
        if record.status == "defined":
            value = _number(record.value)
            lower = -np.inf if metric.startswith("adjusted_lla_") else -1 if metric == "silhouette" else 0
            if value < lower or value > 1 or record.reason is not None:
                raise ValueError("Defined score range/reason violates the metric contract")
        elif record.status in ("undefined", "failed", "interrupted"):
            if record.value is not None or not isinstance(record.reason, str) or not record.reason.strip():
                raise ValueError("Unavailable scores require value=None and an explicit reason")
        else:
            raise ValueError("Unknown score status")
        indexed[key] = record
    expected = {(condition, sample, repeat) for condition in conditions for sample in ids for repeat in REPEATS}
    if set(indexed) != expected:
        raise ValueError("Missing score records; absent runs are not implicit undefined values")
    availability = []
    for condition in conditions:
        for repeat in REPEATS:
            unavailable = tuple(_unavailable(indexed[condition, sample, repeat]) for sample in ids
                                if indexed[condition, sample, repeat].status != "defined")
            availability.append(Availability(condition, repeat, len(ids), len(ids) - len(unavailable), unavailable))
    return indexed, tuple(availability)


def _unavailable(record: ScoreRecord) -> UnavailableScore:
    return UnavailableScore(record.sample_id, record.fold, record.condition_id, record.repeat,
                            record.status, record.reason)


def _summarize(
    rows: tuple[SummaryRow, ...], availability: tuple[Availability, ...],
    expected_test_folds: Mapping[str, int], expression: str, metric: str, k: int,
) -> RepeatedSummary:
    ids = tuple(sorted(expected_test_folds))
    indexed = {(row.sample_id, row.repeat): row for row in rows}
    common = tuple(sample for sample in ids if all(indexed[sample, repeat].value is not None for repeat in REPEATS))
    excluded = tuple(sample for sample in ids if sample not in common)
    samples, repeat_means = [], (None, None, None)
    mean = sample_sd = repeat_sd = None
    if common:
        values = np.array([[indexed[sample, repeat].value for repeat in REPEATS] for sample in common],
                          dtype=np.float32)
        for sample, values_row in zip(common, values, strict=True):
            samples.append(SampleMean(sample, expected_test_folds[sample], tuple(float(value) for value in values_row),
                                      _mean(values_row)))
        repeat_means = tuple(_mean(values[:, index]) for index in range(len(REPEATS)))
        mean = _mean(np.array(repeat_means, dtype=np.float32))
        repeat_sd = _sd(np.array(repeat_means, dtype=np.float32))
        if len(common) >= 2:
            sample_sd = _sd(np.array([sample.mean for sample in samples], dtype=np.float32))
    no_common = "no_common_samples" if not common else None
    return RepeatedSummary(
        expression, metric, k, len(ids), len(common), common, excluded, availability, rows, tuple(samples), REPEATS,
        repeat_means, mean, sample_sd, repeat_sd, no_common,
        "fewer_than_two_common_samples" if len(common) < 2 else None, no_common,
        any(row.status in ("failed", "interrupted") for item in availability for row in item.unavailable),
    )


def aggregate_scores(
    records: Sequence[ScoreRecord], *, expected_test_folds: Mapping[str, int],
    condition_id: str, metric: str, k: int,
) -> RepeatedSummary:
    """Summarize one metric/condition/K over OOF samples, not fold means or pixels.

    expected_test_folds comes from the validated saved manifest. Supply one record
    per expected sample and repeat, with an explicit status even for unsuccessful
    runs. LFR records contain each kind's five-draw sample mean, not individual draws.
    Only samples defined in all three repeats enter the means and both ddof=1 SDs.
    Availability retains each repeat's original defined count and exclusions.
    """
    indexed, availability = _validated_records(records, expected_test_folds, (condition_id,), metric, k)
    rows = tuple(SummaryRow(
        sample, expected_test_folds[sample], repeat,
        float(_number(record.value)) if record.status == "defined" else None,
        () if record.status == "defined" else (_unavailable(record),),
    ) for sample in sorted(expected_test_folds) for repeat in REPEATS
        for record in (indexed[condition_id, sample, repeat],))
    return _summarize(rows, availability, expected_test_folds, condition_id, metric, k)


def paired_difference(
    records: Sequence[ScoreRecord], *, expected_test_folds: Mapping[str, int],
    condition_id: str, reference_condition: str, metric: str, k: int,
) -> RepeatedSummary:
    """Subtract condition minus reference per sample/repeat before averaging.

    Use the intersection defined in both conditions and all three repeats. Keep
    every per-repeat difference and the unavailable source records, including for
    samples excluded from the final common set. Never negate differences merely
    because a smaller metric (such as LFR) is preferable. No ARI contrast is allowed.
    """
    if condition_id == reference_condition:
        raise ValueError("Paired comparisons require two distinct conditions")
    indexed, availability = _validated_records(
        records, expected_test_folds, (condition_id, reference_condition), metric, k,
    )
    rows = []
    for sample in sorted(expected_test_folds):
        for repeat in REPEATS:
            left, right = indexed[condition_id, sample, repeat], indexed[reference_condition, sample, repeat]
            unavailable = tuple(_unavailable(record) for record in (left, right) if record.status != "defined")
            difference = None if unavailable else float(_number(left.value) - _number(right.value))
            if difference is not None and not np.isfinite(difference):
                raise ValueError("Nonfinite paired difference")
            rows.append(SummaryRow(sample, expected_test_folds[sample], repeat, difference, unavailable))
    return _summarize(tuple(rows), availability, expected_test_folds,
                      f"{condition_id} - {reference_condition}", metric, k)


@dataclass(frozen=True)
class ARIMacroSummary:
    condition_id: str
    k: int
    total_samples: int
    defined_samples: int
    defined_sample_ids: tuple[str, ...]
    undefined_sample_ids: tuple[str, ...]
    samples: tuple[RepeatARI, ...]  # Preserve pair values, reasons, counts and degeneracy flags.
    mean: float | None
    sample_sd: float | None
    mean_undefined_reason: str | None
    sample_sd_undefined_reason: str | None


def aggregate_ari(
    records: Sequence[RepeatARI], *, expected_test_folds: Mapping[str, int], condition_id: str, k: int,
) -> ARIMacroSummary:
    """Macro-average each sample's three-pair mean; do not invent a repeat-pair SD."""
    ids = _plan(expected_test_folds, (condition_id,), k)
    indexed = {}
    expected_pairs = tuple(tuple(pair) for pair in experiment_config()["evaluation"]["ari_pairs"])
    for record in records:
        if (record.sample_id not in expected_test_folds or record.sample_id in indexed
                or type(record.fold) is not int or record.fold != expected_test_folds[record.sample_id]
                or record.condition_id != condition_id or type(record.k) is not int or record.k != k
                or type(record.valid_pixels) is not int or record.valid_pixels < 1
                or tuple(pair.repeats for pair in record.pairs) != expected_pairs):
            raise ValueError("ARI sample/run/pair coverage differs from the requested plan")
        if record.valid_pixels < 2:
            if (record.mean is not None or record.undefined_reason != "fewer_than_two_valid_pixels"
                    or any(pair.value is not None or pair.undefined_reason != record.undefined_reason
                           for pair in record.pairs)):
                raise ValueError("Invalid undefined ARI result")
        else:
            values = np.array([_number(pair.value) for pair in record.pairs], dtype=np.float32)
            if (np.any(values < -1) or np.any(values > 1) or record.undefined_reason is not None
                    or any(pair.undefined_reason is not None for pair in record.pairs)
                    or float(_number(record.mean)) != _mean(values)):
                raise ValueError("Invalid ARI pair values or sample mean")
        indexed[record.sample_id] = record
    if set(indexed) != set(ids):
        raise ValueError("Missing ARI sample records; incomplete repetitions must not be averaged")
    defined = tuple(sample for sample in ids if indexed[sample].mean is not None)
    undefined = tuple(sample for sample in ids if indexed[sample].mean is None)
    values = np.array([indexed[sample].mean for sample in defined], dtype=np.float32)
    return ARIMacroSummary(condition_id, k, len(ids), len(defined), defined, undefined,
                           tuple(indexed[sample] for sample in ids), _mean(values) if len(values) else None,
                           _sd(values) if len(values) >= 2 else None,
                           None if len(values) else "no_defined_samples",
                           None if len(values) >= 2 else "fewer_than_two_defined_samples")
