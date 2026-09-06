"""Current run contract and producer provenance validation."""

from __future__ import annotations

import torch

from wood_degradation_map.experiments.config import experiment_config
from wood_degradation_map.experiments.records import make_run_record, matches_run
from wood_degradation_map.experiments.training import _code_hashes, runtime_record


def test_current_contract_is_exact_and_execution_is_historical() -> None:
    fields = {"schema_version": 2, "condition": "A0", "config": experiment_config(),
              "runtime": runtime_record(torch.device("cpu")), "code_sha256": _code_hashes()}
    original_version = fields["runtime"]["chemomae"]
    record = make_run_record(fields)
    assert matches_run(record, fields)
    record["execution"]["runtime"]["chemomae"] = "historical producer"
    assert matches_run(record, fields)
    record["contract"]["runtime"]["chemomae"] = "different reader"
    assert not matches_run(record, fields)
    assert not matches_run({**fields, "schema_version": 1}, fields)
    assert fields["runtime"]["chemomae"] == original_version  # No aliases back to the input.
