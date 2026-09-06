"""Current run format: reader contract and actual producer provenance."""

from __future__ import annotations

from copy import deepcopy


CONTRACT_FIELDS = ("config", "runtime", "code_sha256")


def make_run_record(fields: dict) -> dict:
    """Record the current contract and the environment that actually ran the job.

    Execution provenance is historical evidence, never a version acceptance rule.
    A newly executed job has identical contract and execution records.
    """
    record = deepcopy(fields)
    contract = {key: record.pop(key) for key in CONTRACT_FIELDS}
    record.update(schema_version=2, contract=contract, execution=deepcopy(contract))
    return record


def matches_run(record: object, expected: dict) -> bool:
    """Require schema 2 and exact expected identity/config/code/runtime fields."""
    if (not isinstance(record, dict) or record.get("schema_version") != 2
            or set(CONTRACT_FIELDS).intersection(record)):
        return False
    for name in ("contract", "execution"):
        section = record.get(name)
        if (not isinstance(section, dict) or set(section) != set(CONTRACT_FIELDS)
                or any(not isinstance(section[key], dict) for key in CONTRACT_FIELDS)):
            return False
    return all((record["contract"].get(key) if key in CONTRACT_FIELDS else record.get(key))
               == value for key, value in expected.items())
