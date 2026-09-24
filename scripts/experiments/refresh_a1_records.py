"""One-time, metadata-only A1 contract update; normal readers remain strict."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from wood_degradation_map.experiments import config as protocol
from wood_degradation_map.experiments.global_manifest import global_config


# The only scientific-code differences approved for this one-time update are
# the added A1 condition and its inclusion in the global condition lists.
PREVIOUS_CODE = {
    "config.py": "6d09044cd9bc023dc0f427193633ef1808910a43d1e158f0702a8c7b2247bb8a",
    "global_manifest.py": "a11f6d436d7b4c9534fde03cb04e9ad12f992460428800d4cc6a34888aa4d341",
}
BOUND_REFERENCES = {
    "run_sha256", "completion_sha256", "fit_record_sha256",
    "clustering_completion_sha256", "manifest_sha256",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


class UpdatePlan:
    """Validate the complete JSON dependency graph before changing any file."""

    def __init__(self, experiment: Path, target: dict) -> None:
        self.root = experiment.resolve()
        self.target = target
        self.previous = {**target, "conditions": [c for c in target["conditions"]
                                                  if c["condition_id"] != "A1"]}
        self.originals: dict[Path, bytes] = {}
        self.changes: dict[Path, bytes] = {}
        self.hashes: dict[str, str] = {}
        self.manifest_artifacts: dict[str, str] = {}

    def read(self, path: Path) -> dict:
        if (path.absolute() != path.resolve() or not path.resolve().is_relative_to(self.root)
                or not path.is_file()):
            raise ValueError(f"Invalid metadata path: {path}")
        data = path.read_bytes()
        if path in self.originals and data != self.originals[path]:
            raise ValueError(f"Metadata changed during planning: {path}")
        self.originals[path] = data
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return value

    def save(self, path: Path, value: dict) -> None:
        original = self.originals[path]
        updated = original if json.loads(original) == value else _encode(value)
        self.hashes[_sha(original)] = _sha(updated)
        if updated != original:
            self.changes[path] = updated

    def refresh(self, value: object, *, hash_values: bool = False) -> object:
        if isinstance(value, list):
            return [self.refresh(item, hash_values=hash_values) for item in value]
        if not isinstance(value, dict):
            return self.hashes.get(value, value) if hash_values and isinstance(value, str) else value
        result = {}
        for key, item in value.items():
            if key == "execution":
                result[key] = deepcopy(item)  # Never relabel actual training as a new execution.
            elif key == "config":
                if item not in (self.previous, self.target):
                    raise ValueError("Config differs by more than the A1 condition")
                result[key] = deepcopy(self.target)
            elif key == "code_sha256":
                if not isinstance(item, dict):
                    raise ValueError("Expected code hashes")
                result[key] = {}
                for name, saved_hash in item.items():
                    if Path(name).name != name or not name.endswith(".py"):
                        raise ValueError(f"Invalid code path: {name}")
                    current = _sha(Path(protocol.__file__).with_name(name).read_bytes())
                    if saved_hash != current and (
                            name not in PREVIOUS_CODE or saved_hash != PREVIOUS_CODE[name]):
                        raise ValueError(f"Unreviewed code difference: {name}")
                    result[key][name] = current
            else:
                if key == "manifest_artifact_sha256" and item != self.manifest_artifacts:
                    raise ValueError("Run is bound to a different manifest")
                if key in BOUND_REFERENCES and item not in self.hashes:
                    raise ValueError(f"Unbound or stale source reference: {key}")
                result[key] = self.refresh(item, hash_values=hash_values or key.endswith("sha256"))
        return result

    def add_record(self, path: Path) -> None:
        self.save(path, self.refresh(self.read(path)))

    def completed_run(self, path: Path, status: str) -> None:
        done_path = path.with_name("completion.json")
        done = self.read(done_path)
        if done.get("status") != status or path.with_name("failure.json").exists():
            raise ValueError(f"Only complete successful runs can be updated: {path.parent}")
        run = self.read(path)
        if status != "global_maps_completed" and (
                run.get("schema_version") != 2 or not isinstance(run.get("contract"), dict)
                or not isinstance(run.get("execution"), dict)):
            raise ValueError(f"Expected a current run record: {path}")
        if (status != "global_maps_completed" and run.get("condition") == "A1"
                and run["contract"].get("config") != self.target):
            raise ValueError("An A1 run must already record the A1 recipe")
        self.add_record(path)
        self.save(done_path, self.refresh(done))

    def apply(self) -> None:
        # Includes unchanged sources used to calculate downstream hash references.
        if any(path.read_bytes() != data for path, data in self.originals.items()):
            raise ValueError("Metadata changed after planning")
        if not self.changes:
            return
        audit = self.root / "config/a1_contract_update.json"
        if audit.exists():
            raise FileExistsError("An A1 update audit already exists; inspect it before another update")
        report = {"scope": "A1 metadata update; no training or metric recomputation",
                  "files": {p.relative_to(self.root).as_posix(): {
                      "before_sha256": _sha(self.originals[p]), "after_sha256": _sha(data),
                      "before_json": self.originals[p].decode("utf-8"),
                  } for p, data in self.changes.items()}}
        with TemporaryDirectory(prefix=".a1-records-", dir=self.root) as temporary:
            stage = Path(temporary)
            written = []
            try:
                for index, (path, data) in enumerate(self.changes.items()):
                    pending = stage / f"{index}.json"
                    pending.write_bytes(data)
                    pending.replace(path)
                    written.append(path)
                pending = stage / "audit.json"
                pending.write_bytes(_encode(report))
                pending.replace(audit)
            except BaseException:
                # A failed update must not leave half of the hash graph refreshed.
                for index, path in enumerate(reversed(written)):
                    pending = stage / f"restore-{index}.json"
                    pending.write_bytes(self.originals[path])
                    pending.replace(path)
                raise


def build_plan(experiment: Path, *, scope: str) -> UpdatePlan:
    """Refresh only fixed inputs and completed fit/map/evaluation JSON records."""
    if scope not in ("cv", "global"):
        raise ValueError("Expected cv or global scope")
    plan = UpdatePlan(experiment, protocol.experiment_config() if scope == "cv" else global_config())
    root = plan.root
    config_path, manifest_path = root / "config/experiment.json", root / "manifests/complete.json"
    saved, manifest = plan.read(config_path), plan.read(manifest_path)
    if (saved not in (plan.previous, plan.target) or manifest.get("status") != "complete"
            or manifest["artifact_sha256"]["config/experiment.json"] != _sha(plan.originals[config_path])):
        raise ValueError("Expected a complete manifest with only the A1 config extension")
    plan.manifest_artifacts = manifest["artifact_sha256"]
    plan.save(config_path, plan.target)
    plan.save(manifest_path, plan.refresh(manifest))
    results = root / "results"
    if scope == "cv":
        for path in sorted((results / "baselines").glob("fold_*/repeat_*/fit.json")):
            if plan.read(path).get("status") != "fitted_and_roundtrip_checked":
                raise ValueError(f"Incomplete baseline: {path}")
            plan.add_record(path)
        pattern = "*/fold_*/repeat_*/run.json"
    else:
        baseline = results / "baselines/completion.json"
        if plan.read(baseline).get("status") != "fitted_and_roundtrip_checked":
            raise ValueError("Incomplete global baseline")
        plan.add_record(baseline)
        pattern = "*/repeat_*/run.json"
    for branch, status in (("neural", "training_completed"),
                           ("clustering", "clean_test_maps_completed" if scope == "cv"
                            else "global_maps_completed"),
                           ("evaluation", "full_test_evaluation_completed")):
        if scope == "global" and branch == "evaluation":
            continue
        paths = set((results / branch).glob(pattern))
        conditions = [c["condition_id"] for c in plan.previous["conditions"]
                      if c["experiment"] == "main"
                      and (branch != "neural" or c["representation"] == "chemomae")]
        required = {results / branch / condition / suffix / "run.json"
                    for condition in conditions
                    for suffix in ([f"fold_{fold}/repeat_{repeat}"
                                    for fold in protocol.FOLDS for repeat in protocol.REPEATS]
                                   if scope == "cv" else ["repeat_1"])}
        if not required <= paths:
            raise ValueError(f"Missing or unreadable completed {branch} records: "
                             f"{sorted(str(p) for p in required - paths)}")
        for path in sorted(paths):
            plan.completed_run(path, status)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("cv", "global"), required=True)
    parser.add_argument("--apply", action="store_true", help="Apply the validated metadata update")
    args = parser.parse_args()
    plan = build_plan(args.experiment_dir, scope=args.scope)
    if args.apply:
        plan.apply()
    print(json.dumps({"status": "updated" if args.apply else "planned", "files": len(plan.changes),
                      "paths": [p.relative_to(plan.root).as_posix() for p in plan.changes]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
