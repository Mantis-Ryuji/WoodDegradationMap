"""One-time metadata update, with immutable numerical data and execution provenance."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from wood_degradation_map.experiments import config
from wood_degradation_map.experiments.records import make_run_record, matches_run


def updater() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/experiments/refresh_a1_records.py"
    spec = importlib.util.spec_from_file_location("refresh_a1_records", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Path, dict]:
    module = updater()
    target = config.experiment_config()
    target["conditions"] = [c for c in target["conditions"] if c["condition_id"] in ("A0", "A1")]
    monkeypatch.setattr(module.protocol, "experiment_config", lambda: deepcopy(target))
    monkeypatch.setattr(module.protocol, "FOLDS", (1,))
    monkeypatch.setattr(module.protocol, "REPEATS", (1,))
    previous = {**target, "conditions": target["conditions"][:1]}
    config_hash = write(tmp_path / "config/experiment.json", previous)
    artifacts = {"config/experiment.json": config_hash, "manifests/folds.parquet": "fixed-split"}
    write(tmp_path / "manifests/complete.json", {"status": "complete", "artifact_sha256": artifacts})
    identity = {"condition": "A0", "fold": 1, "repeat": 1, "config": previous,
                "runtime": {"device": "cpu"}, "code_sha256": {
                    "config.py": module.PREVIOUS_CODE["config.py"]},
                "manifest_artifact_sha256": artifacts}
    neural = tmp_path / "results/neural/A0/fold_1/repeat_1"
    neural_hash = write(neural / "run.json", make_run_record({**identity, "mode": "training"}))
    neural_done = write(neural / "completion.json", {"status": "training_completed"})
    source = {"kind": "neural", "run_sha256": neural_hash, "completion_sha256": neural_done}
    cluster = tmp_path / "results/clustering/A0/fold_1/repeat_1"
    cluster_hash = write(cluster / "run.json", make_run_record({**identity, "source": source}))
    cluster_done = write(cluster / "completion.json",
                         {"status": "clean_test_maps_completed", "run_sha256": cluster_hash})
    evaluation = tmp_path / "results/evaluation/A0/fold_1/repeat_1"
    evaluation_hash = write(evaluation / "run.json", make_run_record({
        **identity, "source": source, "clustering_completion_sha256": cluster_done,
    }))
    write(evaluation / "completion.json", {"status": "full_test_evaluation_completed",
                                           "run_sha256": evaluation_hash})
    write(evaluation / "scores.json", {"untouched": [0.2, 0.4]})
    return module, tmp_path, target


def test_refresh_rebinds_graph_without_changing_execution_or_scores(
    saved: tuple[ModuleType, Path, dict],
) -> None:
    module, root, target = saved
    original = {path: path.read_bytes() for path in root.rglob("*.json")}
    plan = module.build_plan(root, scope="cv")
    assert original == {path: path.read_bytes() for path in original}  # Planning is read-only.
    plan.apply()
    for branch in ("neural", "clustering", "evaluation"):
        path = root / f"results/{branch}/A0/fold_1/repeat_1/run.json"
        run = json.loads(path.read_bytes())
        assert run["execution"] == json.loads(original[path])["execution"]
        assert matches_run(run, {"schema_version": 2, "condition": "A0", "config": target,
                                 "code_sha256": {"config.py": module._sha(
                                     Path(config.__file__).read_bytes())}})
        if branch != "neural":
            done = json.loads(path.with_name("completion.json").read_bytes())
            assert done["run_sha256"] == module._sha(path.read_bytes())
    evaluation = root / "results/evaluation/A0/fold_1/repeat_1"
    assert (evaluation / "scores.json").read_bytes() == original[evaluation / "scores.json"]
    source = json.loads((evaluation / "run.json").read_bytes())
    assert source["clustering_completion_sha256"] == module._sha(
        (root / "results/clustering/A0/fold_1/repeat_1/completion.json").read_bytes())
    assert source["source"]["run_sha256"] == module._sha(
        (root / "results/neural/A0/fold_1/repeat_1/run.json").read_bytes())
    assert (root / "config/a1_contract_update.json").exists()
    assert not module.build_plan(root, scope="cv").changes


@pytest.mark.parametrize("problem", ["recipe", "code", "missing"])
def test_refresh_refuses_unreviewed_or_missing_sources_before_writing(
    saved: tuple[ModuleType, Path, dict], problem: str,
) -> None:
    module, root, _ = saved
    path = root / "results/neural/A0/fold_1/repeat_1/run.json"
    record = json.loads(path.read_bytes())
    if problem == "recipe":
        record["contract"]["config"]["training"]["epochs"] = 799
        write(path, record)
    elif problem == "code":
        record["contract"]["code_sha256"]["config.py"] = "unreviewed"
        write(path, record)
    else:
        path.unlink()
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    with pytest.raises(ValueError):
        module.build_plan(root, scope="cv")
    assert before == {p: p.read_bytes() for p in before}


def test_failed_update_rolls_back_all_rewritten_metadata(
    saved: tuple[ModuleType, Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, root, _ = saved
    plan = module.build_plan(root, scope="cv")
    before = {path: path.read_bytes() for path in plan.originals}
    replace = Path.replace

    def fail_once(path: Path, target: Path) -> Path:
        if path.name == "2.json":
            raise OSError("fixture write failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(OSError, match="fixture write"):
        plan.apply()
    assert before == {path: path.read_bytes() for path in before}
    assert not (root / "config/a1_contract_update.json").exists()


def test_completed_global_a1_map_uses_flat_record(tmp_path: Path) -> None:
    module = updater()
    plan = module.UpdatePlan(tmp_path, module.global_config())
    path = tmp_path / "results/clustering/A1/repeat_1/run.json"
    run_hash = write(path, {"schema_version": 1, "scope": "global_clustering", "condition": "A1",
                            "code_sha256": {"config.py": module._sha(
                                Path(config.__file__).read_bytes())}})
    write(path.with_name("completion.json"), {"status": "global_maps_completed",
                                               "run_sha256": run_hash})
    plan.completed_run(path, "global_maps_completed")
    assert not plan.changes
