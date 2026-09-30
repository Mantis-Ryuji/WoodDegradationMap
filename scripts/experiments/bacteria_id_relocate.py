"""Inventory, relocate, and verify completed Bacteria-ID experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[2] / "outputs" / "experiments"
OLD = BASE / "bacteria_id_v1"
OLD_HPO = BASE / "bacteria_id_hpo_cv_v1"
LEGACY = BASE / "bacteria_id_table4_60_20_20_legacy_v1"
PRETRAIN = BASE / "bacteria_id_table5_reference30_pretrain_v1"
HPO = BASE / "bacteria_id_table5_hpo_cv_v1"
TABLE4 = BASE / "bacteria_id_table4_80_20_v1"
INVENTORY = BASE / "bacteria_id_relocation_inventory_20260930.json"
MANIFEST_NAME = "relocation_manifest.json"
CONDITIONS = ("M00", "M11")
CORPORA = ("bacteria4", "bacteria6")
SEEDS = range(5)
LEGACY_PREFIXES = tuple(
    f"{section}/pretrain/{corpus}/"
    for section in ("results", "checkpoints") for corpus in CORPORA
) + ("results/table4/", "checkpoints/table4/")
PRETRAIN_PREFIXES = (
    "results/pretrain/reference30/", "checkpoints/pretrain/reference30/",
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Missing ordinary artifact directory: {root}")
    files: list[Path] = []
    for directory, subdirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in (*subdirs, *names):
            path = parent / name
            if path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"Junction or symlink in artifact tree: {path}")
            if not path.is_dir() and not path.is_file():
                raise ValueError(f"Unsupported artifact entry: {path}")
        files.extend(parent / name for name in names)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _inventory(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size, "sha256": _sha256(path),
        }
        for path in _paths(root)
    }


def _require_hash(
    inventory: dict[str, dict[str, object]], relative: str, expected: object,
) -> None:
    actual = inventory.get(relative, {}).get("sha256")
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"Recorded SHA-256 differs: {relative}")


def _require_layout() -> None:
    for target in (LEGACY, PRETRAIN, HPO):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Relocation target already exists: {target}")
    old_top = {path.name for path in OLD.iterdir()}
    if old_top != {"results", "checkpoints", "table4.md"}:
        raise ValueError(f"Unexpected old Bacteria-ID top level: {sorted(old_top)}")
    for section in ("results", "checkpoints"):
        actual = {path.name for path in (OLD / section).iterdir()}
        if actual != {"pretrain", "table4"}:
            raise ValueError(f"Unexpected {section} layout: {sorted(actual)}")
        actual = {path.name for path in (OLD / section / "pretrain").iterdir()}
        if actual != {*CORPORA, "reference30"}:
            raise ValueError(f"Unexpected {section}/pretrain layout: {sorted(actual)}")
        actual = {path.name for path in (OLD / section / "table4").iterdir()}
        if actual != set(CORPORA):
            raise ValueError(f"Unexpected {section}/table4 layout: {sorted(actual)}")
    hpo_top = {path.name for path in OLD_HPO.iterdir()}
    expected_hpo = {
        "cv", "evaluation", "final", "cv_splits.json", "selected.json",
        "study.sqlite3", "table5_statistics.json", "table5.md",
    }
    if hpo_top != expected_hpo:
        raise ValueError(f"Unexpected Table 5 top level: {sorted(hpo_top)}")
    expected_table4 = {
        "checkpoints", "results", "manifest.json", "splits.npz",
        "table4_seed_scores.csv", "table4_seed_scores.png",
        "table4_statistics.json", "table4.md",
    }
    actual_table4 = {path.name for path in TABLE4.iterdir()}
    if actual_table4 != expected_table4:
        raise ValueError(f"Unexpected new Table 4 top level: {sorted(actual_table4)}")
    for root in (OLD, OLD_HPO, TABLE4):
        _paths(root)


def _partition(
    old: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    legacy = {
        name: record for name, record in old.items()
        if name == "table4.md" or name.startswith(LEGACY_PREFIXES)
    }
    pretrain = {
        name: record for name, record in old.items()
        if name.startswith(PRETRAIN_PREFIXES)
    }
    if set(legacy) & set(pretrain) or set(legacy) | set(pretrain) != set(old):
        raise ValueError("Unclassified or overlapping old Bacteria-ID artifacts")
    return legacy, pretrain


def _validate_records(
    old: dict[str, dict[str, object]],
    hpo: dict[str, dict[str, object]],
    table4: dict[str, dict[str, object]],
) -> dict[str, int]:
    for corpus in (*CORPORA, "reference30"):
        for condition in CONDITIONS:
            for seed in SEEDS:
                suffix = f"{corpus}/{condition}/seed_{seed}"
                completion = _read_json(OLD / "results/pretrain" / suffix / "completion.json")
                if completion.get("status") != "training_completed" or completion.get("completed_epochs") != 800:
                    raise ValueError(f"Incomplete original pretraining: {suffix}")
                weight = f"checkpoints/pretrain/{suffix}/last_model.pt"
                _require_hash(old, weight, completion.get("weights_sha256"))
                if corpus == "reference30":
                    continue
                score = _read_json(OLD / "results/table4" / suffix / "metrics.json")
                if score.get("status") != "completed":
                    raise ValueError(f"Incomplete legacy Table 4 score: {suffix}")
                _require_hash(old, weight, score.get("pretrain_weights_sha256"))
                _require_hash(
                    old, f"checkpoints/table4/{suffix}/centroids.npz",
                    score.get("centroids_sha256"),
                )
                new_completion = _read_json(TABLE4 / "results/pretrain" / suffix / "completion.json")
                new_weight = f"checkpoints/pretrain/{suffix}/last_model.pt"
                if (new_completion.get("status") != "training_completed"
                        or new_completion.get("completed_epochs") != 800):
                    raise ValueError(f"Incomplete new Table 4 pretraining: {suffix}")
                _require_hash(table4, new_weight, new_completion.get("weights_sha256"))
                new_score = _read_json(TABLE4 / "results/table4" / suffix / "metrics.json")
                if new_score.get("status") != "completed":
                    raise ValueError(f"Incomplete new Table 4 score: {suffix}")
                _require_hash(table4, new_weight, new_score.get("pretrain_weights_sha256"))
                _require_hash(
                    table4, f"checkpoints/table4/{suffix}/centroids.npz",
                    new_score.get("centroids_sha256"),
                )

    selected = _read_json(OLD_HPO / "selected.json")
    statistics = _read_json(OLD_HPO / "table5_statistics.json")
    policy = selected.get("selection_policy")
    trials = statistics.get("search_trials")
    if (not isinstance(policy, dict) or not isinstance(trials, list)
            or selected.get("study_contract_sha256") != statistics.get("study_contract_sha256")
            or selected.get("selected_trial_number") != statistics.get("selected_trial_number")
            or policy.get("completed_settings") != len(policy.get("trial_numbers", []))
            or len(trials) < policy["completed_settings"]):
        raise ValueError("Table 5 selection and study summary disagree")
    selected_trial = selected["selected_trial_number"]
    if selected_trial not in policy["trial_numbers"]:
        raise ValueError("Selected trial is outside the declared selection set")
    with closing(sqlite3.connect(
        f"file:{(OLD_HPO / 'study.sqlite3').as_posix()}?mode=ro", uri=True,
    )) as connection:
        complete = connection.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'").fetchone()[0]
    if complete != len(trials):
        raise ValueError("SQLite COMPLETE trials and Table 5 summary disagree")
    for condition in CONDITIONS:
        for seed in SEEDS:
            suffix = f"{condition}/seed_{seed}"
            final = _read_json(OLD_HPO / "final" / suffix / "metrics.json")
            evaluation = _read_json(OLD_HPO / "evaluation" / suffix / "metrics.json")
            weight = f"final/{suffix}/last_model.pt"
            if (final.get("status") != "completed" or final.get("completed_epochs") != 50
                    or evaluation.get("status") != "completed"
                    or evaluation.get("selected_trial_number") != selected_trial
                    or evaluation.get("test_count") != 3000):
                raise ValueError(f"Incomplete Table 5 final/evaluation: {suffix}")
            _require_hash(hpo, weight, final.get("final_weights_sha256"))
            _require_hash(hpo, weight, evaluation.get("final_weights_sha256"))
    return {
        "legacy_table4_scores": 20,
        "legacy_pretraining_runs": 30,
        "new_table4_scores": 20,
        "new_table4_pretraining_runs": 20,
        "reference30_pretraining_runs": 10,
        "hpo_completed_settings": complete,
        "hpo_selected_from_first": policy["completed_settings"],
        "table5_final_runs": 10,
        "table5_evaluations": 10,
    }


def _exclusive_json(path: Path, value: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, ensure_ascii=False)
        target.write("\n")


def _moves() -> list[tuple[Path, Path]]:
    steps = [
        (OLD / section / "pretrain" / corpus,
         LEGACY / section / "pretrain" / corpus)
        for section in ("results", "checkpoints") for corpus in CORPORA
    ]
    steps.extend([
        (OLD / "results/table4", LEGACY / "results/table4"),
        (OLD / "checkpoints/table4", LEGACY / "checkpoints/table4"),
        (OLD / "table4.md", LEGACY / "table4.md"),
        (OLD, PRETRAIN), (OLD_HPO, HPO),
    ])
    return steps


def _same_inventory(
    actual: dict[str, dict[str, object]], expected: dict[str, dict[str, object]],
    label: str,
) -> None:
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
        raise ValueError(f"{label} inventory differs: missing={missing}, unexpected={unexpected}, changed={changed}")


def _apply() -> None:
    _require_layout()
    old = _inventory(OLD)
    old_hpo = _inventory(OLD_HPO)
    new_table4 = _inventory(TABLE4)
    legacy, pretrain = _partition(old)
    counts = _validate_records(old, old_hpo, new_table4)
    timestamp = datetime.now(timezone.utc).isoformat()
    before: dict[str, object] = {
        "schema_version": 1, "recorded_utc": timestamp,
        "counts": counts,
        "sources": {
            str(OLD.resolve()): old,
            str(OLD_HPO.resolve()): old_hpo,
            str(TABLE4.resolve()): new_table4,
        },
        "destinations": {
            str(LEGACY.absolute()): sorted(legacy),
            str(PRETRAIN.absolute()): sorted(pretrain),
            str(HPO.absolute()): sorted(old_hpo),
        },
    }
    if INVENTORY.exists():
        previous = _read_json(INVENTORY)
        if any(previous.get(key) != before[key] for key in ("counts", "sources", "destinations")):
            raise ValueError(f"Existing pre-move inventory differs: {INVENTORY}")
    else:
        _exclusive_json(INVENTORY, before)
    moved: list[tuple[Path, Path]] = []
    written_manifests: list[Path] = []
    try:
        for source, target in _moves():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Relocation target appeared during move: {target}")
            source.rename(target)
            moved.append((source, target))
        _same_inventory(_inventory(LEGACY), legacy, "legacy Table 4")
        _same_inventory(_inventory(PRETRAIN), pretrain, "reference30 pretraining")
        _same_inventory(_inventory(HPO), old_hpo, "Table 5")
        _same_inventory(_inventory(TABLE4), new_table4, "unchanged new Table 4")
        for source, target, items in (
            (OLD, LEGACY, legacy), (OLD, PRETRAIN, pretrain), (OLD_HPO, HPO, old_hpo),
        ):
            manifest = target / MANIFEST_NAME
            _exclusive_json(manifest, {
                "schema_version": 1, "relocated_utc": timestamp,
                "original_root": str(source.absolute()),
                "current_root": str(target.absolute()),
                "pre_move_inventory": str(INVENTORY.absolute()),
                "files": items,
            })
            written_manifests.append(manifest)
    except Exception:
        for manifest in written_manifests:
            manifest.unlink()
        for source, target in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            target.rename(source)
        for parent in (
            LEGACY / "results/pretrain", LEGACY / "checkpoints/pretrain",
            LEGACY / "results", LEGACY / "checkpoints", LEGACY,
        ):
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        raise
    print(json.dumps({"status": "relocated", "counts": counts,
                      "destinations": [str(LEGACY), str(PRETRAIN), str(HPO)]}, indent=2))


def _dry_run() -> None:
    _require_layout()
    old = _paths(OLD)
    old_hpo = _paths(OLD_HPO)
    new_table4 = _paths(TABLE4)
    old_names = {path.relative_to(OLD).as_posix() for path in old}
    legacy_names = {name for name in old_names if name == "table4.md" or name.startswith(LEGACY_PREFIXES)}
    pretrain_names = {name for name in old_names if name.startswith(PRETRAIN_PREFIXES)}
    if legacy_names & pretrain_names or legacy_names | pretrain_names != old_names:
        raise ValueError("Unclassified or overlapping old Bacteria-ID artifacts")
    print(json.dumps({
        "status": "dry-run", "moves": [
            {"source": str(source.absolute()), "target": str(target.absolute())}
            for source, target in _moves()
        ],
        "file_counts": {"legacy": len(legacy_names), "reference30": len(pretrain_names),
                        "table5": len(old_hpo), "unchanged_new_table4": len(new_table4)},
    }, indent=2))


def _verify() -> None:
    if OLD.exists() or OLD_HPO.exists() or not INVENTORY.is_file():
        raise ValueError("Relocation is incomplete or old output paths remain")
    before = _read_json(INVENTORY)
    sources = before["sources"]
    relocated: dict[Path, dict[str, dict[str, object]]] = {}
    for source, target in ((OLD, LEGACY), (OLD, PRETRAIN), (OLD_HPO, HPO)):
        manifest = _read_json(target / MANIFEST_NAME)
        if manifest.get("original_root") != str(source.absolute()) or manifest.get("current_root") != str(target.absolute()):
            raise ValueError(f"Incorrect relocation manifest: {target}")
        expected = manifest["files"]
        actual = _inventory(target)
        actual.pop(MANIFEST_NAME, None)
        _same_inventory(actual, expected, str(target))
        relocated[target] = actual
        original = sources[str(source.absolute())]
        if any(original.get(name) != record for name, record in expected.items()):
            raise ValueError(f"Relocation manifest differs from pre-move inventory: {target}")
    _same_inventory(_inventory(TABLE4), sources[str(TABLE4.absolute())], "unchanged new Table 4")
    selected = _read_json(HPO / "selected.json")
    statistics = _read_json(HPO / "table5_statistics.json")
    trials = statistics.get("search_trials")
    policy = selected.get("selection_policy")
    if (not isinstance(trials, list) or not isinstance(policy, dict)
            or selected.get("study_contract_sha256") != statistics.get("study_contract_sha256")
            or selected.get("selected_trial_number") != statistics.get("selected_trial_number")
            or len(trials) != before["counts"]["hpo_completed_settings"]
            or policy.get("completed_settings") != before["counts"]["hpo_selected_from_first"]):
        raise ValueError("Relocated Table 5 study summary differs from selected setting")
    with closing(sqlite3.connect(
        f"file:{(HPO / 'study.sqlite3').as_posix()}?mode=ro", uri=True,
    )) as connection:
        complete = connection.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'").fetchone()[0]
    if complete != len(trials):
        raise ValueError("Relocated SQLite study differs from Table 5 summary")
    for condition in CONDITIONS:
        for seed in SEEDS:
            suffix = f"reference30/{condition}/seed_{seed}"
            completion = _read_json(PRETRAIN / "results/pretrain" / suffix / "completion.json")
            _require_hash(
                relocated[PRETRAIN], f"checkpoints/pretrain/{suffix}/last_model.pt",
                completion.get("weights_sha256"),
            )
            final_suffix = f"{condition}/seed_{seed}"
            final = _read_json(HPO / "final" / final_suffix / "metrics.json")
            evaluation = _read_json(HPO / "evaluation" / final_suffix / "metrics.json")
            weight = f"final/{final_suffix}/last_model.pt"
            _require_hash(relocated[HPO], weight, final.get("final_weights_sha256"))
            _require_hash(relocated[HPO], weight, evaluation.get("final_weights_sha256"))
            if evaluation.get("selected_trial_number") != selected["selected_trial_number"]:
                raise ValueError(f"Relocated evaluation uses a different trial: {final_suffix}")
    print(json.dumps({"status": "verified", "counts": before["counts"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="Record inventories and relocate")
    group.add_argument("--verify", action="store_true", help="Verify relocated inventories")
    args = parser.parse_args()
    if args.apply:
        _apply()
    elif args.verify:
        _verify()
    else:
        _dry_run()


if __name__ == "__main__":
    main()
