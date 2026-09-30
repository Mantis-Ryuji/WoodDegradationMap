"""Small fixtures for the fixed 80/20 Table 4 split and five-seed summary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wood_degradation_map.experiments import bacteria_table4_80
from wood_degradation_map.experiments.bacteria_config import TABLE4_CLASSES, bacteria_config
from wood_degradation_map.experiments.bacteria_data import _table4_split
from wood_degradation_map.experiments.bacteria_table4_80 import (
    Table4Data, derive_table4_splits, discard_incomplete_score, prepare_table4_80,
    summarize_scores, table4_config,
)


def _saved_legacy_splits() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = np.repeat(np.arange(6, dtype=np.int64), 2000)
    old: dict[str, np.ndarray] = {}
    for corpus, ids in TABLE4_CLASSES.items():
        train, validation, test = _table4_split(labels, ids)
        old[f"{corpus}_train"] = train
        old[f"{corpus}_validation"] = validation
        old[f"{corpus}_test"] = test[::-1].copy()  # Valid saved order must remain unchanged.
    return labels, old


def test_union_preserves_saved_test_order_and_covers_each_class() -> None:
    labels, old = _saved_legacy_splits()
    new = derive_table4_splits(labels, old)
    assert set(new) == {"bacteria4_train", "bacteria4_test", "bacteria6_train", "bacteria6_test"}
    for corpus, ids in TABLE4_CLASSES.items():
        train, test = new[f"{corpus}_train"], new[f"{corpus}_test"]
        np.testing.assert_array_equal(train, np.sort(np.concatenate((
            old[f"{corpus}_train"], old[f"{corpus}_validation"],
        ))))
        np.testing.assert_array_equal(test, old[f"{corpus}_test"])
        assert len(train) == len(ids) * 1600
        assert len(test) == len(ids) * 400
        assert np.intersect1d(train, test).size == 0
        np.testing.assert_array_equal(
            np.sort(np.concatenate((train, test))), np.flatnonzero(np.isin(labels, ids)),
        )
        for label in ids:
            assert np.count_nonzero(labels[train] == label) == 1600
            assert np.count_nonzero(labels[test] == label) == 400


def test_legacy_overlap_is_rejected_before_preparation() -> None:
    labels, old = _saved_legacy_splits()
    old["bacteria4_validation"][0] = old["bacteria4_train"][0]
    with pytest.raises(ValueError, match="partition every target source row"):
        derive_table4_splits(labels, old)


def test_persisted_test_order_is_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels, old = _saved_legacy_splits()
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "manifest.json").write_text("{}", encoding="utf-8")
    (processed / "splits.npz").write_bytes(b"fixture split identity")
    prepared = SimpleNamespace(root=processed, labels={"reference": labels}, splits=old)
    monkeypatch.setattr(bacteria_table4_80, "PreparedBacteria", lambda _: prepared)
    output = tmp_path / "table4_80"
    prepare_table4_80(processed, output)
    Table4Data(processed, output)
    with np.load(output / "splits.npz", allow_pickle=False) as saved:
        changed = {name: saved[name].copy() for name in saved.files}
    changed["bacteria4_test"][:2] = changed["bacteria4_test"][:2][::-1]
    np.savez_compressed(output / "splits.npz", **changed)
    with pytest.raises(ValueError, match="80/20 source rows differ"):
        Table4Data(processed, output)


def test_new_config_keeps_training_and_clustering_recipe() -> None:
    original = bacteria_config()
    updated = table4_config()
    assert updated["table4"]["split"] == [0.8, 0.2]
    assert updated["table4"]["validation"] is None
    assert "table5" not in updated
    assert updated["model"] == original["model"]
    assert updated["pretrain"] == original["pretrain"]
    assert updated["augmentation"] == original["augmentation"]
    assert updated["table4"]["clustering"] == original["table4"]["clustering"]


def test_summary_uses_all_five_seeds_and_sample_sd() -> None:
    values = np.array([[1.0, 11.0, 21.0], [2.0, 12.0, 22.0],
                       [3.0, 13.0, 23.0], [4.0, 14.0, 24.0],
                       [5.0, 15.0, 25.0]])
    result = summarize_scores(values)
    assert result["ACC"]["mean"] == 3.0
    assert result["NMI"]["mean"] == 13.0
    assert result["AMI"]["mean"] == 23.0
    assert result["ACC"]["sample_sd"] == pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError, match="five finite"):
        summarize_scores(values[:4])


def test_only_centroids_only_failed_score_is_discarded(tmp_path: Path) -> None:
    results, weights = tmp_path / "results", tmp_path / "weights"
    results.mkdir()
    weights.mkdir()
    centroids = weights / "centroids.npz"
    centroids.write_bytes(b"incomplete score")
    assert discard_incomplete_score(results, weights)
    assert not results.exists() and not weights.exists()

    results.mkdir()
    weights.mkdir()
    centroids.write_bytes(b"incomplete score")
    (results / "keep.txt").write_text("unrecognized output", encoding="utf-8")
    assert not discard_incomplete_score(results, weights)
    assert centroids.is_file() and (results / "keep.txt").is_file()
