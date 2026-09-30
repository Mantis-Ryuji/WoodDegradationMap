"""Small checks for frozen-global KMeans seeds and streamed score aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from chemomae.clustering.metric import silhouette_samples_cosine_gpu
from sklearn.metrics import silhouette_samples

from wood_degradation_map.experiments.global_k8_aggregate import (
    summarize_ari, summarize_repeats,
)
from wood_degradation_map.experiments.global_k8_repeats import (
    fit_repeat_clusters, load_repeat_clusters, repeat_paths, save_repeat_clusters,
)
from wood_degradation_map.experiments.global_manifest import global_seed
from wood_degradation_map.experiments.global_seed_plan import global_kmeans_plan, global_kmeans_seed
from wood_degradation_map.experiments.global_silhouette import PooledCosineSilhouette


def test_global_kmeans_seeds_preserve_existing_repeat() -> None:
    plan = global_kmeans_plan()
    assert [row["repeat"] for row in plan] == [1, 2, 3]
    assert [row["K"] for row in plan] == [8, 8, 8]
    assert plan[0]["seed"] == global_seed("kmeans")
    assert len({row["seed"] for row in plan}) == 3
    with pytest.raises(ValueError):
        global_kmeans_seed(4)


def test_three_seed_paths_do_not_use_legacy_maps(tmp_path: Path) -> None:
    results, centers = repeat_paths(tmp_path, "B0", 1)
    assert results == tmp_path / "results/global_k8_3seed_v1/clustering/B0/repeat_1"
    assert centers == tmp_path / "checkpoints/global_k8_3seed_v1/clustering/B0/repeat_1"


def test_pooled_silhouette_matches_reference_across_chunks() -> None:
    rng = np.random.default_rng(37)
    features = rng.normal(size=(21, 6)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.repeat(np.array([1, 2, 3], dtype=np.uint8), 7)
    pooled = PooledCosineSilhouette(8, 6)
    for start in (0, 5, 13):
        stop = {0: 5, 5: 13, 13: 21}[start]
        pooled.add(features[start:stop], labels[start:stop])
    assert pooled.finalize() is None
    actual = np.concatenate([pooled.score(features[:9], labels[:9]),
                             pooled.score(features[9:], labels[9:])])
    expected = silhouette_samples(features, labels, metric="cosine")
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    reference = silhouette_samples_cosine_gpu(
        features, labels.astype(np.int64), device="cpu", chunk=7,
        dtype=torch.float32, return_numpy=True, eps=1e-12,
    )
    np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)
    assert pooled.counts.tolist() == [7, 7, 7, 0, 0, 0, 0, 0]


def test_pooled_silhouette_records_single_cluster() -> None:
    values = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (3, 1))
    pooled = PooledCosineSilhouette(8, 2)
    pooled.add(values, np.ones(3, dtype=np.uint8))
    assert pooled.finalize() == "single_cluster"
    with pytest.raises(ValueError, match="undefined"):
        pooled.score(values, np.ones(3, dtype=np.uint8))


def test_isolated_kmeans_centers_reject_wrong_repeat(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(64, 256)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    fitted = fit_repeat_clusters(features, "B0", ("fixture",), 2, device=torch.device("cpu"))
    path = tmp_path / "centers_k8.npz"
    save_repeat_clusters(fitted, path, 2)
    restored = load_repeat_clusters(path, "B0", 2, device=torch.device("cpu"))
    assert restored.record.seed == global_kmeans_seed(2)
    np.testing.assert_array_equal(restored.centroids, fitted.centroids)
    np.testing.assert_array_equal(restored.predict(features), fitted.predict(features))
    with pytest.raises(ValueError, match="contract differs"):
        load_repeat_clusters(path, "B0", 3, device=torch.device("cpu"))


def test_three_repeat_summary_uses_only_common_samples_and_separates_sd() -> None:
    rows = {
        1: {"a": {"silhouette": 0.2}, "b": {"silhouette": 0.5},
            "c": {"silhouette": 0.9}},
        2: {"a": {"silhouette": 0.4}, "b": {"silhouette": 0.6},
            "c": {"silhouette": 0.8}},
        3: {"a": {"silhouette": 0.6}, "b": {"silhouette": 0.7},
            "c": {"silhouette": None, "undefined_reasons": {"silhouette": "single_cluster"}}},
    }
    summary = summarize_repeats(rows, "silhouette")
    assert summary["common_sample_ids"] == ["a", "b"]
    assert summary["excluded_sample_ids"] == ["c"]
    assert summary["availability"][3]["undefined"] == {"c": "single_cluster"}
    np.testing.assert_allclose(summary["repeat_macro_means"][1], 0.35)
    np.testing.assert_allclose(summary["repeat_macro_means"][2], 0.5)
    np.testing.assert_allclose(summary["repeat_macro_means"][3], 0.65)
    np.testing.assert_allclose(summary["mean"], 0.5)
    np.testing.assert_allclose(summary["sample_sd"], np.std([0.4, 0.6], ddof=1))
    np.testing.assert_allclose(summary["repeat_sd"], np.std([0.35, 0.5, 0.65], ddof=1))
    ari = summarize_ari({"a": 0.2, "b": 0.6, "c": None})
    assert ari["repeat_sd"] is None and ari["common_samples"] == 2
