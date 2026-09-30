"""Exact pooled cosine silhouette with bounded feature memory for global maps."""

from __future__ import annotations

import numpy as np


class PooledCosineSilhouette:
    """Collect cluster vector sums, then score the same pixels in a second pass.

    Cosine distance to a cluster is one minus the query dot its mean unit vector.
    Excluding the query from its own cluster needs only the cluster sum and size.
    Accumulating sums in FP64 avoids error over millions of global pixels; input
    representations and returned scores remain FP32. This never builds N x N data.
    """

    def __init__(self, k: int, dimension: int) -> None:
        if type(k) is not int or k < 2 or type(dimension) is not int or dimension < 1:
            raise ValueError("Expected positive dimension and K >= 2")
        self.k = k
        self.dimension = dimension
        self.counts = np.zeros(k, dtype=np.int64)
        self.sums = np.zeros((k, dimension), dtype=np.float64)
        self._reason: str | None = None

    def _unit(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if (not isinstance(values, np.ndarray) or values.dtype != np.float32
                or values.ndim != 2 or values.shape[1] != self.dimension
                or not isinstance(labels, np.ndarray) or labels.shape != (len(values),)
                or labels.dtype.kind not in "iu" or not len(values)
                or np.any(labels < 1) or np.any(labels > self.k)
                or not np.isfinite(values).all()):
            raise ValueError("Expected nonempty finite FP32 features and matching labels in 1..K")
        unit = values.astype(np.float64)
        norms = np.linalg.norm(unit, axis=1)
        if not np.isfinite(norms).all() or np.any(norms < 1e-12):
            raise ValueError("Zero or nonfinite cosine feature")
        unit /= norms[:, None]
        return unit

    def add(self, values: np.ndarray, labels: np.ndarray) -> None:
        """Accumulate one first-pass chunk in its original cluster numbering."""
        if self._reason is not None:
            raise ValueError("Pooled silhouette was already finalized")
        unit = self._unit(values, labels)
        indices = labels.astype(np.int64) - 1
        self.counts += np.bincount(indices, minlength=self.k)
        for cluster in np.unique(indices):
            self.sums[cluster] += unit[indices == cluster].sum(axis=0)

    def finalize(self) -> str | None:
        """Return the pooled undefined reason, if any."""
        n = int(self.counts.sum())
        if n == 0:
            raise ValueError("No global pixels were supplied")
        used = int(np.count_nonzero(self.counts))
        self._reason = "single_cluster" if used == 1 else "all_singletons" if used == n else "defined"
        return None if self._reason == "defined" else self._reason

    def score(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Score one second-pass chunk against all first-pass global pixels."""
        if self._reason != "defined":
            raise ValueError("Pooled silhouette is undefined or not finalized")
        unit = self._unit(values, labels)
        indices = labels.astype(np.int64) - 1
        dots = unit @ self.sums.T
        counts = self.counts[indices]
        within = np.zeros(len(values), dtype=np.float64)
        multiple = counts > 1
        rows = np.arange(len(values))[multiple]
        within[multiple] = 1 - (dots[rows, indices[multiple]] - 1) / (counts[multiple] - 1)
        within = np.maximum(within, 0)
        between = np.full(dots.shape, np.inf, dtype=np.float64)
        nonempty = self.counts > 0
        between[:, nonempty] = 1 - dots[:, nonempty] / self.counts[nonempty]
        between = np.maximum(between, 0)
        between[np.arange(len(values)), indices] = np.inf
        nearest = between.min(axis=1)
        denominator = np.maximum(within, nearest)
        scores = np.divide(nearest - within, denominator,
                           out=np.zeros(len(values), dtype=np.float64), where=denominator > 0)
        scores[~multiple] = 0
        if not np.isfinite(scores).all() or np.any(scores < -1 - 1e-6) or np.any(scores > 1 + 1e-6):
            raise ValueError("Invalid pooled cosine silhouette score")
        return np.clip(scores, -1, 1).astype(np.float32)
