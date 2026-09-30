"""Three global K=8 KMeans seeds without changing the completed neural fit."""

from __future__ import annotations

from .config import ADOPTED_SAMPLE_IDS, REPEATS, _seed
from .global_manifest import global_seed, global_seed_plan


def global_kmeans_seed(repeat: int) -> int:
    """Extend the recorded global scope; repeat 1 retains its original seed."""
    if type(repeat) is not int or repeat not in REPEATS:
        raise ValueError("Global repeat must be 1, 2, or 3")
    if repeat == 1:
        return global_seed("kmeans")
    return _seed("kmeans", "global", repeat, 8)


def global_kmeans_plan() -> list[dict[str, int]]:
    """List the three KMeans seeds and reject collisions before fitting."""
    plan = [
        {"repeat": repeat, "K": 8, "seed": global_kmeans_seed(repeat)}
        for repeat in REPEATS
    ]
    legacy = {row["seed"] for row in global_seed_plan(ADOPTED_SAMPLE_IDS)["records"]}
    if (len({row["seed"] for row in plan}) != len(plan)
            or any(row["seed"] in legacy for row in plan[1:])):
        raise ValueError("Global repeat seed collision")
    return plan
