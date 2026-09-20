"""Render/check the single static 2x5 PNG from fixed global PCA coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wood_degradation_map.experiments.global_pca_reporting import check_pca_figure, render_pca


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check"))
    parser.add_argument("--experiment-dir", type=Path, default=root / "outputs/experiments/global_v1")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error("Require positive DPI")
    return args


def main() -> int:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    result = (render_pca(experiment, dpi=args.dpi) if args.action == "run"
              else check_pca_figure(experiment))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
