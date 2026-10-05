"""Export mask-rate OOF CSVs, optionally with a four-panel K-sweep PNG (CPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from wood_degradation_map.experiments.manifests import _read_json  # noqa: E402
from wood_degradation_map.experiments.mask_rate_reporting import (  # noqa: E402
    export_mask_rate_report,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path,
                        default=root / "outputs/experiments/production_v1")
    parser.add_argument("--snapshot", default="mask_rate_oof_v1")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: experiment/results/figures/SNAPSHOT")
    parser.add_argument("--with-figure", action="store_true",
                        help="Also save the 2x2 LLA-3/5/9 and LFR(TGN+FS) K-sweep PNG")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace generated report files; preserve other files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir or args.experiment_dir / "results/figures" / args.snapshot
    result = export_mask_rate_report(args.experiment_dir, args.snapshot, output,
                                     overwrite=args.overwrite, with_figure=args.with_figure,
                                     dpi=args.dpi)
    print(json.dumps({"output_dir": str(result), **_read_json(result / "completion.json")},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
