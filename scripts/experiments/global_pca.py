"""Prepare global representations and fit/check centered CPU PCA for visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wood_degradation_map.experiments.global_pca import check_pca, run_pca


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fit", "check"))
    parser.add_argument("--experiment-dir", type=Path, default=root / "outputs/experiments/global_v1")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    parser.add_argument("--device", type=int, default=0, help="CUDA device for prepare only")
    parser.add_argument("--chunk-pixels", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", help="Verify/skip completed outputs")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate and replace existing PCA inputs or fits")
    args = parser.parse_args(argv)
    if args.device < 0 or args.chunk_pixels < 1:
        parser.error("Require nonnegative device and positive chunk size")
    if args.resume and args.action not in ("prepare", "fit"):
        parser.error("--resume is for prepare/fit only")
    if args.overwrite and (args.action == "check" or args.resume):
        parser.error("--overwrite is for prepare/fit and cannot be combined with --resume")
    return args


def main() -> int:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    if args.action == "prepare":
        # Saved fits retain their host paths and runtime contract; export there first.
        import torch

        from wood_degradation_map.experiments.global_pca_inputs import prepare_inputs

        if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
            raise RuntimeError("A valid CUDA device is required for representation extraction")
        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
        result = prepare_inputs(experiment, args.processed_dir, args.metadata, device=device,
                                chunk_pixels=args.chunk_pixels, resume=args.resume,
                                overwrite=args.overwrite)
    elif args.action == "fit":
        result = run_pca(experiment, resume=args.resume, overwrite=args.overwrite)
    else:
        result = check_pca(experiment)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
