"""Check a per-sample pixel budget using saved preprocessing metadata only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_integer_dtype


def positive_int(value: str) -> int:
    """Parse a positive integer for the command-line pixel budget."""
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("q must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("q must be a positive integer")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--sample-quality",
        type=Path,
        default=(
            project_root
            / "data/processed/preprocessing/200hz_snr10_linear256/sample_quality.parquet"
        ),
        help="Existing sample_quality.parquet; spectra and HDF5 files are not read.",
    )
    parser.add_argument(
        "--q",
        type=positive_int,
        default=8192,
        help="Proposed number of pixels per sample (default: 8192).",
    )
    return parser.parse_args()


def load_pixel_counts(path: Path) -> pd.DataFrame:
    """Read and validate the saved per-sample pixel counts.

    Parameters
    ----------
    path : pathlib.Path
        Existing preprocessing sample-quality table.

    Returns
    -------
    pandas.DataFrame
        Unique sample IDs and nonnegative integer counts, sorted by count and ID.

    Raises
    ------
    ValueError
        The table is empty or IDs/counts violate the preprocessing contract.
    """
    table = pd.read_parquet(path, columns=["sample_id", "saved_pixel_count"])
    if table.empty:
        raise ValueError("The sample-quality table is empty.")

    sample_ids = table["sample_id"]
    if not sample_ids.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("sample_id must contain nonempty strings without missing values.")
    duplicates = sample_ids[sample_ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate sample_id entries: {duplicates}")

    counts = table["saved_pixel_count"]
    if counts.isna().any() or not is_integer_dtype(counts.dtype):
        raise ValueError("saved_pixel_count must contain integers without missing values.")
    if (counts < 0).any():
        raise ValueError("saved_pixel_count must be nonnegative.")

    return table.sort_values(["saved_pixel_count", "sample_id"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    table = load_pixel_counts(args.sample_quality)
    counts = table["saved_pixel_count"]
    enough = counts >= args.q

    report = table.rename(columns={"saved_pixel_count": "valid_pixels"}).copy()
    report["q"] = args.q
    # A percentage above 100 makes an infeasible request visible; zero has no ratio.
    report["requested_percent"] = 100.0 * args.q / counts.where(counts > 0)
    report["shortfall"] = counts.map(lambda count: max(args.q - int(count), 0))
    report["status"] = enough.map({True: "OK", False: "INSUFFICIENT"})

    print(f"Source: {args.sample_quality.resolve()}")
    print(f"Samples: {len(table):,}; total valid pixels: {sum(int(n) for n in counts):,}")
    print(
        f"Valid pixels per sample: min={int(counts.min()):,}, "
        f"median={counts.median():,.1f}, max={int(counts.max()):,}"
    )
    print(f"Requested q: {args.q:,}; feasible samples: {int(enough.sum())}/{len(table)}")
    print()
    print(
        report.to_string(
            index=False,
            na_rep="N/A",
            formatters={"requested_percent": lambda value: f"{value:.2f}%"},
        )
    )
    print()

    if not enough.all():
        print("FAIL: Some samples have fewer than q valid pixels; see INSUFFICIENT rows.")
        print(f"Largest common count allowed by this table: {int(counts.min()):,}")
        return 1

    print("PASS: Every listed sample has at least q valid pixels.")
    print("This checks pixel availability, not spectral coverage or model performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
