"""Create paired all-sample and representative overview PNGs from saved images."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")

from wood_degradation_map.experiments.oof_sanity import REPRESENTATIVES  # noqa: E402

GRID_SIZE = 7
IMAGE_WIDTH = 720
GUTTER = 24
LABEL_HEIGHT = 128
FONT_SIZE = 80
COLORBAR_HEIGHT = 360
DPI = 240


@dataclass(frozen=True)
class ReflectanceScale:
    """The persisted color mapping used to generate the source reflectance PNGs."""

    cmap: str
    vmin: float
    vmax: float


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reflectance-dir", type=Path,
        default=root / "outputs/preprocessing/production_v1/reflectance_l2_norm",
    )
    parser.add_argument("--raw-dir", type=Path, default=root / "data/raw")
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "outputs/sample_overviews",
        help="Overview PNG directory; must be new unless --overwrite is supplied.",
    )
    parser.add_argument(
        "--representatives-only", action="store_true",
        help="Render only the two representative 1x7 PNGs, leaving all-sample grids untouched.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow an existing output directory and replace the selected overview PNGs.",
    )
    return parser.parse_args()


def paired_sources(reflectance_dir: Path, raw_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Require exactly 49 distinct sample IDs with matching PNG and BMP sources."""
    for directory in (reflectance_dir, raw_dir):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    reflectance = {path.stem: path for path in reflectance_dir.glob("KYOw*.png")}
    raw = {path.stem.removeprefix("RGB_"): path for path in raw_dir.glob("RGB_KYOw*.bmp")}
    if len(reflectance) != GRID_SIZE**2:
        raise ValueError(f"Expected 49 reflectance images, found {len(reflectance)}")
    if reflectance.keys() != raw.keys():
        raise ValueError(
            f"Sample IDs differ: missing BMPs={sorted(reflectance.keys() - raw.keys())}; "
            f"extra BMPs={sorted(raw.keys() - reflectance.keys())}"
        )
    return reflectance, raw


def image_box_height(sources: tuple[dict[str, Path], dict[str, Path]]) -> int:
    """Read image headers to choose a shared cell size without cropping either view."""
    heights = []
    for collection in sources:
        for path in collection.values():
            with Image.open(path) as source:
                heights.append(round(IMAGE_WIDTH * source.height / source.width))
    return max(heights)


def load_reflectance_scale(reflectance_dir: Path) -> ReflectanceScale:
    """Use saved display limits rather than estimating scalar values from RGB pixels."""
    path = reflectance_dir.parent / "report_config.json"
    display = json.loads(path.read_text(encoding="utf-8"))["reflectance_l2_norm"]
    scale = ReflectanceScale(
        cmap=display["cmap"], vmin=float(display["vmin"]), vmax=float(display["vmax"]),
    )
    if display["scope"] != "all samples":
        raise ValueError(f"Expected one shared reflectance scale for all samples: {path}")
    if not (math.isfinite(scale.vmin) and math.isfinite(scale.vmax) and scale.vmin < scale.vmax):
        raise ValueError(f"Invalid reflectance display limits: {path}")
    if scale.cmap not in matplotlib.colormaps:
        raise ValueError(f"Unknown reflectance colormap: {scale.cmap}")
    return scale


def paste_colorbar(canvas: Image.Image, scale: ReflectanceScale) -> None:
    """Add one horizontal scalar colorbar centered below the entire sample grid."""
    figure = Figure(figsize=(canvas.width / DPI, COLORBAR_HEIGHT / DPI), dpi=DPI)
    FigureCanvasAgg(figure)
    axis = figure.add_axes((0.18, 0.60, 0.64, 0.16))
    mappable = matplotlib.cm.ScalarMappable(
        norm=Normalize(vmin=scale.vmin, vmax=scale.vmax), cmap=scale.cmap,
    )
    integer_ticks = np.arange(math.floor(scale.vmin) + 1, math.ceil(scale.vmax), dtype=int)
    ticks = np.concatenate(([scale.vmin], integer_ticks, [scale.vmax]))
    bar = figure.colorbar(mappable, cax=axis, orientation="horizontal", ticks=ticks)
    bar.set_ticklabels(
        [f"{scale.vmin:.2f}", *(str(tick) for tick in integer_ticks), f"{scale.vmax:.2f}"]
    )
    bar.ax.tick_params(labelsize=FONT_SIZE * 72 / DPI, length=5, pad=5)
    bar.set_label("Reflectance L2 norm", fontsize=FONT_SIZE * 72 / DPI, labelpad=8)
    with BytesIO() as buffer:
        figure.savefig(buffer, format="png", dpi=DPI, facecolor="white")
        buffer.seek(0)
        with Image.open(buffer) as colorbar:
            canvas.paste(colorbar, (0, canvas.height - COLORBAR_HEIGHT))
    figure.clear()


def render_overview(
    sources: dict[str, Path], output: Path, *, sample_ids: list[str], image_height: int,
    reflectance_scale: ReflectanceScale | None = None, overwrite: bool = False,
) -> None:
    """Lay out full source images in the supplied order with large IDs underneath."""
    if len(sample_ids) not in (GRID_SIZE, GRID_SIZE**2) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Expected 7 or 49 unique sample IDs")
    missing = set(sample_ids) - sources.keys()
    if missing:
        raise ValueError(f"Missing source images: {sorted(missing)}")
    rows = len(sample_ids) // GRID_SIZE
    cell_height = image_height + LABEL_HEIGHT
    size = (
        GRID_SIZE * IMAGE_WIDTH + (GRID_SIZE + 1) * GUTTER,
        rows * cell_height + (rows + 1) * GUTTER
        + (COLORBAR_HEIGHT if reflectance_scale is not None else 0),
    )
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(str(font_path), size=FONT_SIZE)
    with Image.new("RGB", size, "white") as canvas:
        draw = ImageDraw.Draw(canvas)
        for index, sample_id in enumerate(sample_ids):
            row, column = divmod(index, GRID_SIZE)
            x = GUTTER + column * (IMAGE_WIDTH + GUTTER)
            y = GUTTER + row * (cell_height + GUTTER)
            with Image.open(sources[sample_id]) as source:
                # Only resize for display: keep the full field, colors, orientation and aspect.
                scale = min(IMAGE_WIDTH / source.width, image_height / source.height)
                display_size = (round(source.width * scale), round(source.height * scale))
                with source.convert("RGBA") as rgba:
                    with rgba.resize(display_size, Image.Resampling.LANCZOS) as display:
                        position = (
                            x + (IMAGE_WIDTH - display.width) // 2,
                            y + (image_height - display.height) // 2,
                        )
                        canvas.paste(display, position, display)
            draw.text(
                (x + IMAGE_WIDTH // 2, y + image_height + LABEL_HEIGHT // 2),
                sample_id, fill="black", font=font, anchor="mm",
            )
        if reflectance_scale is not None:
            paste_colorbar(canvas, reflectance_scale)
        # Existing artifacts are replaced only when explicitly requested.
        with output.open("wb" if overwrite else "xb") as handle:
            canvas.save(handle, format="PNG", dpi=(DPI, DPI))
    print(f"{output}: {size[0]} x {size[1]} pixels; {len(sample_ids)} samples", flush=True)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    reflectance, raw = paired_sources(args.reflectance_dir, args.raw_dir)
    scale = load_reflectance_scale(args.reflectance_dir)
    representatives = [sample_id for _, sample_id in REPRESENTATIVES]
    missing = set(representatives) - reflectance.keys()
    if missing:
        raise ValueError(f"Missing fixed representative samples: {sorted(missing)}")
    height = image_box_height((reflectance, raw))
    args.output_dir.mkdir(parents=True, exist_ok=args.overwrite)
    selections = [("representatives_1x7", representatives)]
    if not args.representatives_only:
        selections.insert(0, ("7x7", sorted(reflectance)))
    for suffix, sample_ids in selections:
        render_overview(
            reflectance, args.output_dir / f"reflectance_l2_norm_{suffix}.png",
            sample_ids=sample_ids, image_height=height, reflectance_scale=scale,
            overwrite=args.overwrite,
        )
        render_overview(
            raw, args.output_dir / f"raw_bmp_{suffix}.png",
            sample_ids=sample_ids, image_height=height, overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
