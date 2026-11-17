"""Create two print-sized comparisons of real and Phytomorph growth for Plant 006-25."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


SCRIPT_DIR = Path(__file__).resolve().parent
EXTENSION_ROOT = SCRIPT_DIR.parent
PLANT = "Plant_006-25"
DAY_COUNT = 27
PAIRS_PER_ROW = 2
GRID_ROWS = 7
REAL_IMAGE_ROOT = EXTENSION_ROOT / "Real Plants" / PLANT / "SideView0"
PHYTO_IMAGE_ROOT = SCRIPT_DIR / "13 plant all optimized lsystems" / PLANT / "Phytomorph"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Plant_006-25 growth comparison"


def create_page(start_day: int, end_day: int) -> plt.Figure:
    """Create one letter-sized page of paired target and simulation days."""
    image_pairs = [
        (
            day,
            REAL_IMAGE_ROOT / f"Day_{day:03d}.png",
            PHYTO_IMAGE_ROOT / f"Day_{day:03d}.png",
        )
        for day in range(start_day, end_day + 1)
    ]
    for day, real_path, phyto_path in image_pairs:
        for image_path in (real_path, phyto_path):
            if not image_path.is_file() or image_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Missing or empty image for Day {day:03d}: {image_path}"
                )

    figure, axes = plt.subplots(
        GRID_ROWS,
        PAIRS_PER_ROW * 2,
        figsize=(8.5, 11),
        squeeze=False,
    )
    figure.suptitle(
        f"Target Plant 006-25 — Growth Comparison — Days {start_day:03d}–{end_day:03d}",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.97,
        "Each day-pair: real target plant image (left) and matching Phytomorph "
        "optimized L-system (right).",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )

    for pair_index, (day, real_path, phyto_path) in enumerate(image_pairs):
        row = pair_index // PAIRS_PER_ROW
        pair_column = pair_index % PAIRS_PER_ROW
        real_axis = axes[row, pair_column * 2]
        phyto_axis = axes[row, pair_column * 2 + 1]

        real_axis.imshow(mpimg.imread(real_path))
        real_axis.set_title(
            f"Day {day:03d} — Target",
            fontsize=8,
            fontweight="bold",
            pad=2,
        )
        real_axis.axis("off")

        phyto_axis.imshow(mpimg.imread(phyto_path))
        phyto_axis.set_title(
            f"Day {day:03d} — Phytomorph",
            fontsize=8,
            fontweight="bold",
            pad=2,
        )
        phyto_axis.axis("off")

    for axis_index in range(len(image_pairs), GRID_ROWS * PAIRS_PER_ROW):
        row = axis_index // PAIRS_PER_ROW
        pair_column = axis_index % PAIRS_PER_ROW
        axes[row, pair_column * 2].axis("off")
        axes[row, pair_column * 2 + 1].axis("off")

    figure.subplots_adjust(
        left=0.012,
        right=0.988,
        bottom=0.012,
        top=0.94,
        wspace=0.015,
        hspace=0.09,
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create two letter-sized target/Phytomorph comparison figures for Plant 006-25."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for two PNG pages and a two-page PDF (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = ((1, 14), (15, DAY_COUNT))
    pdf_path = output_dir / "Plant_006-25_growth_comparison_2_pages.pdf"

    with PdfPages(pdf_path) as pdf:
        for page_number, (start_day, end_day) in enumerate(pages, start=1):
            figure = create_page(start_day, end_day)
            image_path = (
                output_dir
                / f"Plant_006-25_growth_comparison_page_{page_number}_"
                f"days_{start_day:03d}-{end_day:03d}.png"
            )
            figure.savefig(image_path, dpi=180)
            pdf.savefig(figure)
            plt.close(figure)
            print(f"Saved Days {start_day:03d}-{end_day:03d} to {image_path}")
    print(f"Saved two-page PDF to {pdf_path}")


if __name__ == "__main__":
    main()
