"""Generate six Day 26 L-systems from a surrogate training dataset.

Animated landmarks use LPFG's exported view volume so the overlay and rendered
image share the same camera as well as the same developmental day.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXTENSION_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_ROOT = (
    EXTENSION_ROOT / "Datasets" / "Plant_063-32-50000_10000_25000"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "synthetic_training_dataset_examples"
    / "six_train_samples_day26_raw_and_annotated.png"
)
SAMPLE_IDS = (0, 1, 2, 3, 4, 5)
DAY = 26
FRAME_PATTERN = re.compile(r"^maize(?P<frame>\d+)_Day(?P<day>\d+)(?:\.\d+)?\.png$")
PARAMETER_COLUMNS = tuple(f"param_{index}" for index in range(13))
LABEL_OFFSETS = ((7, 7), (7, -13), (-25, 7), (-25, -13), (7, 20), (-25, 20))


@dataclass(frozen=True)
class TrainingSample:
    sample_id: int
    parameters: tuple[float, ...]


def load_training_samples(
    dataset_root: Path,
    sample_ids: tuple[int, ...],
) -> list[TrainingSample]:
    csv_path = dataset_root / "Train.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    requested_ids = set(sample_ids)
    found: dict[int, TrainingSample] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        required_columns = {"id", *PARAMETER_COLUMNS}
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"Training CSV {csv_path} is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        for row in reader:
            try:
                sample_id = int(row["id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sample id in {csv_path}: {row.get('id')!r}") from exc
            if sample_id not in requested_ids:
                continue
            if sample_id in found:
                raise ValueError(f"Duplicate sample id {sample_id} in {csv_path}")
            try:
                parameters = tuple(float(row[column]) for column in PARAMETER_COLUMNS)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid parameters for sample {sample_id} in {csv_path}"
                ) from exc
            if len(parameters) != len(PARAMETER_COLUMNS) or not all(
                math.isfinite(value) for value in parameters
            ):
                raise ValueError(f"Non-finite or incomplete parameters for sample {sample_id}")

            structure_path = (
                dataset_root / "Train" / "structures" / f"structure_{sample_id}.pt"
            )
            if not structure_path.is_file():
                raise FileNotFoundError(
                    f"Training structure for sample {sample_id} is missing: "
                    f"{structure_path}"
                )
            found[sample_id] = TrainingSample(sample_id, parameters)

    missing_ids = requested_ids - found.keys()
    if missing_ids:
        raise ValueError(
            f"Sample ids not found in {csv_path}: "
            f"{', '.join(map(str, sorted(missing_ids)))}"
        )
    return [found[sample_id] for sample_id in sample_ids]


def load_plant_name(dataset_root: Path) -> str:
    description_path = dataset_root / "description.txt"
    if not description_path.is_file():
        raise FileNotFoundError(f"Dataset description not found: {description_path}")

    with description_path.open("r", encoding="utf-8") as description_file:
        for line in description_file:
            field, separator, value = line.partition(":")
            if separator and field.strip() == "Plant" and value.strip():
                return value.strip()
    raise ValueError(f"Plant name not found in dataset description: {description_path}")


def find_day_frame(generation_dir: Path, day: int) -> Path:
    frames: list[tuple[int, Path]] = []
    for path in generation_dir.glob("maize*_Day*.png"):
        match = FRAME_PATTERN.fullmatch(path.name)
        if match and int(match.group("day")) == day and path.stat().st_size:
            frames.append((int(match.group("frame")), path))
    if not frames:
        raise FileNotFoundError(
            f"No generated L-system frame for Day {day} in {generation_dir}"
        )
    return max(frames, key=lambda item: item[0])[1]


def generate_sample(
    sample: TrainingSample,
    temporary_root: Path,
    utils_nn: ModuleType,
) -> tuple[Path, np.ndarray, np.ndarray]:
    sample_dir = temporary_root / f"sample_{sample.sample_id:05d}"
    generation_dir = sample_dir / "generation"
    generation_dir.mkdir(parents=True)
    parameter_file = sample_dir / "parameters.vset"
    utils_nn.build_parameter_file(str(parameter_file), sample.parameters)

    previous_cwd = Path.cwd()
    try:
        os.chdir(EXTENSION_ROOT)
        utils_nn.generate_plant(
            str(parameter_file),
            str(generation_dir),
            animation=True,
            width=2454,
            height=2056,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate Day {DAY} L-system for training sample "
            f"{sample.sample_id}: {exc}"
        ) from exc
    finally:
        os.chdir(previous_cwd)

    output_file = generation_dir / "output.txt"
    if not output_file.is_file() or not output_file.stat().st_size:
        raise RuntimeError(
            f"Generation produced no projected landmarks for sample {sample.sample_id}: "
            f"{output_file}"
        )
    branch_stages, end_stages = utils_nn.read_syn_plant(str(output_file))
    # read_syn_plant() stores Day 2 at index 0 and Day N at index N - 2.
    # LPFG writes the image and projected landmarks during the same group-2
    # iteration, so select the coordinates for this image's exact day.
    stage_index = DAY - 2
    if stage_index >= len(branch_stages) or stage_index >= len(end_stages):
        raise RuntimeError(
            f"No Day {DAY} landmark stage for training sample {sample.sample_id}"
        )
    branch_points = np.asarray(branch_stages[stage_index], dtype=float).reshape(-1, 2)
    end_points = np.asarray(end_stages[stage_index], dtype=float).reshape(-1, 2)
    if not branch_points.size and not end_points.size:
        raise RuntimeError(
            f"No Day {DAY} landmarks were parsed for training sample {sample.sample_id}"
        )
    return find_day_frame(generation_dir, DAY), branch_points, end_points


def draw_annotations(
    axis: plt.Axes,
    branch_points: np.ndarray,
    end_points: np.ndarray,
) -> None:
    for points, color, label in (
        (branch_points, "green", "BP"),
        (end_points, "blue", "EP"),
    ):
        if not points.size:
            continue
        axis.scatter(
            points[:, 1],
            points[:, 0],
            s=55,
            c=color,
            marker="o",
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        for index, (row, column) in enumerate(points):
            offset_x, offset_y = LABEL_OFFSETS[index % len(LABEL_OFFSETS)]
            axis.annotate(
                f"{label}{index + 1}",
                xy=(column, row),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=7,
                color=color,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.85},
            )


def create_figure(
    dataset_root: Path,
    sample_ids: tuple[int, ...],
    output_path: Path,
) -> None:
    samples = load_training_samples(dataset_root, sample_ids)
    plant_name = load_plant_name(dataset_root)
    if str(EXTENSION_ROOT) not in sys.path:
        sys.path.insert(0, str(EXTENSION_ROOT))
    import utils_nn

    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_per_figure = 3
    with tempfile.TemporaryDirectory(
        prefix="synthetic-training-samples-",
        dir=output_path.parent,
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        for figure_index in range(2):
            figure_samples = samples[
                figure_index * samples_per_figure :
                (figure_index + 1) * samples_per_figure
            ]
            figure = plt.figure(figsize=(14, 15))
            try:
                grid = figure.add_gridspec(
                    samples_per_figure,
                    3,
                    width_ratios=(0.12, 0.44, 0.44),
                    hspace=0.12,
                    wspace=0.02,
                )
                figure.suptitle(
                    "Six Synthetic Training-Set Samples",
                    fontsize=19,
                    fontweight="bold",
                    y=0.99,
                )
                figure.text(
                    0.5,
                    0.955,
                    f"Dataset: {dataset_root.name} | Plant: {plant_name} | "
                    f"Generated at Day {DAY}",
                    ha="center",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )

                label_axes: list[plt.Axes] = []
                raw_axes: list[plt.Axes] = []
                overlay_axes: list[plt.Axes] = []
                for row_index in range(samples_per_figure):
                    label_axes.append(figure.add_subplot(grid[row_index, 0]))
                    raw_axes.append(figure.add_subplot(grid[row_index, 1]))
                    overlay_axes.append(figure.add_subplot(grid[row_index, 2]))

                raw_axes[0].set_title(
                    f"Raw Generated L-system — Day {DAY}",
                    fontsize=13,
                    fontweight="bold",
                    pad=6,
                )
                overlay_axes[0].set_title(
                    f"Same Day {DAY} Image with BP/EP Overlay",
                    fontsize=13,
                    fontweight="bold",
                    pad=6,
                )

                for row_index, sample in enumerate(figure_samples):
                    frame_path, branch_points, end_points = generate_sample(
                        sample,
                        temporary_root,
                        utils_nn,
                    )
                    image = mpimg.imread(frame_path)

                    label_axis = label_axes[row_index]
                    label_axis.text(
                        0.98,
                        0.5,
                        f"Sample\n{sample.sample_id}",
                        ha="right",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        transform=label_axis.transAxes,
                    )
                    label_axis.axis("off")

                    raw_axis = raw_axes[row_index]
                    raw_axis.imshow(image)
                    raw_axis.axis("off")

                    overlay_axis = overlay_axes[row_index]
                    overlay_axis.imshow(image)
                    draw_annotations(overlay_axis, branch_points, end_points)
                    overlay_axis.axis("off")

                    print(
                        f"Generated sample {sample.sample_id} Day {DAY}: "
                        f"{len(branch_points)} branch points, "
                        f"{len(end_points)} endpoints."
                    )

                figure.subplots_adjust(
                    left=0.02,
                    right=0.99,
                    bottom=0.01,
                    top=0.91,
                )
                figure_path = output_path.with_name(
                    f"{output_path.stem}_part{figure_index + 1}{output_path.suffix}"
                )
                figure.savefig(figure_path, dpi=180)
                figure.savefig(figure_path.with_suffix(".pdf"))
            finally:
                plt.close(figure)

            print(f"Saved figure to {figure_path}")
            print(f"Saved print-ready PDF to {figure_path.with_suffix('.pdf')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate six Day 26 L-systems from a synthetic training dataset and "
            "compare raw images with their BP/EP annotations."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Synthetic dataset directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--sample-ids",
        type=int,
        nargs=6,
        default=SAMPLE_IDS,
        help="Six Train.csv sample IDs (default: 0 1 2 3 4 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = tuple(args.sample_ids)
    if len(set(sample_ids)) != 6:
        raise ValueError("Sample IDs must be unique.")
    create_figure(args.dataset_root.resolve(), sample_ids, args.output.resolve())


if __name__ == "__main__":
    main()
