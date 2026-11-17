"""Generate the best-verified Plant_063-32 L-system and save its overlay.

The hardcoded parameters are the Phytomorph/Sinkhorn best verified LPFG-cost
parameters. The script reuses utils_nn.build_parameter_file, generate_plant,
and read_syn_plant, then saves the selected developmental-day frame with
branch points (green) and end points (blue).
"""

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXT_DIR = SCRIPT_DIR.parent
OUTPUT_ROOT = SCRIPT_DIR / "generated_lsystem_overlays"
PROJECT_WIDTH = 2454
PROJECT_HEIGHT = 2056
# Choose one or more LPFG frame days to overlay (for example: [12, 18, 27]).
TARGET_DAYS = [27, 18, 8]
# Phytomorph/Sinkhorn Rep_2: best verified LPFG-cost fit for Plant_063-32.
PARAMETERS = [
    9.892227172851562,
    3.1357641220092773,
    112.67990112304688,
    7.9647321701049805,
    125.08934020996094,
    4.946861267089844,
    0.5200015306472778,
    1.1997971534729004,
    86.02056884765625,
    174.11761474609375,
    0.7996162176132202,
    0.9200208187103271,
    0.4799894392490387,
]


def save_overlay(
    image,
    bp: np.ndarray,
    ep: np.ndarray,
    frame_day: int,
    coordinate_day: int,
    output_path: Path,
    *,
    show_coordinates: bool = True,
    marker_size: int = 55,
) -> None:
    """Save BP/EP markers, optionally labeling each with pixel coordinates."""
    fig, ax = plt.subplots(figsize=(12, 10))
    # LPFG renders at the same native pixel dimensions used by project, so no
    # image upscaling is needed and the projected pixel coordinates align 1:1.
    ax.imshow(image)

    for points, color, label in (
        (bp, "green", "BPs"),
        (ep, "blue", "EPs"),
    ):
        if points.size:
            ax.scatter(
                points[:, 1],
                points[:, 0],
                s=marker_size,
                c=color,
                marker="o",
                edgecolors="white",
                linewidths=0.8,
                label=f"{label} ({len(points)})",
            )

    if show_coordinates:
        text_offsets = [(8, 8), (8, -16), (-58, 8), (-58, -16), (8, 22), (-58, 22)]
        for label, points, color in (("BP", bp, "green"), ("EP", ep, "blue")):
            for index, (row, column) in enumerate(points):
                offset_x, offset_y = text_offsets[index % len(text_offsets)]
                ax.annotate(
                    f"{label}{index + 1} ({int(column)}, {int(row)})",
                    xy=(column, row),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8},
                )

    ax.set_title(
        f"Generated L-system — frame Day {frame_day:02d}; "
        f"project coordinates Day {coordinate_day:02d}"
    )
    ax.legend(loc="best")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate the plant, project its landmarks, and save the overlay PNG."""
    if not TARGET_DAYS:
        raise ValueError("Set TARGET_DAYS to at least one LPFG frame day.")
    if any(day < 3 for day in TARGET_DAYS):
        raise ValueError("TARGET_DAYS must contain frame days >= 3.")

    requested_days = list(dict.fromkeys(TARGET_DAYS))
    output_root = OUTPUT_ROOT.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    parameter_file = run_dir / "parameters.vset"
    generation_dir = run_dir / "generation"
    generation_dir.mkdir()

    # Import from the project's extension directory, regardless of launch cwd.
    sys.path.insert(0, str(EXT_DIR))
    import utils_nn

    utils_nn.build_parameter_file(str(parameter_file), PARAMETERS)
    print(f"Generating L-system from parameters in {parameter_file}...")
    started = time.perf_counter()

    original_cwd = Path.cwd()
    try:
        os.chdir(EXT_DIR)
        utils_nn.generate_plant(
            str(parameter_file),
            str(generation_dir),
            animation=True,
            width=PROJECT_WIDTH,
            height=PROJECT_HEIGHT,
        )
    finally:
        os.chdir(original_cwd)
    print(f"L-system generation completed in {time.perf_counter() - started:.1f}s.")

    output_file = generation_dir / "output.txt"
    if not output_file.is_file() or output_file.stat().st_size == 0:
        log_path = generation_dir / "lpfg_log.txt"
        raise RuntimeError(
            f"Generation did not produce project output: {output_file}. "
            f"See LPFG log: {log_path}"
        )

    print(f"Reading projected landmarks from {output_file}...")
    synthetic_bp, synthetic_ep = utils_nn.read_syn_plant(str(output_file))
    # LPFG names frame files maizeNNN_DayN.png; choose the last frame for the day.
    frame_pattern = re.compile(r"^maize(\d+)_Day(\d+)(?:\.\d+)?\.png$")
    frames_by_day: dict[int, list[tuple[int, Path]]] = {}
    for frame_path in generation_dir.glob("maize*_Day*.png"):
        match = frame_pattern.match(frame_path.name)
        if match:
            frame_number, frame_day = int(match.group(1)), int(match.group(2))
            frames_by_day.setdefault(frame_day, []).append((frame_number, frame_path))
    if not frames_by_day:
        available_frames = sorted(path.name for path in generation_dir.glob("maize*_Day*.png"))
        raise FileNotFoundError(
            f"No LPFG-rendered day images found in {generation_dir}. "
            f"Available frames: {available_frames[:8]}"
        )

    for day in requested_days:
        if day not in frames_by_day:
            raise FileNotFoundError(f"No LPFG-rendered image found for requested Day {day}.")
        stage_index = day - 2
        if stage_index >= len(synthetic_bp) or stage_index >= len(synthetic_ep):
            raise RuntimeError(
                f"No synthetic landmark data for frame Day {day} "
                f"(expected project state Day {day}); utils_nn.read_syn_plant "
                f"returned {len(synthetic_bp)} BP and {len(synthetic_ep)} EP stages."
            )

        # LPFG writes the frame and projected landmarks during the same
        # group-2 iteration, so both correspond to Day N.
        coordinate_day = day
        bp = np.asarray(synthetic_bp[stage_index], dtype=float).reshape(-1, 2)
        ep = np.asarray(synthetic_ep[stage_index], dtype=float).reshape(-1, 2)
        if not bp.size and not ep.size:
            raise RuntimeError(f"No synthetic landmarks were parsed for Day {day}.")

        _, frame_path = max(frames_by_day[day], key=lambda item: item[0])
        image = utils_nn.io.imread(str(frame_path))
        print(
            f"Day {day}: native LPFG resolution "
            f"{image.shape[1]}x{image.shape[0]} pixels."
        )

        overlay_path = run_dir / f"generated_lsystem_day_{day:02d}_coordinates.png"
        circles_path = run_dir / f"generated_lsystem_day_{day:02d}_circles_only.png"
        save_overlay(image, bp, ep, day, coordinate_day, overlay_path)
        save_overlay(
            image,
            bp,
            ep,
            day,
            coordinate_day,
            circles_path,
            show_coordinates=False,
            marker_size=180,
        )
        print(f"Generated frame: {frame_path}")
        print(f"Using project coordinates for Day {coordinate_day:02d} with frame Day {day:02d}.")
        print(f"Coordinate-labeled overlay saved to: {overlay_path}")
        print(f"Larger circles-only overlay saved to: {circles_path}")

    print(f"All generation files retained in: {run_dir}")


if __name__ == "__main__":
    main()
