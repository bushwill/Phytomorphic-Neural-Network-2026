"""Save BP/EP landmarks over the topo, original-photo, and skeleton images.

Edit TARGET_PLANT and TARGET_DAY below, then run this script. Landmark extraction
uses the same ``utils_nn.read_real_plant_day`` routine as the main pipeline.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Hardcoded selection: change these two values to inspect another plant/day.
TARGET_PLANT = "Plant_063-32"
TARGET_DAY = 27
SOURCE_VIEW = "SideView0"

SCRIPT_DIR = Path(__file__).resolve().parent
EXT_DIR = SCRIPT_DIR.parent
REAL_PLANTS_DIR = EXT_DIR / "Real Plants"
OUTPUT_DIR = SCRIPT_DIR / "landmark_overlays"

# Make the existing landmark extraction code importable when run from anywhere.
sys.path.insert(0, str(EXT_DIR))


def save_landmark_overlay(
    image,
    title: str,
    output_path: Path,
    bp,
    ep,
    *,
    show_coordinates: bool = True,
    marker_size: int = 55,
) -> None:
    """Save BP/EP circles, optionally labeling each point with pixel coordinates."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image)
    if bp.size:
        ax.scatter(
            bp[:, 1], bp[:, 0],
            s=marker_size, c="green", marker="o", edgecolors="white",
            linewidths=0.8, label=f"BPs ({len(bp)})",
        )
    if ep.size:
        ax.scatter(
            ep[:, 1], ep[:, 0],
            s=marker_size, c="blue", marker="o", edgecolors="white",
            linewidths=0.8, label=f"EPs ({len(ep)})",
        )

    if show_coordinates:
        text_offsets = [(8, 8), (8, -16), (-52, 8), (-52, -16), (8, 22), (-52, 22)]
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

    ax.set_title(title)
    ax.legend(loc="best")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load the selected real image, extract BPs/EPs, and save the overlay."""
    if TARGET_DAY < 0:
        raise ValueError("TARGET_DAY must be a non-negative day number")

    day_image_name = f"Day_{TARGET_DAY:03d}.png"
    topology_path = REAL_PLANTS_DIR / TARGET_PLANT / "topo" / day_image_name
    photo_path = REAL_PLANTS_DIR / TARGET_PLANT / SOURCE_VIEW / day_image_name
    skeleton_path = REAL_PLANTS_DIR / TARGET_PLANT / "Skeleton" / day_image_name
    for image_path in (topology_path, photo_path, skeleton_path):
        if not image_path.is_file():
            raise FileNotFoundError(f"Real plant image not found: {image_path}")

    print("Loading the shared plant-landmark pipeline...")
    stage_start = time.perf_counter()
    import utils_nn

    print(f"Pipeline import completed in {time.perf_counter() - stage_start:.1f}s.")

    print(f"Extracting landmarks from {topology_path}...")
    stage_start = time.perf_counter()
    topology_image, branch_points, end_points = utils_nn.read_real_plant_day(
        TARGET_PLANT,
        TARGET_DAY,
        image_root=REAL_PLANTS_DIR,
    )
    photo = utils_nn.io.imread(str(photo_path))
    skeleton_image = utils_nn.io.imread(str(skeleton_path))
    print(
        f"Landmark extraction completed in {time.perf_counter() - stage_start:.1f}s "
        f"({len(branch_points)} BPs, {len(end_points)} EPs)."
    )

    if not branch_points and not end_points:
        raise RuntimeError(
            f"No landmarks were extracted from {TARGET_PLANT} Day {TARGET_DAY:03d}"
        )

    reference_shape = topology_image.shape[:2]
    for image_label, image in (
        ("original photo", photo),
        ("skeleton", skeleton_image),
    ):
        if image.shape[:2] != reference_shape:
            raise ValueError(
                f"Cannot overlay topology coordinates: {image_label} and topology "
                f"dimensions differ ({image.shape[:2]} vs {reference_shape})."
            )

    # The pipeline coordinates are [row, column]; plot x=column, y=row.
    bp = np.asarray(branch_points, dtype=float).reshape(-1, 2)
    ep = np.asarray(end_points, dtype=float).reshape(-1, 2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{TARGET_PLANT}_Day_{TARGET_DAY:03d}"
    topology_output_path = OUTPUT_DIR / f"{output_prefix}_topology_source.png"
    photo_coords_output_path = OUTPUT_DIR / f"{output_prefix}_original_plant_coordinates.png"
    photo_circles_output_path = OUTPUT_DIR / f"{output_prefix}_original_plant_circles.png"
    skeleton_output_path = OUTPUT_DIR / f"{output_prefix}_skeletonized.png"
    title_prefix = f"{TARGET_PLANT} — Day {TARGET_DAY:03d}"

    save_landmark_overlay(
        topology_image,
        f"{title_prefix} — topology source used for coordinate extraction",
        topology_output_path,
        bp,
        ep,
    )
    save_landmark_overlay(
        photo,
        f"{title_prefix} — original {SOURCE_VIEW} plant photo with coordinates",
        photo_coords_output_path,
        bp,
        ep,
    )
    save_landmark_overlay(
        photo,
        f"{title_prefix} — original {SOURCE_VIEW} plant photo, circles only",
        photo_circles_output_path,
        bp,
        ep,
        show_coordinates=False,
        marker_size=180,
    )
    save_landmark_overlay(
        skeleton_image,
        f"{title_prefix} — skeletonized image",
        skeleton_output_path,
        bp,
        ep,
    )
    print(f"Topology-source overlay saved to: {topology_output_path}")
    print(f"Original-photo overlay with coordinates saved to: {photo_coords_output_path}")
    print(f"Original-photo circles-only overlay saved to: {photo_circles_output_path}")
    print(f"Skeleton overlay saved to: {skeleton_output_path}")


if __name__ == "__main__":
    main()
