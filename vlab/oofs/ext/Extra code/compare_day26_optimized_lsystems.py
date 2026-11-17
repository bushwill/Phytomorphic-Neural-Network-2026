"""Create one Day 26 real-versus-optimized L-system figure per maize plant."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from generate_daily_lpfg_frames import (
    Candidate,
    DEFAULT_OPTIMIZER_ROOT,
    DEFAULT_PLANTS_ROOT,
    DEFAULT_OUTPUT_ROOT as DAILY_IMAGES_ROOT,
    discover_plants,
    generate_frames,
    select_candidates,
)


SCRIPT_DIR = Path(__file__).resolve().parent
EXTENSION_ROOT = SCRIPT_DIR.parent
DEFAULT_TRAINING_ROOT = EXTENSION_ROOT / "Training Data"
DEFAULT_COMPARISON_ROOT = SCRIPT_DIR / "13 plant day 26 comparisons"
DAY = 26
REAL_IMAGE_VIEW = "SideView0"


def _finite_r2(value: object, source: Path) -> float:
    try:
        r2 = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid test R² value in {source}: {value!r}") from exc
    if not math.isfinite(r2):
        raise ValueError(f"Non-finite test R² value in {source}: {value!r}")
    return r2


def read_candidate_r2(candidate: Candidate, training_root: Path) -> float:
    """Read test R² for the model replicate that produced this optimizer result."""
    source = candidate.source
    if len(source.parents) < 6:
        raise ValueError(f"Unexpected optimizer result path: {source}")

    replicate = source.parent.name
    family = source.parents[1].name
    dataset_name = source.parents[3].name
    experiment_tag = source.parents[5].name
    run_dir = training_root / f"{experiment_tag}_{dataset_name}" / family / replicate
    metrics_path = run_dir / "metrics.json"

    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict) or "test_r2" not in metrics:
            raise ValueError(f"Missing metrics.test_r2 in {metrics_path}")
        return _finite_r2(metrics["test_r2"], metrics_path)

    summary_path = run_dir.parent.parent / "summary_results.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"No metrics.json or summary_results.csv for selected candidate {source}; "
            f"expected {metrics_path} or {summary_path}."
        )

    with summary_path.open("r", newline="", encoding="utf-8-sig") as source_file:
        rows = csv.DictReader(source_file)
        matches = [
            row
            for row in rows
            if row.get("model_name", "").strip() == family
            and row.get("replicate", "").strip() == replicate.removeprefix("Rep_")
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one test-R² row for {family}/{replicate} in {summary_path}, "
            f"found {len(matches)}."
        )
    return _finite_r2(matches[0].get("test_r2"), summary_path)


def create_figure(
    plant: str,
    real_image_path: Path,
    phytomorph_image_path: Path,
    mlp_image_path: Path,
    phytomorph: Candidate,
    phytomorph_r2: float,
    mlp: Candidate,
    mlp_r2: float,
    output_path: Path,
) -> None:
    """Save a three-panel figure with the selected runs' matching metrics."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = (
        (axes[0], real_image_path, f"Target Plant {plant.removeprefix('Plant_')}"),
        (axes[1], phytomorph_image_path, "Phytomorph Optimized L-system"),
        (axes[2], mlp_image_path, "MLP Optimized L-system"),
    )
    for axis, image_path, title in panels:
        axis.imshow(mpimg.imread(image_path))
        axis.set_title(title, fontsize=14, fontweight="bold", pad=6)
        axis.axis("off")

    figure.suptitle(
        f"Target Plant {plant.removeprefix('Plant_')} — Day {DAY}",
        fontsize=19,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.945,
        f"Phytomorphic R² = {phytomorph_r2:.2f}, cost {phytomorph.cost:,.0f}; "
        f"MLP R² = {mlp_r2:.2f}, cost {mlp.cost:,.0f}",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.86, wspace=0.015)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def create_comparison_figures(
    plants: list[str],
    candidates: dict[tuple[str, str], Candidate],
    daily_root: Path,
    training_root: Path,
    output_root: Path,
) -> None:
    """Create one Day 26 comparison for each requested plant."""
    for plant in plants:
        real_image = DEFAULT_PLANTS_ROOT / plant / REAL_IMAGE_VIEW / f"Day_{DAY:03d}.png"
        phytomorph_image = daily_root / plant / "Phytomorph" / f"Day_{DAY:03d}.png"
        mlp_image = daily_root / plant / "MLP" / f"Day_{DAY:03d}.png"
        for image_path in (real_image, phytomorph_image, mlp_image):
            if not image_path.is_file() or image_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Required Day {DAY} image is missing or empty: {image_path}"
                )

        phytomorph = candidates[(plant, "sinkhorn")]
        mlp = candidates[(plant, "baseline")]
        phytomorph_r2 = read_candidate_r2(phytomorph, training_root)
        mlp_r2 = read_candidate_r2(mlp, training_root)
        output_path = output_root / f"{plant}_Day_{DAY:03d}_comparison.png"
        create_figure(
            plant,
            real_image,
            phytomorph_image,
            mlp_image,
            phytomorph,
            phytomorph_r2,
            mlp,
            mlp_r2,
            output_path,
        )
        print(
            f"Saved {output_path}: "
            f"Phytomorph R²={phytomorph_r2:.4f}, cost={phytomorph.cost:,.0f}; "
            f"MLP R²={mlp_r2:.4f}, cost={mlp.cost:,.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Day 26 real/Phytomorph/MLP comparison figures."
    )
    parser.add_argument(
        "--plants",
        nargs="+",
        help="Optional plant directory names to include (for example Plant_006-25).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_COMPARISON_ROOT,
        help=f"Figure output directory (default: {DEFAULT_COMPARISON_ROOT})",
    )
    parser.add_argument(
        "--daily-images-root",
        type=Path,
        default=DAILY_IMAGES_ROOT,
        help=f"Previously generated optimized L-system images (default: {DAILY_IMAGES_ROOT})",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
        help=f"Training metrics directory (default: {DEFAULT_TRAINING_ROOT})",
    )
    parser.add_argument(
        "--optimizer-results-root",
        type=Path,
        default=DEFAULT_OPTIMIZER_ROOT,
        help=f"Optimizer results directory (default: {DEFAULT_OPTIMIZER_ROOT})",
    )
    parser.add_argument(
        "--render-best-candidates",
        action="store_true",
        help="Render selected candidates into a temporary directory instead of reusing saved images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available_plants = discover_plants(DEFAULT_PLANTS_ROOT)
    plants = args.plants or available_plants
    unknown_plants = sorted(set(plants) - set(available_plants))
    if unknown_plants:
        raise ValueError(f"Unknown plant directory name(s): {', '.join(unknown_plants)}")
    if len(set(plants)) != len(plants):
        raise ValueError("Plant list contains duplicates.")

    optimizer_root = args.optimizer_results_root.resolve()
    training_root = args.training_root.resolve()
    output_root = args.output_root.resolve()
    candidates = select_candidates(optimizer_root, plants)

    if args.render_best_candidates:
        with tempfile.TemporaryDirectory(prefix="day26-best-candidates-") as temporary_name:
            daily_root = Path(temporary_name)
            for plant in plants:
                for family in ("sinkhorn", "baseline"):
                    generate_frames(
                        plant=plant,
                        family=family,
                        candidate=candidates[(plant, family)],
                        output_root=daily_root,
                    )
            create_comparison_figures(
                plants, candidates, daily_root, training_root, output_root
            )
    else:
        create_comparison_figures(
            plants,
            candidates,
            args.daily_images_root.resolve(),
            training_root,
            output_root,
        )
    print(f"Created {len(plants)} Day {DAY} comparison figures in {output_root}.")


if __name__ == "__main__":
    main()
