"""Generate image-only daily LPFG frames for the 13 real plants.

The best verified candidate is selected across all optimizer result CSVs for
each plant and optimizer family. LPFG's intermediate files are kept in a
per-generation temporary directory and only the selected daily PNG frames
are copied to the output tree.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
EXTENSION_ROOT = SCRIPT_DIR.parent
DEFAULT_OPTIMIZER_ROOT = EXTENSION_ROOT / "Optimizer Data"
DEFAULT_PLANTS_ROOT = EXTENSION_ROOT / "Real Plants"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "13 plant all optimized lsystems"
PARAMETER_COLUMNS = tuple(f"param_{index}" for index in range(13))
FAMILY_TO_METHOD = {"sinkhorn": "Phytomorph", "baseline": "MLP"}
FRAME_PATTERN = re.compile(r"^maize(?P<frame>\d+)_Day(?P<day>\d+)(?:\.\d+)?\.png$")
EXPECTED_PLANT_COUNT = 13


@dataclass(frozen=True)
class Candidate:
    """A verified optimizer candidate and its source location."""

    cost: float
    parameters: tuple[float, ...]
    source: Path
    row_number: int


def method_for_family(family: str) -> str:
    """Return the display folder name for a supported optimizer family."""
    try:
        return FAMILY_TO_METHOD[family.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported optimizer family {family!r}; expected one of "
            f"{', '.join(sorted(FAMILY_TO_METHOD))}."
        ) from exc


def select_best_candidate(
    rows: Iterable[tuple[Mapping[str, str], Path, int]],
    *,
    plant: str,
    family: str,
) -> Candidate:
    """Choose the lowest finite-cost row, then validate its 13 parameters."""
    finite_cost_rows: list[tuple[float, Mapping[str, str], Path, int]] = []

    for row, source, row_number in rows:
        try:
            cost = float(row.get("verified_sim_cost", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(cost):
            continue
        finite_cost_rows.append((cost, row, source, row_number))

    if not finite_cost_rows:
        raise ValueError(
            f"No valid candidate for plant {plant}, family {family!r}: "
            "no rows have a finite verified_sim_cost."
        )

    cost, row, source, row_number = min(
        finite_cost_rows,
        key=lambda item: (item[0], str(item[2]), item[3]),
    )
    try:
        parameters = tuple(float(row[column]) for column in PARAMETER_COLUMNS)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Lowest-cost candidate for plant {plant}, family {family!r} "
            f"({source}, row {row_number}, cost {cost:g}) does not contain "
            "13 valid parameter values."
        ) from exc
    if len(parameters) != 13 or not all(math.isfinite(value) for value in parameters):
        raise ValueError(
            f"Lowest-cost candidate for plant {plant}, family {family!r} "
            f"({source}, row {row_number}, cost {cost:g}) does not contain "
            "13 finite parameter values."
        )
    return Candidate(cost, parameters, source, row_number)


def _read_csv_candidates(csv_paths: Sequence[Path]) -> Iterable[tuple[Mapping[str, str], Path, int]]:
    for csv_path in sorted(csv_paths):
        with csv_path.open("r", newline="", encoding="utf-8-sig") as source_file:
            reader = csv.DictReader(source_file)
            for row_number, row in enumerate(reader, start=2):
                yield row, csv_path, row_number


def select_candidates(
    optimizer_root: Path,
    plant_names: Sequence[str],
) -> dict[tuple[str, str], Candidate]:
    """Select one best valid candidate for each plant and method family."""
    if not optimizer_root.is_dir():
        raise FileNotFoundError(f"Optimizer results root does not exist: {optimizer_root}")

    plant_set = set(plant_names)
    csvs_by_key: dict[tuple[str, str], list[Path]] = {}
    for csv_path in optimizer_root.rglob("opt_result.csv"):
        parts = {part.lower(): part for part in csv_path.parts}
        plant = next((name for name in plant_names if name.lower() in parts), None)
        family = next(
            (name for name in FAMILY_TO_METHOD if name in parts),
            None,
        )
        if plant in plant_set and family is not None:
            csvs_by_key.setdefault((plant, family), []).append(csv_path)

    selected: dict[tuple[str, str], Candidate] = {}
    for plant in plant_names:
        for family in FAMILY_TO_METHOD:
            paths = csvs_by_key.get((plant, family), [])
            if not paths:
                raise ValueError(
                    f"No opt_result.csv found for plant {plant}, family {family!r} "
                    f"under {optimizer_root}."
                )
            selected[(plant, family)] = select_best_candidate(
                _read_csv_candidates(paths),
                plant=plant,
                family=family,
            )
    return selected


def discover_plants(plants_root: Path) -> list[str]:
    """Find and validate the expected set of real plant directories."""
    if not plants_root.is_dir():
        raise FileNotFoundError(f"Real Plants directory does not exist: {plants_root}")
    plants = sorted(path.name for path in plants_root.iterdir() if path.is_dir())
    if len(plants) != EXPECTED_PLANT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PLANT_COUNT} real plant directories in {plants_root}, "
            f"found {len(plants)}."
        )
    return plants


def _find_daily_frames(generation_dir: Path) -> dict[int, Path]:
    """Return the highest-index rendered frame for each developmental day."""
    by_day: dict[int, tuple[int, Path]] = {}
    for frame_path in generation_dir.glob("maize*_Day*.png"):
        match = FRAME_PATTERN.fullmatch(frame_path.name)
        if not match or not frame_path.is_file() or frame_path.stat().st_size == 0:
            continue
        frame_index, day = int(match.group("frame")), int(match.group("day"))
        current = by_day.get(day)
        if current is None or frame_index > current[0]:
            by_day[day] = (frame_index, frame_path)
    return {day: item[1] for day, item in by_day.items()}


def _replace_output_frames(destination: Path, frames: Mapping[int, Path]) -> None:
    """Replace a plant/method output folder with daily PNG frames only."""
    if destination.is_symlink():
        raise ValueError(f"Refusing to replace symlink output directory: {destination}")
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(f"Frame output path is not a directory: {destination}")
        for child in destination.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        destination.mkdir(parents=True)

    for day, source in sorted(frames.items()):
        target = destination / f"Day_{day:03d}.png"
        shutil.copy2(source, target)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"Failed to copy rendered frame to {target}.")


def generate_frames(
    *,
    plant: str,
    family: str,
    candidate: Candidate,
    output_root: Path,
) -> list[Path]:
    """Generate one animated LPFG simulation and copy only its daily PNGs."""
    method = method_for_family(family)
    output_root.mkdir(parents=True, exist_ok=True)

    extension_root = EXTENSION_ROOT.resolve()
    if str(extension_root) not in sys.path:
        sys.path.insert(0, str(extension_root))
    import utils_nn

    with tempfile.TemporaryDirectory(
        prefix=f".lpfg-{plant}-{method.lower()}-",
        dir=output_root,
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        parameter_file = temporary_dir / "parameters.vset"
        generation_dir = temporary_dir / "generation"
        generation_dir.mkdir()
        utils_nn.build_parameter_file(str(parameter_file), candidate.parameters)

        previous_cwd = Path.cwd()
        try:
            os.chdir(extension_root)
            utils_nn.generate_plant(
                str(parameter_file),
                str(generation_dir),
                animation=True,
                width=2454,
                height=2056,
            )
        except Exception as exc:
            raise RuntimeError(
                f"LPFG generation failed for plant {plant}, method {method}, "
                f"candidate {candidate.source} row {candidate.row_number}: {exc}"
            ) from exc
        finally:
            os.chdir(previous_cwd)

        project_output = generation_dir / "output.txt"
        if not project_output.is_file() or project_output.stat().st_size == 0:
            raise RuntimeError(
                f"Incomplete LPFG generation for plant {plant}, method {method}: "
                f"missing or empty {project_output}."
            )
        frames = _find_daily_frames(generation_dir)
        if not frames:
            raise RuntimeError(
                f"Incomplete LPFG generation for plant {plant}, method {method}: "
                f"no nonempty maizeNNN_DayN.png frames in {generation_dir}."
            )

        destination = output_root / plant / method
        _replace_output_frames(destination, frames)
        return [destination / f"Day_{day:03d}.png" for day in sorted(frames)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate image-only daily LPFG frames for all 13 real plants."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Daily image destination (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--optimizer-results-root",
        type=Path,
        default=DEFAULT_OPTIMIZER_ROOT,
        help=f"Optimizer Data directory (default: {DEFAULT_OPTIMIZER_ROOT})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    plants_root = DEFAULT_PLANTS_ROOT.resolve()
    optimizer_root = args.optimizer_results_root.resolve()
    plants = discover_plants(plants_root)
    candidates = select_candidates(optimizer_root, plants)

    print(
        f"Generating daily LPFG frames for {len(plants)} plants "
        f"({len(plants) * len(FAMILY_TO_METHOD)} simulations)."
    )
    for plant in plants:
        for family, method in (("sinkhorn", "Phytomorph"), ("baseline", "MLP")):
            candidate = candidates[(plant, family)]
            written = generate_frames(
                plant=plant,
                family=family,
                candidate=candidate,
                output_root=output_root,
            )
            print(
                f"{plant}/{method}: selected cost {candidate.cost:g} from "
                f"{candidate.source} row {candidate.row_number}; "
                f"copied {len(written)} daily frames."
            )


if __name__ == "__main__":
    main()
