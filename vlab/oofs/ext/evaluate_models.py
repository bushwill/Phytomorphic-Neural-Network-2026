#!/usr/bin/env python3
# This script was written by ChatGPT-5 and reviewed by William Bushell

"""
Final all-plant experiment analyses and figures.

This script is specialized for the Stage-5 all-plant comparison between:
- baseline (MLP)
- sinkhorn_no_aggregator (selected phytomorphic surrogate)

It can be used in two ways:
1) CLI mode: pass explicit arguments.
2) Top-config mode: set the USER_* constants below and run with no args,
   or pass --use-top-config.

Required outputs:
- fig_final_all_plant_verified_cost.pdf/.png
- fig_r2_vs_verified_cost.pdf/.png
- table_r2_verified_correlation.csv
- table_accuracy_utility_analysis.csv
- table_accuracy_utility_analysis.txt
- fig_boundary_abs_zscore_heatmap.pdf/.png
- boundary_abs_zscore_heatmap_data.csv
- boundary_metrics_by_candidate.csv
- boundary_metrics_by_model.csv
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# Keep docker-created files host-editable.
os.umask(0)


# -----------------------------
# Top-of-file user configuration
# -----------------------------
USER_USE_TOP_CONFIG = True
USER_EXPERIMENT_DIR = "Optimizer Data/Experiment_061526_224301"
USER_DATASETS_DIR = "Datasets"
USER_TRAINING_ROOT_DIR = "Training Data"
USER_RESULTS_CSV = ""
USER_NORM_STATS_CSV = ""
USER_OUTPUT_DIR = ""
USER_MODELS = ["MLP", "Phytomorph"]
USER_MODEL_ALIASES = {
    # The labels are used in figures and CSV outputs.
    "baseline": "MLP",
    "sinkhorn": "Phytomorph",
    "sinkhorn_no_aggregator": "Phytomorph",
}


# Parameter order and names expected by optimizer output.
PARAM_COLS = [
    "opt_max_phytomers",
    "opt_plastochron",
    "opt_plant_roll_angle",
    "opt_plant_down_angle",
    "opt_branch_angle",
    "opt_leaf_len",
    "opt_exp_leaf_wid",
    "opt_leaf_wid",
    "opt_leaf_bend_scale",
    "opt_leaf_twist_scale",
    "opt_node_len",
    "opt_int_wid",
    "opt_exp_int_rad",
]

REQUIRED_RESULT_COLS = [
    "target_plant",
    "model",
    "replicate",
    "test_r2",
    "verified_vlab_cost",
    "surrogate_predicted_cost",
    "selected_restart",
] + PARAM_COLS

REQUIRED_NORM_COLS = ["target_plant", "parameter", "train_mean", "train_std"]


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve(path_str: str, base: Optional[Path] = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base or _script_dir()) / p


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _normalize_model_name(name: str, aliases: Dict[str, str]) -> str:
    key = (name or "").strip()
    if key in aliases:
        return aliases[key]
    return key


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _read_selected_restart(rep_dir: Path, surrogate_cost: float, verified_cost: float) -> float:
    restart_csv = rep_dir / "restart_results.csv"
    if not restart_csv.exists():
        return float("nan")

    try:
        rdf = pd.read_csv(restart_csv)
    except Exception:
        return float("nan")

    if "restart" not in rdf.columns:
        return float("nan")

    if "verified_sim_cost" in rdf.columns and np.isfinite(verified_cost):
        cand = rdf.copy()
        cand["_dist"] = (cand["verified_sim_cost"].astype(float) - verified_cost).abs()
        cand = cand.sort_values("_dist")
        if len(cand) > 0 and np.isfinite(cand.iloc[0]["_dist"]):
            return _safe_float(cand.iloc[0]["restart"])

    if "surrogate_pred_cost" in rdf.columns and np.isfinite(surrogate_cost):
        cand = rdf.copy()
        cand["_dist"] = (cand["surrogate_pred_cost"].astype(float) - surrogate_cost).abs()
        cand = cand.sort_values("_dist")
        if len(cand) > 0 and np.isfinite(cand.iloc[0]["_dist"]):
            return _safe_float(cand.iloc[0]["restart"])

    return float("nan")


def _read_test_r2(training_root: Path, experiment_tag: str, dataset_name: str, model: str, replicate: int) -> float:
    run_dir = training_root / f"{experiment_tag}_{dataset_name}" / model / f"Rep_{int(replicate)}"
    metrics_json = run_dir / "metrics.json"
    if metrics_json.exists():
        try:
            payload = json.loads(metrics_json.read_text())
            return _safe_float(payload.get("metrics", {}).get("test_r2", np.nan))
        except Exception:
            pass

    # Legacy fallback.
    summary_csv = training_root / f"{experiment_tag}_{dataset_name}" / "summary_results.csv"
    if summary_csv.exists():
        try:
            sdf = pd.read_csv(summary_csv)
            sub = sdf[(sdf["model_name"] == model) & (sdf["replicate"].astype(int) == int(replicate))]
            if len(sub) > 0 and "test_r2" in sub.columns:
                return _safe_float(sub.iloc[0]["test_r2"])
        except Exception:
            pass

    return float("nan")


def load_results_from_experiment(
    experiment_dir: Path,
    training_root_dir: Path,
    model_aliases: Dict[str, str],
    model_filter: Optional[set] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    experiment_tag = experiment_dir.name

    for summary_path in sorted(experiment_dir.glob("*/*/Run_*/summary_results.csv")):
        run_dir = summary_path.parent
        dataset_dir = run_dir.parent
        plant_dir = dataset_dir.parent

        target_plant = plant_dir.name
        dataset_name = dataset_dir.name

        try:
            sdf = pd.read_csv(summary_path)
        except Exception as exc:
            print(f"Warning: failed to read {summary_path}: {exc}")
            continue

        for _, row in sdf.iterrows():
            raw_model = str(row.get("model_name", "")).strip()
            if not raw_model:
                continue

            model = _normalize_model_name(raw_model, model_aliases)
            if model_filter and model not in model_filter:
                continue

            rep = int(_safe_float(row.get("replicate", np.nan)))
            if not np.isfinite(rep):
                continue

            rep_dir = run_dir / raw_model / f"Rep_{rep}"
            surrogate_cost = _safe_float(row.get("best_pred_vlab_cost", row.get("best_pred_lpfg_cost", np.nan)))
            verified_cost = _safe_float(row.get("best_vlab_cost_optimized_true", row.get("best_lpfg_cost_optimized_true", np.nan)))

            opt_values = {}
            for pname in PARAM_COLS:
                src = pname.replace("opt_", "best_")
                opt_values[pname] = _safe_float(row.get(src, np.nan))

            out_row = {
                "target_plant": target_plant,
                "model": model,
                "replicate": rep,
                "test_r2": _read_test_r2(
                    training_root=training_root_dir,
                    experiment_tag=experiment_tag,
                    dataset_name=dataset_name,
                    model=raw_model,
                    replicate=rep,
                ),
                "verified_vlab_cost": verified_cost,
                "surrogate_predicted_cost": surrogate_cost,
                "selected_restart": _read_selected_restart(
                    rep_dir=rep_dir,
                    surrogate_cost=surrogate_cost,
                    verified_cost=verified_cost,
                ),
                **opt_values,
            }
            rows.append(out_row)

    if not rows:
        raise ValueError(f"No usable summary_results.csv rows found under: {experiment_dir}")

    return pd.DataFrame(rows)


def load_norm_stats_from_datasets(datasets_dir: Path, target_plants: Iterable[str]) -> pd.DataFrame:
    records = []
    for plant in sorted(set(target_plants)):
        candidates = sorted(datasets_dir.glob(f"{plant}-*/Train.csv"))
        if not candidates:
            raise ValueError(f"No Train.csv found for plant {plant} in {datasets_dir}")

        # Prefer the stage-5 dataset size if present.
        preferred = [p for p in candidates if "-50000_10000_25000" in str(p)]
        train_csv = preferred[0] if preferred else candidates[0]

        tdf = pd.read_csv(train_csv)
        for idx, pname in enumerate(PARAM_COLS):
            col = f"param_{idx}"
            if col not in tdf.columns:
                raise ValueError(f"Missing column {col} in {train_csv}")
            vals = pd.to_numeric(tdf[col], errors="coerce")
            records.append(
                {
                    "target_plant": plant,
                    "parameter": pname,
                    "train_mean": float(vals.mean()),
                    "train_std": float(vals.std(ddof=0)),
                }
            )

    return pd.DataFrame(records)


def validate_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _sort_plants(plants: Iterable[str]) -> List[str]:
    return sorted(set(plants), key=lambda x: (str(x).split("_")[-1], str(x)))


def plot_verified_cost_paired(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    agg = (
        df.groupby(["target_plant", "model"], as_index=False)["verified_vlab_cost"]
        .agg(["mean", "std"]) 
        .reset_index()
        .rename(columns={"mean": "mean_verified_vlab_cost", "std": "std_verified_vlab_cost"})
    )

    plants = _sort_plants(agg["target_plant"].unique())
    models = sorted(agg["model"].unique())

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(plants))

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    palette = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    for i, model in enumerate(models):
        sub = agg[agg["model"] == model].set_index("target_plant")
        y = [sub.loc[p, "mean_verified_vlab_cost"] if p in sub.index else np.nan for p in plants]
        yerr = [sub.loc[p, "std_verified_vlab_cost"] if p in sub.index else np.nan for p in plants]
        offset = (i - (len(models) - 1) / 2.0) * 0.18
        ax.errorbar(
            x + offset,
            y,
            yerr=yerr,
            fmt="o",
            capsize=3,
            color=palette[model],
            label=model,
            markersize=5,
        )

    # Connect model means within each plant (if >=2 models).
    if len(models) >= 2:
        for pidx, plant in enumerate(plants):
            plant_sub = agg[agg["target_plant"] == plant]
            if len(plant_sub) < 2:
                continue
            xs = []
            ys = []
            for i, model in enumerate(models):
                model_row = plant_sub[plant_sub["model"] == model]
                if len(model_row) == 0:
                    continue
                xs.append(pidx + (i - (len(models) - 1) / 2.0) * 0.18)
                ys.append(float(model_row.iloc[0]["mean_verified_vlab_cost"]))
            if len(xs) >= 2:
                ax.plot(xs, ys, color="#9e9e9e", linewidth=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(plants, rotation=45, ha="right")
    ax.set_xlabel("Target Plant")
    ax.set_ylabel("Mean verified VLAB cost")
    ax.set_title("Final all-plant verified VLAB cost by model")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(output_dir / "fig_final_all_plant_verified_cost.pdf")
    fig.savefig(output_dir / "fig_final_all_plant_verified_cost.png", dpi=300)
    plt.close(fig)

    return agg


def plot_verified_cost_replicate_bars(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["verified_vlab_cost"] = pd.to_numeric(plot_df["verified_vlab_cost"], errors="coerce")
    plot_df["replicate"] = pd.to_numeric(plot_df["replicate"], errors="coerce")

    # Aggregate in case there are duplicate rows for the same plant/model/replicate.
    rep_cost = (
        plot_df.groupby(["target_plant", "model", "replicate"], as_index=False)["verified_vlab_cost"]
        .mean()
        .reset_index(drop=True)
    )

    plants = _sort_plants(rep_cost["target_plant"].unique())
    if not plants:
        return

    preferred_models = ["MLP", "Phytomorph"]
    models = [m for m in preferred_models if m in set(rep_cost["model"].astype(str).unique())]
    if not models:
        models = sorted(rep_cost["model"].astype(str).unique().tolist())[:2]
    if not models:
        return

    reps_per_model = 2
    rep_order: Dict[str, List[int]] = {}
    for model in models:
        model_reps = sorted(
            pd.to_numeric(
                rep_cost.loc[rep_cost["model"] == model, "replicate"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        rep_order[model] = model_reps[:reps_per_model]

    fig, ax = plt.subplots(figsize=(max(12, len(plants) * 0.7), 6))
    x = np.arange(len(plants), dtype=float)
    group_width = 0.82
    bars_per_plant = max(1, len(models) * reps_per_model)
    bar_width = group_width / bars_per_plant

    MLP_colors = ["#4C78A8", "#9EC1E6"]
    Phytomorph_colors = ["#F58518", "#FFBF79"]
    fallback_colors = ["#2CA02C", "#98DF8A"]

    for mi, model in enumerate(models):
        model_sub = rep_cost[rep_cost["model"] == model]
        model_lower = str(model).lower()
        if "mlp" in model_lower:
            model_colors = MLP_colors
        elif "phytomorph" in model_lower:
            model_colors = Phytomorph_colors
        else:
            model_colors = fallback_colors

        for ri in range(reps_per_model):
            rep_label = f"{model} rep {ri + 1}"
            if ri < len(rep_order[model]):
                rep_value = rep_order[model][ri]
                rep_sub = model_sub[model_sub["replicate"].astype(int) == int(rep_value)].set_index("target_plant")
                y = [rep_sub.loc[p, "verified_vlab_cost"] if p in rep_sub.index else np.nan for p in plants]
            else:
                y = [np.nan] * len(plants)

            offset_idx = mi * reps_per_model + ri
            x_pos = x - group_width / 2.0 + (offset_idx + 0.5) * bar_width

            valid = np.isfinite(np.asarray(y, dtype=float))
            if np.any(valid):
                ax.bar(
                    x_pos[valid],
                    np.asarray(y, dtype=float)[valid],
                    width=bar_width * 0.95,
                    color=model_colors[ri % len(model_colors)],
                    edgecolor="black",
                    linewidth=0.35,
                    label=rep_label,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(plants, rotation=45, ha="right")
    ax.set_xlabel("Target Plant")
    ax.set_ylabel("Verified VLAB cost")
    ax.set_title("Per-plant verified VLAB cost by model replicate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()

    fig.savefig(output_dir / "fig_verified_cost_replicate_bars.pdf")
    fig.savefig(output_dir / "fig_verified_cost_replicate_bars.png", dpi=300)
    plt.close(fig)


def plot_r2_vs_verified(df: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _plot_r2_vs_verified_impl(df: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        plot_df = df.copy()
        plot_df["replicate"] = pd.to_numeric(plot_df["replicate"], errors="coerce")
        plot_df["test_r2"] = pd.to_numeric(plot_df["test_r2"], errors="coerce")
        plot_df["verified_vlab_cost"] = pd.to_numeric(plot_df["verified_vlab_cost"], errors="coerce")

        agg = (
            plot_df.groupby(["target_plant", "model"], as_index=False)
            .agg(
                mean_test_r2=("test_r2", "mean"),
                std_test_r2=("test_r2", "std"),
                mean_verified_vlab_cost=("verified_vlab_cost", "mean"),
                std_verified_vlab_cost=("verified_vlab_cost", "std"),
            )
            .reset_index(drop=True)
        )

        plants = _sort_plants(agg["target_plant"].unique())
        models = [m for m in ["MLP", "Phytomorph"] if m in set(agg["model"].astype(str).unique())]
        if not models:
            models = sorted(agg["model"].astype(str).unique().tolist())

        markers = {"MLP": "o", "Phytomorph": "s"}
        faces = {"MLP": "black", "Phytomorph": "white"}

        cost_min = float(np.nanmin(agg["mean_verified_vlab_cost"].to_numpy()))
        cost_max = float(np.nanmax(agg["mean_verified_vlab_cost"].to_numpy()))
        cost_pad = max(500.0, (cost_max - cost_min) * 0.08)

        r2_min = float(np.nanmin(agg["mean_test_r2"].to_numpy()))
        r2_max = float(np.nanmax(agg["mean_test_r2"].to_numpy()))
        r2_low_min = min(r2_min - 0.05, -0.10)
        r2_low_max = 0.20
        r2_high_min = 0.95
        r2_high_max = max(1.0, r2_max + 0.01)

        fig = plt.figure(figsize=(16.0, max(6.5, len(plants) * 0.5)))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.22)
        ax_cost = fig.add_subplot(outer[0, 0])
        r2_gs = outer[0, 1].subgridspec(1, 2, width_ratios=[0.55, 1.0], wspace=0.05)
        ax_r2_lo = fig.add_subplot(r2_gs[0, 0], sharey=ax_cost)
        ax_r2_hi = fig.add_subplot(r2_gs[0, 1], sharey=ax_cost)

        def _draw_marker(ax, x_val: float, y_val: int, model: str) -> None:
            ax.scatter(
                x_val,
                y_val,
                s=78,
                alpha=1.0,
                facecolor=faces.get(model, "white"),
                edgecolor="black",
                linewidth=0.9,
                marker=markers.get(model, "o"),
                zorder=3,
            )

        for yidx, plant in enumerate(plants):
            plant_sub = agg[agg["target_plant"] == plant].set_index("model")
            if len(plant_sub) < 2:
                continue

            pairs = []
            for model in models:
                if model in plant_sub.index:
                    pairs.append((model, plant_sub.loc[model]))
            if len(pairs) != 2:
                continue

            cost_xs = [float(r[1]["mean_verified_vlab_cost"]) for r in pairs]
            ax_cost.plot(cost_xs, [yidx, yidx], color="#b0b0b0", linewidth=1.0, alpha=0.9, zorder=0)
            for model, row in pairs:
                _draw_marker(ax_cost, float(row["mean_verified_vlab_cost"]), yidx, model)

            r2_vals = [float(r[1]["mean_test_r2"]) for r in pairs]
            low_points = [(m, r) for (m, r), x in zip(pairs, r2_vals) if x <= r2_low_max]
            high_points = [(m, r) for (m, r), x in zip(pairs, r2_vals) if x >= r2_high_min]

            if len(low_points) == 2:
                ax_r2_lo.plot(
                    [float(low_points[0][1]["mean_test_r2"]), float(low_points[1][1]["mean_test_r2"])],
                    [yidx, yidx],
                    color="#b0b0b0",
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=0,
                )
            elif len(high_points) == 2:
                ax_r2_hi.plot(
                    [float(high_points[0][1]["mean_test_r2"]), float(high_points[1][1]["mean_test_r2"])],
                    [yidx, yidx],
                    color="#b0b0b0",
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=0,
                )
            elif len(low_points) == 1 and len(high_points) == 1:
                con = ConnectionPatch(
                    xyA=(float(high_points[0][1]["mean_test_r2"]), yidx),
                    coordsA=ax_r2_hi.transData,
                    xyB=(float(low_points[0][1]["mean_test_r2"]), yidx),
                    coordsB=ax_r2_lo.transData,
                    color="#b0b0b0",
                    linewidth=1.0,
                    alpha=0.9,
                )
                fig.add_artist(con)

            for model, row in pairs:
                x_val = float(row["mean_test_r2"])
                if x_val <= r2_low_max:
                    _draw_marker(ax_r2_lo, x_val, yidx, model)
                elif x_val >= r2_high_min:
                    _draw_marker(ax_r2_hi, x_val, yidx, model)

        ax_cost.set_xlabel("Verified VLAB cost")
        ax_cost.set_ylabel("Target plant")
        ax_cost.set_title("Verified VLAB cost")
        ax_cost.grid(axis="x", alpha=0.25)
        ax_cost.set_xlim(cost_min - cost_pad, cost_max + cost_pad)
        ax_cost.set_yticks(np.arange(len(plants)))
        ax_cost.set_yticklabels(plants)
        ax_cost.invert_yaxis()

        ax_r2_lo.set_xlim(r2_low_min, r2_low_max)
        ax_r2_hi.set_xlim(r2_high_min, r2_high_max)
        ax_r2_lo.set_title("Test R^2 (split axis)")
        ax_r2_hi.set_title("")
        ax_r2_lo.set_xlabel("Test R^2")
        ax_r2_hi.set_xlabel("Test R^2")
        ax_r2_lo.grid(axis="x", alpha=0.25)
        ax_r2_hi.grid(axis="x", alpha=0.25)
        ax_r2_lo.spines["right"].set_visible(False)
        ax_r2_hi.spines["left"].set_visible(False)
        ax_r2_lo.tick_params(labelright=False)
        ax_r2_hi.tick_params(labelleft=False)
        ax_r2_hi.yaxis.tick_right()
        ax_r2_hi.tick_params(axis="y", length=0)

        d = 0.015
        kwargs = dict(transform=ax_r2_lo.transAxes, color="k", clip_on=False, linewidth=1.0)
        ax_r2_lo.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax_r2_lo.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs = dict(transform=ax_r2_hi.transAxes, color="k", clip_on=False, linewidth=1.0)
        ax_r2_hi.plot((-d, +d), (-d, +d), **kwargs)
        ax_r2_hi.plot((-d, +d), (1 - d, 1 + d), **kwargs)

        model_handles = [
            Line2D([0], [0], marker=markers["MLP"], color="w", markerfacecolor=faces["MLP"], markeredgecolor="black", markersize=8, label="MLP"),
            Line2D([0], [0], marker=markers["Phytomorph"], color="w", markerfacecolor=faces["Phytomorph"], markeredgecolor="black", markersize=8, label="Phytomorph"),
        ]
        fig.legend(handles=model_handles, loc="upper center", ncols=2, frameon=False)
        fig.suptitle("Plant-wise model comparison: VLAB cost and test R^2", y=1.02)
        fig.tight_layout()

        fig.savefig(output_dir / "fig_r2_vs_verified_cost.pdf")
        fig.savefig(output_dir / "fig_r2_vs_verified_cost.png", dpi=300)
        plt.close(fig)

        corr_rows = []
        for model in models:
            sub = agg[agg["model"] == model].dropna(subset=["mean_test_r2", "mean_verified_vlab_cost"])
            if len(sub) < 2:
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = float(sub["mean_test_r2"].corr(sub["mean_verified_vlab_cost"], method="pearson"))
                spearman = float(sub["mean_test_r2"].corr(sub["mean_verified_vlab_cost"], method="spearman"))
            corr_rows.append(
                {
                    "model": model,
                    "pearson_r2_vs_verified_cost": pearson,
                    "spearman_r2_vs_verified_cost": spearman,
                }
            )

        corr_df = pd.DataFrame(corr_rows)
        corr_df.to_csv(output_dir / "table_r2_verified_correlation.csv", index=False)

        return agg, corr_df

    return _plot_r2_vs_verified_impl(df, output_dir)


def export_accuracy_utility_analysis(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    work = df.copy()
    work["replicate"] = pd.to_numeric(work["replicate"], errors="coerce")
    work["test_r2"] = pd.to_numeric(work["test_r2"], errors="coerce")
    work["verified_vlab_cost"] = pd.to_numeric(work["verified_vlab_cost"], errors="coerce")

    pm = (
        work.groupby(["target_plant", "model"], as_index=False)
        .agg(
            mean_test_r2=("test_r2", "mean"),
            mean_verified_vlab_cost=("verified_vlab_cost", "mean"),
            n_reps=("replicate", "count"),
        )
        .reset_index(drop=True)
    )

    rows: List[Dict] = []
    scipy_available = True
    try:
        from scipy.stats import pearsonr, spearmanr
    except Exception:
        scipy_available = False

    def _add_corr_block(label: str, frame: pd.DataFrame, r2_col: str, cost_col: str) -> None:
        sub = frame[[r2_col, cost_col]].dropna().copy()
        n = int(len(sub))
        if n >= 2:
            pearson_r = _safe_float(sub[r2_col].corr(sub[cost_col], method="pearson"))
            spearman_rho = _safe_float(sub[r2_col].corr(sub[cost_col], method="spearman"))
            if scipy_available:
                try:
                    pearson_p = _safe_float(pearsonr(sub[r2_col], sub[cost_col]).pvalue)
                except Exception:
                    pearson_p = float("nan")
                try:
                    spearman_p = _safe_float(spearmanr(sub[r2_col], sub[cost_col], nan_policy="omit").pvalue)
                except Exception:
                    spearman_p = float("nan")
            else:
                pearson_p = float("nan")
                spearman_p = float("nan")
        else:
            pearson_r = float("nan")
            spearman_rho = float("nan")
            pearson_p = float("nan")
            spearman_p = float("nan")
        rows.append(
            {
                "row_type": "correlation",
                "analysis_name": label,
                "n": n,
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
                "pearson_p_value": pearson_p,
                "spearman_p_value": spearman_p,
            }
        )

    _add_corr_block("all_means_plant_model_26", pm, "mean_test_r2", "mean_verified_vlab_cost")
    for model in sorted(pm["model"].dropna().astype(str).unique()):
        _add_corr_block(
            f"{model}_means_plant_13",
            pm[pm["model"].astype(str) == model],
            "mean_test_r2",
            "mean_verified_vlab_cost",
        )

    raw = work[["target_plant", "model", "replicate", "test_r2", "verified_vlab_cost"]].dropna(
        subset=["test_r2", "verified_vlab_cost"]
    )
    _add_corr_block("all_raw_reps", raw, "test_r2", "verified_vlab_cost")

    collapse_mask = (
        (raw["target_plant"].astype(str) == "Plant_104-24")
        & (raw["model"].astype(str) == "Phytomorph")
        & (raw["replicate"].astype(int) == 1)
    )
    _add_corr_block("all_raw_reps_minus_collapse", raw[~collapse_mask], "test_r2", "verified_vlab_cost")

    wide = (
        pm.pivot(index="target_plant", columns="model", values=["mean_test_r2", "mean_verified_vlab_cost"])
        .sort_index()
        .copy()
    )
    for col in [
        ("mean_test_r2", "MLP"),
        ("mean_test_r2", "Phytomorph"),
        ("mean_verified_vlab_cost", "MLP"),
        ("mean_verified_vlab_cost", "Phytomorph"),
    ]:
        if col not in wide.columns:
            wide[col] = np.nan

    count_total = 0
    count_MLP_higher_r2 = 0
    count_Phytomorph_lower_cost = 0
    count_higher_r2_also_lower_cost = 0

    for plant, prow in wide.iterrows():
        b_r2 = _safe_float(prow[("mean_test_r2", "MLP")])
        s_r2 = _safe_float(prow[("mean_test_r2", "Phytomorph")])
        b_cost = _safe_float(prow[("mean_verified_vlab_cost", "MLP")])
        s_cost = _safe_float(prow[("mean_verified_vlab_cost", "Phytomorph")])

        if not (np.isfinite(b_r2) and np.isfinite(s_r2) and np.isfinite(b_cost) and np.isfinite(s_cost)):
            continue

        count_total += 1
        if b_r2 > s_r2:
            higher_r2_model = "MLP"
            count_MLP_higher_r2 += 1
        elif s_r2 > b_r2:
            higher_r2_model = "Phytomorph"
        else:
            higher_r2_model = "tie"

        if s_cost < b_cost:
            lower_cost_model = "Phytomorph"
            count_Phytomorph_lower_cost += 1
        elif b_cost < s_cost:
            lower_cost_model = "MLP"
        else:
            lower_cost_model = "tie"

        higher_r2_also_lower_cost = int(
            (higher_r2_model == "MLP" and lower_cost_model == "MLP")
            or (higher_r2_model == "Phytomorph" and lower_cost_model == "Phytomorph")
        )
        count_higher_r2_also_lower_cost += higher_r2_also_lower_cost

        rows.append(
            {
                "row_type": "per_plant",
                "analysis_name": "model_comparison_means",
                "target_plant": plant,
                "MLP_mean_test_r2": b_r2,
                "Phytomorph_mean_test_r2": s_r2,
                "MLP_mean_verified_vlab_cost": b_cost,
                "Phytomorph_mean_verified_vlab_cost": s_cost,
                "delta_r2_MLP_minus_Phytomorph": b_r2 - s_r2,
                "delta_cost_MLP_minus_Phytomorph": b_cost - s_cost,
                "higher_r2_model": higher_r2_model,
                "lower_cost_model": lower_cost_model,
                "higher_r2_also_lower_cost": higher_r2_also_lower_cost,
            }
        )

    rows.extend(
        [
            {
                "row_type": "count_summary",
                "analysis_name": "higher_r2_model_also_lower_cost",
                "count_value": count_higher_r2_also_lower_cost,
                "count_total": count_total,
            },
            {
                "row_type": "count_summary",
                "analysis_name": "MLP_higher_r2",
                "count_value": count_MLP_higher_r2,
                "count_total": count_total,
            },
            {
                "row_type": "count_summary",
                "analysis_name": "Phytomorph_lower_cost",
                "count_value": count_Phytomorph_lower_cost,
                "count_total": count_total,
            },
        ]
    )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "table_accuracy_utility_analysis.csv", index=False)

    txt_lines = []
    txt_lines.append("Accuracy vs Utility Analysis")
    txt_lines.append("================================")
    txt_lines.append("")
    txt_lines.append(f"scipy_available_for_p_values: {scipy_available}")
    txt_lines.append("")
    txt_lines.append("Correlation blocks")
    txt_lines.append("------------------")
    corr_df = out_df[out_df["row_type"] == "correlation"].copy()
    for _, r in corr_df.iterrows():
        txt_lines.append(
            f"{r.get('analysis_name', '')}: "
            f"n={int(r.get('n', 0))}, "
            f"pearson_r={_safe_float(r.get('pearson_r')):.6f}, "
            f"pearson_p={_safe_float(r.get('pearson_p_value')):.6g}, "
            f"spearman_rho={_safe_float(r.get('spearman_rho')):.6f}, "
            f"spearman_p={_safe_float(r.get('spearman_p_value')):.6g}"
        )

    txt_lines.append("")
    txt_lines.append("Count summaries")
    txt_lines.append("---------------")
    cnt_df = out_df[out_df["row_type"] == "count_summary"].copy()
    for _, r in cnt_df.iterrows():
        val = _safe_float(r.get("count_value"))
        total = _safe_float(r.get("count_total"))
        txt_lines.append(f"{r.get('analysis_name', '')}: {int(val)}/{int(total)}")

    txt_lines.append("")
    txt_lines.append("Per-plant model comparison")
    txt_lines.append("--------------------------")
    plant_df = out_df[out_df["row_type"] == "per_plant"].copy()
    for _, r in plant_df.iterrows():
        txt_lines.append(
            f"{r.get('target_plant', '')}: "
            f"MLP_r2={_safe_float(r.get('MLP_mean_test_r2')):.4f}, "
            f"Phytomorph_r2={_safe_float(r.get('Phytomorph_mean_test_r2')):.4f}, "
            f"MLP_cost={_safe_float(r.get('MLP_mean_verified_vlab_cost')):.1f}, "
            f"Phytomorph_cost={_safe_float(r.get('Phytomorph_mean_verified_vlab_cost')):.1f}, "
            f"higher_r2={r.get('higher_r2_model', '')}, "
            f"lower_cost={r.get('lower_cost_model', '')}"
        )

    (output_dir / "table_accuracy_utility_analysis.txt").write_text("\n".join(txt_lines) + "\n")
    return out_df


def _merge_stats(results_df: pd.DataFrame, norm_df: pd.DataFrame) -> pd.DataFrame:
    long_rows = []
    for _, row in results_df.iterrows():
        for p in PARAM_COLS:
            long_rows.append(
                {
                    "target_plant": row["target_plant"],
                    "model": row["model"],
                    "replicate": row["replicate"],
                    "selected_restart": row["selected_restart"],
                    "parameter": p,
                    "raw_value": _safe_float(row[p]),
                }
            )
    long_df = pd.DataFrame(long_rows)
    merged = long_df.merge(norm_df, on=["target_plant", "parameter"], how="left")

    std = pd.to_numeric(merged["train_std"], errors="coerce")
    std = std.mask(std.abs() < 1e-12, np.nan)

    merged["z_score"] = (pd.to_numeric(merged["raw_value"], errors="coerce") - pd.to_numeric(merged["train_mean"], errors="coerce")) / std
    merged["abs_z_score"] = merged["z_score"].abs()
    return merged


def boundary_abs_heatmap(boundary_long: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    heat = (
        boundary_long.groupby(["target_plant", "model", "parameter"], as_index=False)
        .agg(mean_abs_z_score=("abs_z_score", "mean"), max_abs_z_score=("abs_z_score", "max"))
        .reset_index(drop=True)
    )
    heat.to_csv(output_dir / "boundary_abs_zscore_heatmap_data.csv", index=False)

    models = sorted(heat["model"].unique())
    plants = _sort_plants(heat["target_plant"].unique())
    params = list(PARAM_COLS)

    fig, axes = plt.subplots(
        nrows=max(1, len(models)),
        ncols=1,
        figsize=(max(12, len(params) * 0.8), max(5.5, len(models) * 4.4)),
        squeeze=False,
        constrained_layout=True,
    )

    for ridx, model in enumerate(models):
        ax = axes[ridx, 0]
        sub = heat[heat["model"] == model]
        mat = np.full((len(plants), len(params)), np.nan, dtype=float)

        for i, plant in enumerate(plants):
            for j, param in enumerate(params):
                m = sub[(sub["target_plant"] == plant) & (sub["parameter"] == param)]
                if len(m) > 0:
                    mat[i, j] = float(m.iloc[0]["mean_abs_z_score"])

        mat_plot = np.clip(mat, 0.0, 2.0)
        im = ax.imshow(mat_plot, aspect="auto", vmin=0.0, vmax=2.0, cmap="viridis")
        ax.set_yticks(np.arange(len(plants)))
        ax.set_yticklabels(plants)
        ax.set_xticks(np.arange(len(params)))
        ax.set_xticklabels(params, rotation=45, ha="right")
        ax.set_title(f"Boundary proximity heatmap ({model})")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Mean absolute z-score, |z|")

    fig.savefig(output_dir / "fig_boundary_abs_zscore_heatmap.pdf")
    fig.savefig(output_dir / "fig_boundary_abs_zscore_heatmap.png", dpi=300)
    plt.close(fig)

    return heat


def boundary_hit_metrics(boundary_long: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cand = (
        boundary_long.groupby(["target_plant", "model", "replicate", "selected_restart"], as_index=False)
        .agg(
            boundary_hit_count_1p9=("abs_z_score", lambda s: int(np.sum(s >= 1.9))),
            near_boundary_hit_count_1p8=("abs_z_score", lambda s: int(np.sum(s >= 1.8))),
            mean_abs_z_score=("abs_z_score", "mean"),
            max_abs_z_score=("abs_z_score", "max"),
        )
        .reset_index(drop=True)
    )

    cand["boundary_hit_rate_1p9"] = cand["boundary_hit_count_1p9"] / float(len(PARAM_COLS))
    cand["near_boundary_hit_rate_1p8"] = cand["near_boundary_hit_count_1p8"] / float(len(PARAM_COLS))

    cand = cand[
        [
            "target_plant",
            "model",
            "replicate",
            "selected_restart",
            "boundary_hit_count_1p9",
            "boundary_hit_rate_1p9",
            "near_boundary_hit_count_1p8",
            "near_boundary_hit_rate_1p8",
            "mean_abs_z_score",
            "max_abs_z_score",
        ]
    ]

    cand.to_csv(output_dir / "boundary_metrics_by_candidate.csv", index=False)

    model = (
        cand.groupby("model", as_index=False)
        .agg(
            mean_boundary_hit_count_1p9=("boundary_hit_count_1p9", "mean"),
            std_boundary_hit_count_1p9=("boundary_hit_count_1p9", "std"),
            median_boundary_hit_count_1p9=("boundary_hit_count_1p9", "median"),
            mean_boundary_hit_rate_1p9=("boundary_hit_rate_1p9", "mean"),
            std_boundary_hit_rate_1p9=("boundary_hit_rate_1p9", "std"),
            median_boundary_hit_rate_1p9=("boundary_hit_rate_1p9", "median"),
            mean_near_boundary_hit_rate_1p8=("near_boundary_hit_rate_1p8", "mean"),
            std_near_boundary_hit_rate_1p8=("near_boundary_hit_rate_1p8", "std"),
            median_near_boundary_hit_rate_1p8=("near_boundary_hit_rate_1p8", "median"),
            mean_abs_z_score=("mean_abs_z_score", "mean"),
            std_abs_z_score=("mean_abs_z_score", "std"),
            median_abs_z_score=("mean_abs_z_score", "median"),
            mean_max_abs_z_score=("max_abs_z_score", "mean"),
            std_max_abs_z_score=("max_abs_z_score", "std"),
            median_max_abs_z_score=("max_abs_z_score", "median"),
        )
        .reset_index(drop=True)
    )

    model.to_csv(output_dir / "boundary_metrics_by_model.csv", index=False)
    return cand, model


def run_full_analysis(results_df: pd.DataFrame, norm_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_columns(results_df, REQUIRED_RESULT_COLS, "results_df")
    validate_columns(norm_df, REQUIRED_NORM_COLS, "norm_df")

    # Ensure numeric columns are numeric for aggregation/correlation.
    for c in ["replicate", "test_r2", "verified_vlab_cost", "surrogate_predicted_cost", "selected_restart"] + PARAM_COLS:
        results_df[c] = pd.to_numeric(results_df[c], errors="coerce")

    norm_df["train_mean"] = pd.to_numeric(norm_df["train_mean"], errors="coerce")
    norm_df["train_std"] = pd.to_numeric(norm_df["train_std"], errors="coerce")

    # 1) Final all-plant verified-cost comparison.
    plot_verified_cost_paired(results_df, output_dir)
    plot_verified_cost_replicate_bars(results_df, output_dir)

    # 2) Predictive accuracy vs optimization utility + correlation table.
    missing_r2 = results_df["test_r2"].isna().sum()
    if missing_r2 > 0:
        print(
            f"Warning: test_r2 missing for {missing_r2}/{len(results_df)} rows. "
            "R2-vs-cost plot/correlations will use available rows only."
        )
    plot_r2_vs_verified(results_df, output_dir)
    export_accuracy_utility_analysis(results_df, output_dir)

    # 3A/3B) Boundary analyses.
    boundary_long = _merge_stats(results_df, norm_df)
    boundary_abs_heatmap(boundary_long, output_dir)
    boundary_hit_metrics(boundary_long, output_dir)


def _parse_aliases(alias_items: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for item in alias_items:
        if "=" not in item:
            raise ValueError(f"Invalid alias '{item}'. Expected format old=new")
        src, dst = item.split("=", 1)
        aliases[src.strip()] = dst.strip()
    return aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final all-plant analysis and boundary exploitation plots")

    parser.add_argument("--use-top-config", action="store_true", help="Use USER_* variables at the top of this file")
    parser.add_argument("--experiment-dir", type=str, default="", help="Path to optimizer experiment root (contains plant folders)")
    parser.add_argument("--datasets-dir", type=str, default="", help="Path to datasets root (contains Plant_*-*/Train.csv)")
    parser.add_argument("--training-root-dir", type=str, default="", help="Path to Training Data root")
    parser.add_argument("--results-csv", type=str, default="", help="Explicit replicate-level input CSV (alternative to --experiment-dir)")
    parser.add_argument("--norm-stats-csv", type=str, default="", help="Explicit normalization stats CSV")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for generated outputs")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model labels to keep")
    parser.add_argument(
        "--model-alias",
        action="append",
        default=[],
        help="Optional model rename mapping old=new. Can be repeated.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cli_mode = len(os.sys.argv) > 1 and not args.use_top_config
    use_top = USER_USE_TOP_CONFIG and not cli_mode
    if args.use_top_config:
        use_top = True

    if use_top:
        experiment_dir = _resolve(USER_EXPERIMENT_DIR)
        datasets_dir = _resolve(USER_DATASETS_DIR)
        training_root_dir = _resolve(USER_TRAINING_ROOT_DIR)
        results_csv = _resolve(USER_RESULTS_CSV) if USER_RESULTS_CSV else None
        norm_stats_csv = _resolve(USER_NORM_STATS_CSV) if USER_NORM_STATS_CSV else None
        output_dir = _resolve(USER_OUTPUT_DIR) if USER_OUTPUT_DIR else experiment_dir
        models = set(USER_MODELS) if USER_MODELS else None
        aliases = dict(USER_MODEL_ALIASES)
    else:
        script_dir = _script_dir()
        experiment_dir = _resolve(args.experiment_dir, script_dir) if args.experiment_dir else None
        datasets_dir = _resolve(args.datasets_dir, script_dir) if args.datasets_dir else None
        training_root_dir = _resolve(args.training_root_dir, script_dir) if args.training_root_dir else None
        results_csv = _resolve(args.results_csv, script_dir) if args.results_csv else None
        norm_stats_csv = _resolve(args.norm_stats_csv, script_dir) if args.norm_stats_csv else None
        output_dir = _resolve(args.output_dir, script_dir) if args.output_dir else (experiment_dir or script_dir)
        models = set(args.models) if args.models else None
        aliases = _parse_aliases(args.model_alias)

    if not aliases:
        aliases = {
            "baseline": "MLP",
            "sinkhorn_no_aggregator": "Phytomorph",
        }

    if results_csv is not None:
        results_df = pd.read_csv(results_csv)
        if models:
            results_df = results_df[results_df["model"].isin(models)].copy()
    else:
        if experiment_dir is None:
            raise ValueError("Provide --results-csv or --experiment-dir")
        if training_root_dir is None:
            training_root_dir = _resolve("Training Data")
        results_df = load_results_from_experiment(
            experiment_dir=experiment_dir,
            training_root_dir=training_root_dir,
            model_aliases=aliases,
            model_filter=models,
        )

    loaded_plants = sorted(results_df["target_plant"].astype(str).unique().tolist())
    loaded_models = sorted(results_df["model"].astype(str).unique().tolist())
    print(f"Loaded rows: {len(results_df)}")
    print(f"Loaded plants ({len(loaded_plants)}): {loaded_plants}")
    print(f"Loaded models ({len(loaded_models)}): {loaded_models}")

    if norm_stats_csv is not None:
        norm_df = pd.read_csv(norm_stats_csv)
    else:
        if datasets_dir is None:
            datasets_dir = _resolve("Datasets")
        norm_df = load_norm_stats_from_datasets(datasets_dir, results_df["target_plant"].unique())

    run_full_analysis(results_df=results_df, norm_df=norm_df, output_dir=output_dir)

    print("Analysis complete. Wrote files:")
    for name in [
        "fig_final_all_plant_verified_cost.pdf",
        "fig_final_all_plant_verified_cost.png",
        "fig_verified_cost_replicate_bars.pdf",
        "fig_verified_cost_replicate_bars.png",
        "fig_r2_vs_verified_cost.pdf",
        "fig_r2_vs_verified_cost.png",
        "table_r2_verified_correlation.csv",
        "table_accuracy_utility_analysis.csv",
        "table_accuracy_utility_analysis.txt",
        "fig_boundary_abs_zscore_heatmap.pdf",
        "fig_boundary_abs_zscore_heatmap.png",
        "boundary_abs_zscore_heatmap_data.csv",
        "boundary_metrics_by_candidate.csv",
        "boundary_metrics_by_model.csv",
    ]:
        print(f"  - {output_dir / name}")


# Backward-compatible no-op for legacy imports from train_models.py.
def evaluate_run(*args, **kwargs):
    print(
        "evaluate_run() is deprecated in this rewritten evaluate_models.py. "
        "Use main() to generate final all-plant analysis artifacts."
    )


if __name__ == "__main__":
    main()
