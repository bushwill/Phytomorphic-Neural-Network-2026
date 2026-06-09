"""
Hierarchical Surrogate Optimization Script.

Purpose:
    Utilizes pre-trained, fully differentiable surrogate neural networks (Sinkhorn, Hungarian, MLP)
    to discover optimal Procedural Generation (L-System) parameters for a target biological phenotype.

Methodology:
    Because direct parameter gradient descent is susceptible to out-of-bound errors and vanishing gradients,
    we deploy an implicit Generator Network (`OptimizerNet`). By freezing the Surrogate Network's weights and 
    backpropagating the hierarchical structural loss through it into the Generator Network, the Adam optimizer 
    can robustly navigate the continuous parameter space to minimize topological discrepancies between 
    the synthesized output and the ground-truth real plant architecture.

Execution:
    This script is designed to be executed mechanically via bash orchestration pipelines (e.g. `experiment.sh`).
"""

import os
import sys
import csv
import time
import shutil
import uuid
import logging
import argparse
import subprocess
import numpy as np
import torch
import concurrent.futures

# Prevent docker generated files from locking out host user
os.umask(0)
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
import utils_nn  # Ensure we have access to the module's root scope to set the variable

# --- Local Imports ---
# Cleanly import project modules, handling potential path issues
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils_nn import read_real_plants, calculate_cost, build_parameter_file, read_syn_plant
    import model_mlp as benchmark_model
    import model_hungarian as baseline_model
    import model_sinkhorn as sinkhorn_model
except ImportError as e:
    print(f"Error: Could not import project modules: {e}")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LSYSTEM_PARAM_NAMES = [
    "max_phytomers",
    "plastochron",
    "plant_roll_angle",
    "plant_down_angle",
    "branch_angle",
    "leaf_len",
    "exp_leaf_wid",
    "leaf_wid",
    "leaf_bend_scale",
    "leaf_twist_scale",
    "node_len",
    "int_wid",
    "exp_int_rad",
]

# --- Configuration Constants ---
# Edit these defaults to control optimizer behavior without touching CLI args.
DEFAULT_MODEL_RUN_DIR = "Training Data/Run_041326"
DEFAULT_MODEL_FILES = []
DEFAULT_MODEL_FAMILIES = ["baseline", "hungarian", "sinkhorn"]
DEFAULT_OUTPUT_DIR = "Optimizer Data"
DEFAULT_RESTARTS = 10
DEFAULT_STEPS = 1000
DEFAULT_LR = 0.1
DEFAULT_DRY_RUN = False
DEFAULT_LOG_EVERY = 25
DEFAULT_VERIFY_EACH_RESTART = True
DEFAULT_PARAM_JITTER_STD = 0.01
DEFAULT_BOUND_SIGMA = 2.0
SINKHORN_ABLATION_FAMILIES = {
    "sinkhorn_no_encoder",
    "sinkhorn_no_scaler",
    "sinkhorn_no_aggregator",
    "sinkhorn_hollow",
}
SUPPORTED_MODEL_FAMILIES = {"baseline", "hungarian", "sinkhorn", *SINKHORN_ABLATION_FAMILIES}

# Fallback parameter constraints (Min, Max)
# Corresponds to: [max_phytomers, plastochron, roll, down, branch, leaf_len, exp_wid, leaf_wid, bend, twist, node, int_wid, exp_rad]
# These are only used if a loaded surrogate does not expose normalization stats.
PARAM_MIN = torch.tensor([8.0, 2.8, -110.0, -4.0, 125.0, 3.0, 0.48, 0.8, 80.0, 170.0, 0.6, 0.88, 0.48])
PARAM_MAX = torch.tensor([12.0, 3.2, 110.0, 4.0, 145.0, 7.0, 0.52, 1.2, 100.0, 190.0, 0.8, 0.92, 0.52])

def derive_parameter_bounds_from_model(model, sigma_multiplier=DEFAULT_BOUND_SIGMA):
    """Derive optimizer bounds from the surrogate's own input normalization stats."""
    input_mean = getattr(model, "input_mean", None)
    input_std = getattr(model, "input_std", None)

    if input_mean is None or input_std is None:
        return PARAM_MIN.clone(), PARAM_MAX.clone(), False

    input_mean = torch.as_tensor(input_mean, dtype=torch.float32).flatten()
    input_std = torch.as_tensor(input_std, dtype=torch.float32).flatten().clamp_min(1e-6)

    if input_mean.numel() != PARAM_MIN.numel() or input_std.numel() != PARAM_MIN.numel():
        return PARAM_MIN.clone(), PARAM_MAX.clone(), False

    param_min = input_mean - sigma_multiplier * input_std
    param_max = input_mean + sigma_multiplier * input_std
    return param_min, param_max, True

class OptimizerNet(nn.Module):
    """
    A lightweight neural network used as a trainable parameter generator.
    
    Why use this?
    Directly optimizing a tensor `params = torch.tensor(..., requires_grad=True)` can sometimes 
    lead to issues with constraints or gradients. Using a small network that outputs 
    the parameters (passed through a Sigmoid to enforce 0-1 range, then scaled) 
    keeps the optimization process stable and allows us to use standard NN optimizers.
    """
    def __init__(self, input_dim=1, output_dim=13, param_min=None, param_max=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()  # Force output to [0, 1]
        )
        # Register boundaries as buffers (not trainable parameters)
        if param_min is None:
            param_min = PARAM_MIN
        if param_max is None:
            param_max = PARAM_MAX
        self.register_buffer('param_min', torch.as_tensor(param_min, dtype=torch.float32))
        self.register_buffer('param_max', torch.as_tensor(param_max, dtype=torch.float32))

    def forward(self, x):
        # 1. Generate normalized parameters [0, 1]
        out = self.net(x)
        # 2. Scale to actual parameter range [min, max]
        return out * (self.param_max - self.param_min) + self.param_min

def setup_logging(output_dir):
    """Configures logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "optimizer_log.txt")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_run_output_dir(output_root):
    """Create a dated run folder under the optimizer output root."""
    date_str = datetime.now().strftime("%m%d%y")
    run_name = f"Run_{date_str}"
    candidate = run_name
    counter = 0

    while os.path.exists(os.path.join(output_root, candidate)):
        counter += 1
        candidate = f"{run_name}_{counter}"

    run_output_dir = os.path.join(output_root, candidate)
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir, candidate

def resolve_model_output_dir(run_output_dir, model_path, run_dir=None):
    """Mirror the source training path inside the optimizer run folder."""
    model_dir = os.path.dirname(os.path.abspath(model_path))

    if run_dir:
        run_dir_abs = os.path.abspath(run_dir)
        model_dir_abs = os.path.abspath(model_dir)
        try:
            rel_dir = os.path.relpath(model_dir_abs, run_dir_abs)
        except ValueError:
            rel_dir = os.path.basename(model_dir_abs)

        if rel_dir.startswith(".."):  # model is outside the searched run folder
            rel_dir = os.path.basename(model_dir_abs)
    else:
        rel_dir = os.path.basename(model_dir)

    return os.path.join(run_output_dir, rel_dir)

def extract_model_family(model_path):
    """
    Extract model family from known training layouts:
      - <run>/<family>/Rep_n/best_model.pt
      - <run>/<family>/best_model.pt
    """
    path_parts = [part.lower() for part in Path(model_path).parts]
    for part in reversed(path_parts):
        if part in SUPPORTED_MODEL_FAMILIES:
                        return part
    return os.path.basename(os.path.dirname(os.path.abspath(model_path))).lower()

def normalize_model_family(family):
    """Map supported model folders to the surrogate implementation they use."""
    if family in {"sinkhorn", *SINKHORN_ABLATION_FAMILIES}:
        return "sinkhorn"
    return family

def get_sinkhorn_model_kwargs(family):
    """Reconstruct the Sinkhorn architecture flags used when training a checkpoint."""
    if family == "sinkhorn_no_encoder":
        return {"use_encoder": False, "use_scaler": True, "use_aggregator": True}
    if family == "sinkhorn_no_scaler":
        return {"use_encoder": True, "use_scaler": False, "use_aggregator": True}
    if family == "sinkhorn_no_aggregator":
        return {"use_encoder": True, "use_scaler": True, "use_aggregator": False}
    if family == "sinkhorn_hollow":
        return {"use_encoder": False, "use_scaler": False, "use_aggregator": False}
    return {"use_encoder": True, "use_scaler": True, "use_aggregator": True}

def load_model(model_path):
    """
    Identifies and loads a trained surrogate model from a .pt file.
    Returns (model_instance, model_type_string).
    """
    path_str = str(model_path)
    variant = extract_model_family(path_str)
    model_family = normalize_model_family(variant)
    model_kwargs = get_sinkhorn_model_kwargs(variant) if model_family == "sinkhorn" else {}

    if model_family == "sinkhorn":
        logging.info(f"Loading {path_str} (Type: Sinkhorn Hierarchical)")
        ModelClass = sinkhorn_model.HierarchicalPlantSurrogateNet
        model_type = "sinkhorn"
    elif model_family == "baseline":
        logging.info(f"Loading {path_str} (Type: Baseline MLP)")
        ModelClass = benchmark_model.BenchmarkSurrogateNet
        model_type = "mlp"
    elif model_family == "hungarian":
        logging.info(f"Loading {path_str} (Type: Hungarian Hierarchical)")
        ModelClass = baseline_model.HierarchicalPlantSurrogateNet
        model_type = "hungarian"
    else:
        logging.error(
            f"Could not determine model family from path: {path_str}. "
            f"Expected family folder in {sorted(SUPPORTED_MODEL_FAMILIES)}."
        )
        return None, None

    model = ModelClass(**model_kwargs)
    try:
        # Load weights (map to CPU for safety)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model, model_type
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        return None, None

def run_simulation_verification(params, output_dir="Optimizer Data/Verify"):
    """
    Runs the actual LPFG simulation with the optimized parameters
    to verify the 'True Cost' vs. the 'Predicted Cost'.
    """
    worker_ws = None
    param_file = None
    output_txt = None
    leafposition_path = None
    verify_tmp_dir = None
    os.makedirs(output_dir, exist_ok=True)
    try:
        temp_id = uuid.uuid4().hex[:6]
        lsystem_dir = os.path.join(SCRIPT_DIR, "lsystem")
        verify_tmp_root = os.path.join(lsystem_dir, "verify_tmp")
        os.makedirs(verify_tmp_root, exist_ok=True)
        worker_ws = os.path.join(verify_tmp_root, f"worker_{temp_id}")
        os.makedirs(worker_ws, exist_ok=True)
        
        # Copy essential files to an isolated workspace for this worker
        for item in os.listdir(lsystem_dir):
            src = os.path.join(lsystem_dir, item)
            if os.path.isfile(src) and not item.endswith('.o') and not item.endswith('.so'):
                shutil.copy2(src, os.path.join(worker_ws, item))

        # Recompile project.cpp safely in the isolated path if necessary
        if not os.path.exists(os.path.join(worker_ws, "project")):
            os.system(f"g++ -o {os.path.join(worker_ws, 'project')} -Wall -Wextra {os.path.join(worker_ws, 'project.cpp')} -lm")
            
        verify_tmp_dir = os.path.join(worker_ws, f"run_{temp_id}")
        os.makedirs(verify_tmp_dir, exist_ok=True)
        param_file = os.path.join(worker_ws, f"opt_{temp_id}.vset")
        
        # Write parameters to file.
        build_parameter_file(param_file, params.tolist())
        
        # 1. LPFG (Generate Structure)
        # Assumes 'lpfg' is in PATH or otherwise discoverable by the shell.
        lsystem_l = os.path.join(worker_ws, "lsystem.l")
        view_v = os.path.join(worker_ws, "view.v")
        materials_mat = os.path.join(worker_ws, "materials.mat")
        contours_cset = os.path.join(worker_ws, "contours.cset")
        functions_fset = os.path.join(worker_ws, "functions.fset")
        functions_tset = os.path.join(worker_ws, "functions.tset")

        lpfg_cmd = [
            "lpfg", "-w", "306", "256",
            lsystem_l, view_v, materials_mat,
            contours_cset, functions_fset, functions_tset,
            param_file,
        ]
        
        # We explicitly run CWD into this isolated environment and wait sequentially
        lpfg_result = subprocess.run(lpfg_cmd, capture_output=True, text=True, check=True, cwd=worker_ws)

        if lpfg_result.stderr:
            # LPFG can emit useful warnings to stderr even on success.
            logging.info(f"LPFG message: {lpfg_result.stderr.strip()}")

        leafposition_path = os.path.join(worker_ws, "leafposition.dat")
        project_bin = os.path.join(worker_ws, "project")
        project_src = os.path.join(worker_ws, "project.cpp")

        # 2. Project (Extract geometry to leafposition.dat)
        if not os.path.exists(leafposition_path):
            logging.error("LPFG completed but leafposition.dat was not generated.")
            return None, None

        # 3. Process Geometry
        # This requires the 'project' binary (compiled from C++).
        if not os.path.exists(project_bin):
            logging.info("Compiling 'project' tool (one-time)...")
            compile_cmd = ["g++", "-o", project_bin, "-Wall", "-Wextra", project_src, "-lm"]
            compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
            if compile_result.returncode != 0:
                logging.error(f"Failed to compile project tool: {(compile_result.stderr or '').strip()}")
                return None, None
            
        output_txt = os.path.join(verify_tmp_dir, f"output_{temp_id}.txt")
        with open(output_txt, "w") as out_f:
            proj_result = subprocess.run(
                [project_bin, "2454", "2056", leafposition_path],
                stdout=out_f,
                stderr=subprocess.PIPE,
                text=True,
                cwd=SCRIPT_DIR,
                check=True,
            )
            if proj_result.stderr:
                logging.info(f"project message: {proj_result.stderr.strip()}")
        
        # 4. Read Data
        syn_bp, syn_ep = read_syn_plant(output_txt)
        return syn_bp, syn_ep
    except FileNotFoundError:
        logging.error("LPFG execution failed. Ensure 'lpfg' is in PATH.")
        return None, None
    except subprocess.CalledProcessError as e:
        stderr_msg = (e.stderr or "").strip()
        if stderr_msg:
            logging.error(f"LPFG execution failed: {stderr_msg}")
        else:
            logging.error("LPFG execution failed with non-zero exit code.")
        return None, None
    except Exception as e:
        logging.error(f"Verification failed: {e}")
        return None, None
    finally:
        if worker_ws and os.path.exists(worker_ws):
            shutil.rmtree(worker_ws, ignore_errors=True)

def evaluate_real_cost(syn_bp, syn_ep, real_bp, real_ep):
    """Calculates the structural cost between generated (syn) and real plant data."""
    if not syn_bp:
        return float('inf')
        
    total_cost = 0.0
    num_days = min(len(syn_bp), len(real_bp))
    
    if num_days == 0:
        return float('inf')
        
    for i in range(num_days):
        total_cost += calculate_cost(syn_bp[i], syn_ep[i], real_bp[i], real_ep[i])
        
    return total_cost

def cleanup_empty_verify_dirs(model_output_dir):
    """Remove empty verification folders left behind by temporary LPFG checks."""
    verify_root = os.path.join(model_output_dir, "Verify")
    if not os.path.isdir(verify_root):
        return

    for current_dir, _, _ in os.walk(verify_root, topdown=False):
        if not os.listdir(current_dir):
            os.rmdir(current_dir)

def _safe_float(value):
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _extract_optimizer_record(result_path):
    try:
        with open(result_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            record = next(reader, None)
    except Exception as e:
        logging.warning(f"Skipping optimizer result read for {result_path}: {e}")
        return None

    if not record:
        return None

    row = {
        "best_pred_lpfg_cost": _safe_float(record.get("surrogate_pred_cost", "nan")),
        "best_lpfg_cost_optimized_true": _safe_float(record.get("verified_sim_cost", "nan")),
    }

    for idx, name in enumerate(LSYSTEM_PARAM_NAMES):
        named_key = f"opt_{name}"
        legacy_key = f"param_{idx}"
        row[f"best_{name}"] = _safe_float(record.get(named_key, record.get(legacy_key, "nan")))

    row["optimized_true"] = bool(np.isfinite(row["best_lpfg_cost_optimized_true"]))
    return row


def _load_tuning_r2_map(run_dir):
    """Scan run_dir for tuning_summary.csv files and return mapping:
    { model_name: { replicate_str: best_val_r2_float } }
    """
    mapping = {}
    if not run_dir:
        return mapping
    for root, _, files in os.walk(run_dir):
        if "tuning_summary.csv" not in files:
            continue
        path = os.path.join(root, "tuning_summary.csv")
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = row.get("model", "")
                    replicate = str(row.get("replicate", ""))
                    try:
                        val = float(row.get("best_val_r2", "nan"))
                    except Exception:
                        val = float("nan")
                    if model:
                        mapping.setdefault(model, {})[replicate] = val
        except Exception:
            logging.warning(f"Failed to read tuning_summary.csv at {path}")
    return mapping


def build_optimizer_summaries(run_output_dir, run_dir=None):
    """Collect optimizer results into run-level and model-level summary CSV files."""
    summary_csv_path = os.path.join(run_output_dir, "summary.csv")
    if os.path.exists(summary_csv_path):
        try:
            os.remove(summary_csv_path)
        except OSError as e:
            logging.warning(f"Could not remove stale summary.csv at {summary_csv_path}: {e}")

    summary_results_path = os.path.join(run_output_dir, "summary_results.csv")
    summary_by_model_path = os.path.join(run_output_dir, "summary_by_model.csv")
    best_lpfg_report_path = os.path.join(run_output_dir, "best_lpfg_report.csv")

    # Load training tuning R2 values if provided
    tuning_r2_map = _load_tuning_r2_map(run_dir) if run_dir else {}

    rows = []
    for root, _, files in os.walk(run_output_dir):
        if "opt_result.csv" not in files:
            continue

        result_path = os.path.join(root, "opt_result.csv")
        rel_path = os.path.relpath(root, run_output_dir)
        parts = rel_path.split(os.sep)
        model_name = parts[-2] if len(parts) >= 2 else os.path.basename(root)
        replicate = parts[-1].replace("Rep_", "") if parts else ""
        plant_name = parts[0] if len(parts) >= 5 else ""
        dataset_name = parts[1] if len(parts) >= 5 else ""
        optimizer_run = parts[2] if len(parts) >= 5 else ""

        record = _extract_optimizer_record(result_path)
        if record is None:
            continue

        row = {
            "model_name": model_name,
            "replicate": replicate,
            "plant_name": plant_name,
            "dataset_name": dataset_name,
            "optimizer_run": optimizer_run,
            "relative_path": rel_path,
            "selection_basis": "optimized_true" if record["optimized_true"] else "predicted",
        }
        # Attach any available tuning R2
        r2_val = tuning_r2_map.get(model_name, {}).get(str(replicate), None)
        row["best_val_r2"] = r2_val if r2_val is not None else float("nan")
        row.update(record)
        rows.append(row)

    if not rows:
        logging.warning("No optimizer result files found for summary generation.")
        return

    rows.sort(key=lambda item: (item.get("model_name", ""), item.get("replicate", "")))

    detail_fieldnames = [
        "model_name",
        "replicate",
        "plant_name",
        "dataset_name",
        "optimizer_run",
        "relative_path",
        "selection_basis",
        "best_pred_lpfg_cost",
        "best_lpfg_cost_optimized_true",
        "best_val_r2",
        "optimized_true",
    ] + [f"best_{name}" for name in LSYSTEM_PARAM_NAMES]

    with open(summary_results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in detail_fieldnames})

    grouped = {}
    for row in rows:
        grouped.setdefault(row.get("model_name", "unknown"), []).append(row)

    def rank_row(row):
        verified_cost = row.get("best_lpfg_cost_optimized_true", float("nan"))
        pred_cost = row.get("best_pred_lpfg_cost", float("nan"))
        if np.isfinite(verified_cost):
            return (0, verified_cost, pred_cost)
        return (1, pred_cost, verified_cost)

    summary_model_fieldnames = [
        "model_name",
        "n_runs",
        "n_verified",
        "best_replicate",
        "selection_basis",
        "best_pred_lpfg_cost",
        "best_lpfg_cost_optimized_true",
        "best_val_r2",
        "optimized_true",
    ] + [f"best_{name}" for name in LSYSTEM_PARAM_NAMES]

    with open(summary_by_model_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_model_fieldnames)
        writer.writeheader()
        for model_name, model_rows in sorted(grouped.items()):
            best_row = min(model_rows, key=rank_row)
            writer.writerow({
                "model_name": model_name,
                "n_runs": len(model_rows),
                "n_verified": sum(1 for row in model_rows if np.isfinite(row.get("best_lpfg_cost_optimized_true", float("nan")))),
                "best_replicate": best_row.get("replicate", ""),
                "selection_basis": best_row.get("selection_basis", ""),
                "best_pred_lpfg_cost": best_row.get("best_pred_lpfg_cost", ""),
                "best_lpfg_cost_optimized_true": best_row.get("best_lpfg_cost_optimized_true", ""),
                "best_val_r2": best_row.get("best_val_r2", ""),
                "optimized_true": best_row.get("optimized_true", ""),
                **{f"best_{name}": best_row.get(f"best_{name}", "") for name in LSYSTEM_PARAM_NAMES},
            })

    ranked_rows = sorted(rows, key=rank_row)
    best_report_fields = [
        "rank",
        "plant_name",
        "dataset_name",
        "model_name",
        "replicate",
        "selection_basis",
        "best_lpfg_cost_optimized_true",
        "best_pred_lpfg_cost",
        "best_val_r2",
        "optimized_true",
        "relative_path",
    ]
    with open(best_lpfg_report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=best_report_fields)
        writer.writeheader()
        for idx, row in enumerate(ranked_rows, start=1):
            writer.writerow({
                "rank": idx,
                "plant_name": row.get("plant_name", ""),
                "dataset_name": row.get("dataset_name", ""),
                "model_name": row.get("model_name", ""),
                "replicate": row.get("replicate", ""),
                "selection_basis": row.get("selection_basis", ""),
                "best_lpfg_cost_optimized_true": row.get("best_lpfg_cost_optimized_true", ""),
                "best_pred_lpfg_cost": row.get("best_pred_lpfg_cost", ""),
                "best_val_r2": row.get("best_val_r2", ""),
                "optimized_true": row.get("optimized_true", ""),
                "relative_path": row.get("relative_path", ""),
            })

    logging.info(f"Summary saved to {summary_results_path}")
    logging.info(f"Model summary saved to {summary_by_model_path}")
    logging.info(f"Best LPFG report saved to {best_lpfg_report_path}")

def optimize_params_for_model(model, model_type, real_bp_batch, real_ep_batch, args):
    """
    Runs the optimization loop to find parameters.
    Returns: best_params (Tensor), best_predicted_cost (float)
    """
    best_cost = float('inf')
    best_params = None
    restart_records = []
    
    logging.info(
        f"Starting optimization for {model_type} "
        f"({args.restarts} restarts, {args.steps} steps, lr={args.lr})..."
    )

    param_min, param_max, bounds_from_model = derive_parameter_bounds_from_model(model)
    if bounds_from_model:
        logging.info("  Parameter bounds derived from surrogate normalization stats (mean ± %.1f std).", DEFAULT_BOUND_SIGMA)
    else:
        logging.info("  Parameter bounds fallback to hard-coded ranges.")
    
    for restart in range(args.restarts):
        # Create fresh generator for each restart
        opt_net = OptimizerNet(param_min=param_min, param_max=param_max).to(real_bp_batch.device) 
        optimizer = optim.Adam(opt_net.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(50, args.steps // 4),
            gamma=0.8,
        )
        
        restart_best_cost = float('inf')
        restart_best_params = None
        
        # Gradient Descent Loop
        for step in range(args.steps):
            optimizer.zero_grad()
            latent = torch.rand(1, 1).to(real_bp_batch.device)
            pred_params = opt_net(latent)

            # Small jitter helps avoid getting stuck on flat local regions.
            if args.param_jitter_std > 0:
                pred_params = pred_params + torch.randn_like(pred_params) * args.param_jitter_std
                pmin = opt_net.param_min.to(pred_params.device)
                pmax = opt_net.param_max.to(pred_params.device)
                pred_params = torch.max(torch.min(pred_params, pmax), pmin)
            
            # Predict Cost using Surrogate Model
            if model_type == "mlp":
                pred_cost = model(pred_params)
            else:
                pred_cost = model(pred_params, real_bp_batch, real_ep_batch)
            
            # We want to minimize predicted cost
            loss = pred_cost.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track best within this restart
            if loss.item() < restart_best_cost:
                restart_best_cost = loss.item()
                restart_best_params = pred_params.detach()

            if ((step + 1) % args.log_every == 0) or (step == 0) or (step + 1 == args.steps):
                logging.info(
                    f"  Restart {restart+1}/{args.restarts} | "
                    f"Step {step+1}/{args.steps} | "
                    f"surrogate={loss.item():.4f} | "
                    f"restart_best={restart_best_cost:.4f} | "
                    f"global_best={best_cost:.4f}"
                )
                
        # Compare with global best
        if restart_best_cost < best_cost:
            best_cost = restart_best_cost
            best_params = restart_best_params.squeeze()
            logging.info(f"  New Global Best: {best_cost:.4f} (Restart {restart+1}/{args.restarts})")

        if restart_best_params is not None:
            restart_records.append({
                "restart": restart + 1,
                "surrogate_cost": float(restart_best_cost),
                "params": restart_best_params.squeeze().detach().cpu(),
            })

        logging.info(
            f"  Restart {restart+1}/{args.restarts} finished | "
            f"best surrogate cost={restart_best_cost:.4f}"
        )

    return best_params, best_cost, restart_records

def _verify_worker_task(task_args):
    rec, model_output_dir, real_bp, real_ep = task_args
    restart_idx = rec["restart"]
    restart_params = rec["params"]
    restart_surrogate = rec["surrogate_cost"]
    restart_verify_dir = os.path.join(model_output_dir, "Verify", f"Restart_{restart_idx}")
    syn_bp, syn_ep = run_simulation_verification(restart_params, restart_verify_dir)
    if syn_bp:
        try:
            restart_real_cost = evaluate_real_cost(syn_bp, syn_ep, real_bp, real_ep)
            return restart_idx, True, restart_surrogate, restart_real_cost, restart_params
        except Exception as e:
            return restart_idx, False, restart_surrogate, float('inf'), restart_params
    else:
        return restart_idx, False, restart_surrogate, float('inf'), restart_params

def main():
    parser = argparse.ArgumentParser(description="Optimize Plant Parameters using Surrogate Models")
    parser.add_argument(
        "--plant",
        default="Plant_063-32",
        help="Target real plant name residing in Real Plants folder",
    )
    parser.add_argument(
        "--run_dir",
        default=DEFAULT_MODEL_RUN_DIR,
        help="Directory containing surrogate models (searched for best_model.pt)",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=list(DEFAULT_MODEL_FILES),
        help="Specific model file(s) to optimize (can be repeated)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_FAMILIES),
        help="Model families to include when searching run_dir (baseline, hungarian, sinkhorn)",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save optimization results",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=DEFAULT_RESTARTS,
        help="Number of random restarts per model",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Optimization steps per restart",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for the parameter generator",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=DEFAULT_DRY_RUN,
        help="Skip LPFG verification simulation",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Log surrogate optimization progress every N steps",
    )
    parser.add_argument(
        "--verify_each_restart",
        dest="verify_each_restart",
        action="store_true",
        default=DEFAULT_VERIFY_EACH_RESTART,
        help="Run LPFG verification for each restart best candidate",
    )
    parser.add_argument(
        "--no_verify_each_restart",
        dest="verify_each_restart",
        action="store_false",
        help="Disable per-restart LPFG verification",
    )
    parser.add_argument(
        "--param_jitter_std",
        type=float,
        default=DEFAULT_PARAM_JITTER_STD,
        help="Small Gaussian jitter added to parameters during optimization",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only aggregate existing optimization outputs into summary CSVs",
    )
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve relative run/output paths against this script location so the
    # defaults work even when the process starts from a different cwd.
    if not os.path.isabs(args.run_dir):
        args.run_dir = os.path.join(script_dir, args.run_dir)

    # Resolve relative output paths against this script location so logs/results
    # are always written to the intended project folders.
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)

    if args.summary_only:
        os.makedirs(args.output_dir, exist_ok=True)
        run_output_dir = args.output_dir
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(run_output_dir, "optimizer_log.txt")),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(f"Summary Output Directory: {run_output_dir}")
        logging.info(f"Optimizer Results Directory: {args.run_dir}")
        build_optimizer_summaries(run_output_dir, args.run_dir)
        return
    
    # Create a run-scoped output folder to match the rest of the project.
    run_output_dir, run_name = create_run_output_dir(args.output_dir)

    # 1. Setup
    log_path = utils_nn.configure_output_file_logging(run_output_dir, run_name)
    setup_logging(run_output_dir)
    logging.info(f"Terminal Log: {log_path}")
    logging.info(f"Output Directory: {run_output_dir}")
    logging.info(f"Model Search Directory: {args.run_dir}")
    
    # 2. Load Real Data (Target)
    utils_nn.real_plant_name = args.plant
    utils_nn.plant_image_path = utils_nn.plant_images_path + args.plant
    
    logging.info(f"Loading real plant data (Target Structure) for {args.plant}...")
    real_bp, real_ep = read_real_plants()
    real_bp_batch, real_ep_batch = utils_nn.prepare_real_plant_batch(real_bp, real_ep)
    
    # 3. Find Models to Optimize
    # If explicit model files are provided via --model, treat that list as
    # authoritative and skip run_dir auto-discovery.
    models_to_process = []
    if args.model:
        for m in args.model:
            if os.path.exists(m):
                models_to_process.append(m)
            else:
                logging.warning(f"Explicit model path not found (skipping): {m}")
    elif args.run_dir and os.path.exists(args.run_dir):
        selected_families = {m.strip().lower() for m in args.models}
        unknown_requested = sorted(selected_families - SUPPORTED_MODEL_FAMILIES)
        if unknown_requested:
            logging.warning(f"Ignoring unsupported model filters: {unknown_requested}")
        selected_families = selected_families & SUPPORTED_MODEL_FAMILIES
        if "sinkhorn" in selected_families:
            selected_families |= SINKHORN_ABLATION_FAMILIES

        for root, _, files in os.walk(args.run_dir):
            if "final_model.pt" in files:
                model_path = os.path.join(root, "final_model.pt")
            elif "best_model.pt" in files:
                model_path = os.path.join(root, "best_model.pt")
            else:
                continue

            rel_path = os.path.relpath(model_path, args.run_dir)
            rel_parts = [part.lower() for part in Path(rel_path).parts]
            model_family = None
            for part in rel_parts:
                if part in SUPPORTED_MODEL_FAMILIES:
                    model_family = normalize_model_family(part)
                    break

            if model_family is None:
                model_family = extract_model_family(model_path)

            if model_family in selected_families:
                models_to_process.append(model_path)
    else:
        logging.warning(f"Model search directory not found: {args.run_dir}")
                
    models_to_process = list(set(models_to_process))
    
    if not models_to_process:
        logging.warning("No models found! Process finished.")
        return

    # 4. Optimization Loop
    for model_path in models_to_process:
        logging.info(f"--- Processing Model: {model_path} ---")
        model, mtype = load_model(model_path)
        if not model:
            continue

        model_output_dir = resolve_model_output_dir(run_output_dir, model_path, args.run_dir)
        os.makedirs(model_output_dir, exist_ok=True)
            
        # A. Find Optimal Parameters (via Surrogate)
        best_params, surrogate_cost, restart_records = optimize_params_for_model(
            model,
            mtype,
            real_bp_batch,
            real_ep_batch,
            args,
        )

        verified_restart_rows = []
        if not args.dry_run and args.verify_each_restart and restart_records:
            logging.info("  Running per-restart LPFG verification...")
            
            task_list = [(rec, model_output_dir, real_bp, real_ep) for rec in restart_records]
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for idx, success, surrogate_cost_res, real_cost, params in executor.map(_verify_worker_task, task_list):
                    if success:
                        verified_restart_rows.append((idx, surrogate_cost_res, real_cost, params))
                        logging.info(
                            f"  Restart {idx}/{args.restarts} | "
                            f"surrogate={surrogate_cost_res:.4f} | actual={real_cost:.4f}"
                        )
                    else:
                        logging.warning(
                            f"  Restart {idx}/{args.restarts} | "
                            f"surrogate={surrogate_cost_res:.4f} | actual=FAILED"
                        )

            if verified_restart_rows:
                best_by_real = min(verified_restart_rows, key=lambda row: row[2])
                best_params = best_by_real[3]
                surrogate_cost = best_by_real[1]
                logging.info(
                    f"  Selected by real cost from restarts: "
                    f"restart={best_by_real[0]} | surrogate={best_by_real[1]:.4f} | actual={best_by_real[2]:.4f}"
                )

        if restart_records:
            restart_rows = {rec["restart"]: {
                "restart": rec["restart"],
                "surrogate_pred_cost": rec["surrogate_cost"],
                "verified_sim_cost": float("nan"),
                "params": rec["params"],
            } for rec in restart_records}

            for idx, surrogate_cost_res, real_cost, params in verified_restart_rows:
                restart_rows[idx] = {
                    "restart": idx,
                    "surrogate_pred_cost": surrogate_cost_res,
                    "verified_sim_cost": real_cost,
                    "params": params,
                }

            restart_results_path = os.path.join(model_output_dir, "restart_results.csv")
            with open(restart_results_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["restart", "surrogate_pred_cost", "verified_sim_cost"] + [f"opt_{name}" for name in LSYSTEM_PARAM_NAMES]
                writer.writerow(header)
                for restart_idx in sorted(restart_rows):
                    row = restart_rows[restart_idx]
                    param_values = row["params"].detach().cpu().view(-1).tolist()
                    writer.writerow([
                        row["restart"],
                        row["surrogate_pred_cost"],
                        row["verified_sim_cost"],
                        *param_values,
                    ])
            logging.info(f"Restart results saved to {restart_results_path}")
        
        # B. Verify Result (via Simulation)
        real_cost = float('nan')
        if not args.dry_run and best_params is not None:
             logging.info("  Verifying with actual LPFG simulation...")
             verify_dir = os.path.join(model_output_dir, "Verify")
             syn_bp, syn_ep = run_simulation_verification(best_params, verify_dir)
             
             if syn_bp:
                 real_cost = evaluate_real_cost(syn_bp, syn_ep, real_bp, real_ep)
                 logging.info(f"  Simulation Verification Cost: {real_cost:.4f}")
             else:
                 logging.warning("  Simulation verification failed (LPFG/Project error).")
        
        # C. Save Results
        if best_params is not None:
            out_file = os.path.join(model_output_dir, "opt_result.csv")
            
            with open(out_file, "w", newline="") as f:
                writer = csv.writer(f)
                legacy_headers = [f"param_{i}" for i in range(len(best_params))]
                named_headers = [f"opt_{name}" for name in LSYSTEM_PARAM_NAMES]
                header = ["surrogate_pred_cost", "verified_sim_cost"] + legacy_headers + named_headers
                writer.writerow(header)
                param_values = best_params.detach().cpu().view(-1).tolist()
                row = [surrogate_cost, real_cost] + param_values + param_values
                writer.writerow(row)
            logging.info(f"Results saved to {out_file}")

            cleanup_empty_verify_dirs(model_output_dir)

    build_optimizer_summaries(run_output_dir, args.run_dir)


if __name__ == "__main__":
    main()
