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
SUPPORTED_MODEL_FAMILIES = {"baseline", "hungarian", "sinkhorn"}

# Parameter Constraints (Min, Max)
# Corresponds to: [max_phytomers, plastochron, roll, down, branch, leaf_len, exp_wid, leaf_wid, bend, twist, node, int_wid, exp_rad]
# These ranges should match the bounds used during data generation.
PARAM_MIN = torch.tensor([8.0, 2.8, -110.0, -4.0, 125.0, 3.0, 0.48, 0.8, 80.0, 170.0, 0.6, 0.88, 0.48])
PARAM_MAX = torch.tensor([12.0, 3.2, 110.0, 4.0, 145.0, 7.0, 0.52, 1.2, 100.0, 190.0, 0.8, 0.92, 0.52])

class OptimizerNet(nn.Module):
    """
    A lightweight neural network used as a trainable parameter generator.
    
    Why use this?
    Directly optimizing a tensor `params = torch.tensor(..., requires_grad=True)` can sometimes 
    lead to issues with constraints or gradients. Using a small network that outputs 
    the parameters (passed through a Sigmoid to enforce 0-1 range, then scaled) 
    keeps the optimization process stable and allows us to use standard NN optimizers.
    """
    def __init__(self, input_dim=1, output_dim=13):
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
        self.register_buffer('param_min', PARAM_MIN)
        self.register_buffer('param_max', PARAM_MAX)

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
    model_dir = os.path.dirname(os.path.abspath(model_path))
    parent = os.path.basename(model_dir).lower()

    if parent.startswith("rep_"):
        family = os.path.basename(os.path.dirname(model_dir)).lower()
    else:
        family = parent

    return family

def load_model(model_path):
    """
    Identifies and loads a trained surrogate model from a .pt file.
    Returns (model_instance, model_type_string).
    """
    path_str = str(model_path)
    family = extract_model_family(path_str)

    if family == "sinkhorn":
        logging.info(f"Loading {path_str} (Type: Sinkhorn Hierarchical)")
        ModelClass = sinkhorn_model.HierarchicalPlantSurrogateNet
        model_type = "sinkhorn"
    elif family == "baseline":
        logging.info(f"Loading {path_str} (Type: Baseline MLP)")
        ModelClass = benchmark_model.BenchmarkSurrogateNet
        model_type = "mlp"
    elif family == "hungarian":
        logging.info(f"Loading {path_str} (Type: Hungarian Hierarchical)")
        ModelClass = baseline_model.HierarchicalPlantSurrogateNet
        model_type = "hungarian"
    else:
        logging.error(
            f"Could not determine model family from path: {path_str}. "
            f"Expected family folder in {sorted(SUPPORTED_MODEL_FAMILIES)}."
        )
        return None, None

    model = ModelClass()
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

def build_optimizer_summaries(run_output_dir):
    """Collect per-model optimization results into root-level summary CSV files."""
    result_rows = []

    for root, _, files in os.walk(run_output_dir):
        if "opt_result.csv" not in files:
            continue

        result_path = os.path.join(root, "opt_result.csv")
        rel_path = os.path.relpath(root, run_output_dir)
        parts = rel_path.split(os.sep)
        model_family = parts[0] if parts else os.path.basename(root)
        run_variant = os.path.join(*parts[1:]) if len(parts) > 1 else ""

        try:
            with open(result_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                values = next(reader, None)

            if not header or not values:
                continue

            record = dict(zip(header, values))
            row = {
                "model_family": model_family,
                "run_variant": run_variant,
                "relative_path": rel_path,
                "surrogate_pred_cost": float(record.get("surrogate_pred_cost", "nan")),
                "verified_sim_cost": float(record.get("verified_sim_cost", "nan")),
            }

            for key, value in record.items():
                if key.startswith("param_"):
                    row[key] = float(value)

            result_rows.append(row)
        except Exception as e:
            logging.warning(f"Skipping summary read for {result_path}: {e}")

    if not result_rows:
        logging.warning("No optimizer result files found for summary generation.")
        return

    result_rows.sort(key=lambda item: (item["model_family"], item["run_variant"], item["relative_path"]))

    param_keys = sorted([key for key in result_rows[0].keys() if key.startswith("param_")], key=lambda x: int(x.split("_")[1]))
    summary_path = os.path.join(run_output_dir, "summary_results.csv")
    fieldnames = ["model_family", "run_variant", "relative_path", "surrogate_pred_cost", "verified_sim_cost"] + param_keys

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    grouped = {}
    for row in result_rows:
        grouped.setdefault(row["model_family"], []).append(row)

    summary_by_model_path = os.path.join(run_output_dir, "summary_by_model.csv")
    agg_fieldnames = ["model_family", "n_runs", "surrogate_pred_cost_mean", "surrogate_pred_cost_std", "verified_sim_cost_mean", "verified_sim_cost_std"]
    with open(summary_by_model_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        for model_family, rows in sorted(grouped.items()):
            surrogate_vals = np.array([row["surrogate_pred_cost"] for row in rows], dtype=float)
            verified_vals = np.array([row["verified_sim_cost"] for row in rows], dtype=float)
            writer.writerow({
                "model_family": model_family,
                "n_runs": len(rows),
                "surrogate_pred_cost_mean": float(np.mean(surrogate_vals)),
                "surrogate_pred_cost_std": float(np.std(surrogate_vals)),
                "verified_sim_cost_mean": float(np.mean(verified_vals)),
                "verified_sim_cost_std": float(np.std(verified_vals)),
            })

    logging.info(f"Summary saved to {summary_path}")
    logging.info(f"Model summary saved to {summary_by_model_path}")

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
    
    for restart in range(args.restarts):
        # Create fresh generator for each restart
        opt_net = OptimizerNet().to(real_bp_batch.device) 
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
                pmin = PARAM_MIN.to(pred_params.device)
                pmax = PARAM_MAX.to(pred_params.device)
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
    models_to_process = []
    if args.run_dir and os.path.exists(args.run_dir):
        selected_families = {m.strip().lower() for m in args.models}
        unknown_requested = sorted(selected_families - SUPPORTED_MODEL_FAMILIES)
        if unknown_requested:
            logging.warning(f"Ignoring unsupported model filters: {unknown_requested}")
        selected_families = selected_families & SUPPORTED_MODEL_FAMILIES

        for root, _, files in os.walk(args.run_dir):
            if "best_model.pt" in files:
                model_path = os.path.join(root, "best_model.pt")
                model_family = extract_model_family(model_path)
                if model_family in selected_families:
                    models_to_process.append(model_path)
    else:
        logging.warning(f"Model search directory not found: {args.run_dir}")
                
    if args.model:
        for m in args.model:
            if os.path.exists(m):
                models_to_process.append(m)
                
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
                header = ["surrogate_pred_cost", "verified_sim_cost"] + [f"param_{i}" for i in range(len(best_params))]
                writer.writerow(header)
                row = [surrogate_cost, real_cost] + best_params.tolist()
                writer.writerow(row)
            logging.info(f"Results saved to {out_file}")

            cleanup_empty_verify_dirs(model_output_dir)

    build_optimizer_summaries(run_output_dir)


if __name__ == "__main__":
    main()
