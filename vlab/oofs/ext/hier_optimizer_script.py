"""
Hierarchical Optimizer Script.
Optimizes plant parameters to minimize surrogate model predicted cost.
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
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path

# Ensure local modules are in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils_nn import read_real_plants, calculate_cost, build_parameter_file, read_syn_plant
    import model_mlp as benchmark_model
    import model_hungarian as baseline_model
    import model_sinkhorn as sinkhorn_model
except ImportError:
    print("Error: Could not import project modules. Ensure they are in the python path.")
    sys.exit(1)

# --- Configuration Constants ---
DEFAULT_OUTPUT_DIR = "Optimizer Data"
DEFAULT_RESTARTS = 10
DEFAULT_STEPS = 1000
DEFAULT_LR = 0.01

# Parameter Constraints (Min, Max)
# Based on [max_phytomers, plastochron, roll, down, branch, leaf_len, exp_wid, leaf_wid, bend, twist, node, int_wid, exp_rad]
PARAM_MIN = torch.tensor([8.0, 2.8, -110.0, -4.0, 125.0, 3.0, 0.48, 0.8, 80.0, 170.0, 0.6, 0.88, 0.48])
PARAM_MAX = torch.tensor([12.0, 3.2, 110.0, 4.0, 145.0, 7.0, 0.52, 1.2, 100.0, 190.0, 0.8, 0.92, 0.52])

class OptimizerNet(nn.Module):
    """
    A simple neural network that learns to output optimal parameters.
    Used as a trainable generator for gradient descent optimization.
    """
    def __init__(self, input_dim=1, output_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid() 
        )
        self.register_buffer('param_min', PARAM_MIN)
        self.register_buffer('param_max', PARAM_MAX)

    def forward(self, x):
        # Output in [0, 1] range via Sigmoid
        out = self.net(x)
        # Scale to parameter range [min, max]
        return out * (self.param_max - self.param_min) + self.param_min

def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "optimizer_log.txt")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def prepare_real_plant_batch(real_bp, real_ep, max_points=50, device='cpu'):
    """Convert real plant lists to padded tensors for batch processing."""
    num_days = len(real_bp)
    bp_batch = torch.zeros(1, num_days, max_points, 2, device=device)
    ep_batch = torch.zeros(1, num_days, max_points, 2, device=device)
    
    for day in range(num_days):
        if len(real_bp[day]) > 0:
            count = min(len(real_bp[day]), max_points)
            bp_batch[0, day, :count, :] = torch.tensor(real_bp[day][:count], dtype=torch.float32)
        if len(real_ep[day]) > 0:
            count = min(len(real_ep[day]), max_points)
            ep_batch[0, day, :count, :] = torch.tensor(real_ep[day][:count], dtype=torch.float32)
            
    return bp_batch, ep_batch

def load_model(model_path):
    """Loads a surrogate model from a .pt file."""
    path_str = str(model_path)
    model_name = os.path.basename(path_str).lower()
    
    ModelClass = None
    model_type = "unknown"

    # Identify by filename
    if "sinkhorn" in model_name:
        logging.info(f"Loading {model_name} (Type: Sinkhorn Hierarchical)")
        ModelClass = sinkhorn_model.HierarchicalPlantSurrogateNet
        model_type = "sinkhorn"
    elif "mlp" in model_name or "benchmark" in model_name:
        logging.info(f"Loading {model_name} (Type: Benchmark MLP)")
        ModelClass = benchmark_model.BenchmarkSurrogateNet
        model_type = "mlp"
    elif "hungarian" in model_name or "baseline" in model_name or "scheduler" in model_name: # Scheduler implies hierarchical usually
        logging.info(f"Loading {model_name} (Type: Hungarian Hierarchical)")
        ModelClass = baseline_model.HierarchicalPlantSurrogateNet
        model_type = "hungarian"
    else:
        # Fallback default if unsure, or error
        logging.warning(f"Unknown model type for {model_name}. Assuming Hungarian Hierarchical.")
        ModelClass = baseline_model.HierarchicalPlantSurrogateNet
        model_type = "hungarian"

    if ModelClass is None:
        logging.error(f"Could not determine model class for {model_name}")
        return None, None

    model = ModelClass()
    try:
        # Load weights
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model, model_type
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        return None, None

def run_simulation(param_file):
    """Executes LPFG simulation and Project analysis to get real cost."""
    output_dir = "data/surrogate_eval"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. LPFG (Generate Structure)
    lpfg_cmd = [
        "lpfg", "-w", "306", "256",
        "lsystem/lsystem.l", "lsystem/view.v", "lsystem/materials.mat",
        "lsystem/contours.cset", "lsystem/functions.fset", "lsystem/functions.tset",
        param_file
    ]
    
    try:
        subprocess.run(lpfg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("LPFG execution failed. Ensure 'lpfg' is in PATH/vlab/bin.")
        return None, None
        
    # 2. Project (Extract geometry)
    # Generates leafposition.dat
    if not os.path.exists("leafposition.dat"):
        # logging.error("LPFG did not generate leafposition.dat.") 
        # Only error if LPFG succeeded but no file.
        return None, None
        
    # Compile 'project' if missing? (User script implied this)
    if not os.path.exists("./project"):
        logging.info("Compiling 'project' tool...")
        os.system("g++ -o project -Wall -Wextra lsystem/project.cpp -lm")
        
    output_txt = os.path.join(output_dir, "output.txt")
    cmd = f"./project 2454 2056 leafposition.dat > {output_txt}"
    os.system(cmd)
    
    try:
        syn_bp, syn_ep = read_syn_plant(output_txt)
        return syn_bp, syn_ep
    except Exception:
        return None, None

def evaluate_real_cost(syn_bp, syn_ep, real_bp, real_ep):
    """Calculates cost between synthetic and real plant structures."""
    if not syn_bp:
        return float('inf')
        
    total_cost = 0.0
    num_days = min(len(syn_bp), len(real_bp))
    
    if num_days == 0:
        return float('inf')
        
    for i in range(num_days):
        total_cost += calculate_cost(syn_bp[i], syn_ep[i], real_bp[i], real_ep[i])
        
    return total_cost

def optimize_model(model, model_type, real_bp_batch, real_ep_batch, args):
    """Runs gradient descent optimization on model input parameters."""
    best_cost = float('inf')
    best_params = None
    
    logging.info(f"Starting optimization ({args.restarts} restarts, {args.steps} steps)...")
    
    for restart in range(args.restarts):
        opt_net = OptimizerNet().to(real_bp_batch.device) # Ensure device match if gpu used
        optimizer = optim.Adam(opt_net.parameters(), lr=args.lr)
        dummy_input = torch.randn(1, 1).to(real_bp_batch.device)
        
        # Optimization Loop
        for _ in range(args.steps):
            optimizer.zero_grad()
            pred_params = opt_net(dummy_input)
            
            # Predict Cost
            # Note: Surrogate models expect (params, real_bp, real_ep)
            pred_cost = model(pred_params, real_bp_batch, real_ep_batch)
                
            loss = pred_cost.mean()
            loss.backward()
            optimizer.step()
            
        # Evaluate Final Parameters of this restart
        with torch.no_grad():
            final_params = opt_net(dummy_input)
            final_cost = model(final_params, real_bp_batch, real_ep_batch).item()
            
            if final_cost < best_cost:
                best_cost = final_cost
                best_params = final_params.squeeze()
                logging.info(f"  New Best (Surrogate): {best_cost:.4f} [Restart {restart}]")

    return best_params, best_cost

def main():
    parser = argparse.ArgumentParser(description="Optimize Plant Parameters")
    parser.add_argument("--run_dir", help="Directory containing surrogate models")
    parser.add_argument("--model", action="append", help="Specific model file(s)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--dry_run", action="store_true", help="Skip LPFG verification")
    args = parser.parse_args()
    
    # Setup Output
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    
    # 1. Load Real Data
    logging.info("Loading real plant data...")
    real_bp, real_ep = read_real_plants()
    real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep)
    
    # 2. Collect Models
    models_to_process = []
    if args.run_dir and os.path.exists(args.run_dir):
        for root, _, files in os.walk(args.run_dir):
            if "best_model.pt" in files:
                models_to_process.append(os.path.join(root, "best_model.pt"))
                
    if args.model:
        for m in args.model:
            if os.path.exists(m):
                models_to_process.append(m)
                
    models_to_process = list(set(models_to_process))
    
    if not models_to_process:
        logging.warning("No models found. Use --run_dir or --model.")
        return

    # 3. Optimize Each Model
    for model_path in models_to_process:
        logging.info(f" Processing model: {model_path}")
        model, mtype = load_model(model_path)
        if not model:
            continue
            
        # A. Optimize Surrogate
        best_params, surrogate_cost = optimize_model(model, mtype, real_bp_batch, real_ep_batch, args)
        
        # B. Verify with LPFG simulation
        real_cost = float('inf')
        if not args.dry_run and best_params is not None:
             logging.info("  Verifying with LPFG simulation...")
             params_list = best_params.tolist()
             
             # Create temp parameter file
             temp_file = f"temp_opt_{uuid.uuid4().hex[:6]}.vset"
             build_parameter_file(temp_file, params_list)
             
             # Run Sim
             syn_bp, syn_ep = run_simulation(temp_file)
             if syn_bp:
                 real_cost = evaluate_real_cost(syn_bp, syn_ep, real_bp, real_ep)
                 logging.info(f"  Real Simulation Cost: {real_cost:.4f}")
             else:
                 logging.warning("  Simulation failed.")
                 
             if os.path.exists(temp_file):
                 os.remove(temp_file)
        
        # C. Save Results
        if best_params is not None:
            model_name = os.path.basename(os.path.dirname(model_path))
            run_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
            out_file = os.path.join(args.output_dir, f"opt_{run_name}_{model_name}.csv")
            
            with open(out_file, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["surrogate_cost", "real_sim_cost"] + [f"param_{i}" for i in range(len(best_params))]
                writer.writerow(header)
                row = [surrogate_cost, real_cost] + best_params.tolist()
                writer.writerow(row)
            logging.info(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
