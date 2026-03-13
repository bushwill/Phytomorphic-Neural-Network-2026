"""
Optimizer Script for Plant Parameters using Surrogate Models.
Optimizes input parameters to minimize predicted cost against real plant data.
Supports both Baseline MLP and Sinkhorn Transformer models.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import csv
import time
import subprocess
import shutil
from datetime import datetime
from plant_comparison_nn import read_real_plants, calculate_cost
from utils_nn import build_parameter_file
import surrogate_nn_dataset as baseline_model
import surrogate_nn_dataset_sinkhorn as sinkhorn_model

# ==========================================
#              USER CONFIGURATION 
# ==========================================

# 1. Select Operation Mode
# Set which paths to use for optimization.
# You can uncomment or add paths to these lists.

# Option A: Process entire run folders (finds all best_model.pt inside)
TARGET_RUN_DIRS = [
    # "Training Data/Run_030426",
    # "Training Data/Run_030426_1",
]

# Option B: Process specific model files
TARGET_MODEL_PATHS = [
    "Training Data/Run_030426_1/baseline_mlp/best_model.pt",
    "Training Data/Run_030426_1/sinkhorn_transformer/best_model.pt",
]

# 2. Optimization Settings
OUTPUT_DIR = "Optimizer Data" # Base directory for optimizer runs
NUM_RESTARTS = 10       # Number of times to restart optimization to avoid local minima
NUM_STEPS = 1000        # Gradient descent steps per restart
LEARNING_RATE = 0.01    # Learning rate for the optimizer

# 3. Parameter Constraints
PARAM_MIN = torch.tensor([8.0, 2.8, -110.0, -4.0, 125.0, 3.0, 0.48, 0.8, 80.0, 170.0, 0.6, 0.88, 0.48])
PARAM_MAX = torch.tensor([12.0, 3.2, 110.0, 4.0, 145.0, 7.0, 0.52, 1.2, 100.0, 190.0, 0.8, 0.92, 0.52])

# ==========================================

def get_parser():
    parser = argparse.ArgumentParser(description="Optimize plant parameters using surrogate models.")
    parser.add_argument("--run_dir", type=str, help="Path to the training run directory (e.g., Data/Run_030426_1)")
    parser.add_argument("--model_path", type=str, help="Path to a specific model file (.pt)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save optimization results")
    parser.add_argument("--restarts", type=int, default=NUM_RESTARTS, help="Number of optimization restarts")
    parser.add_argument("--steps", type=int, default=NUM_STEPS, help="Number of optimization steps per restart")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate for optimizer")
    return parser

class OptimizerNet(nn.Module):
    """Simple generator network to output optimal parameters"""
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
        out = self.net(x)
        # Scale to parameter range
        return out * (self.param_max - self.param_min) + self.param_min

def prepare_real_plant_batch(real_bp, real_ep, max_points=50, device='cpu'):
    """Convert real plant data to fixed-size tensors"""
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
    """Detects model type and loads appropriate class"""
    model_path = os.path.abspath(model_path)
    # Check absolute path first
    
    # Determine type from path or parent folder name
    parent_folder = os.path.basename(os.path.dirname(model_path))
    
    if "sinkhorn" in parent_folder.lower() or "sinkhorn" in model_path.lower():
        print(f"Loading {os.path.basename(model_path)} (Type: Sinkhorn Transformer)")
        ModelClass = sinkhorn_model.HierarchicalPlantSurrogateNet
        model_type = "sinkhorn"
    elif "baseline" in parent_folder.lower() or "baseline" in model_path.lower():
        print(f"Loading {os.path.basename(model_path)} (Type: Baseline MLP)")
        ModelClass = baseline_model.HierarchicalPlantSurrogateNet
        model_type = "baseline"
    else:
        # Default fallback
        print(f"Loading {os.path.basename(model_path)} (Type: Unknown, assuming Baseline MLP)")
        ModelClass = baseline_model.HierarchicalPlantSurrogateNet
        model_type = "baseline"

    model = ModelClass()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        print(f"Error loading state dict for {model_path}: {e}")
        return None, None
        
    model.eval()
    return model, model_type


def read_syn_plant_surrogate(file_name):
    try:
        with open(file_name, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading synthetic plant output: {e}")
        return [], []
        
    day_temp = 0
    syn_bp = []
    syn_ep = []
    syn_bp_day = []
    syn_ep_day = []
    day = []

    for line in lines:
        temp = line.split(" ")
        if len(temp) < 2: continue
        
        if temp[0] == "Day:":
            try:
                day_temp = int(temp[1])
            except ValueError:
                continue
                
            if day_temp > 2:
                syn_bp.append(syn_bp_day)
                syn_ep.append(syn_ep_day)
                syn_bp_day = []
                syn_ep_day = []
                
        if (temp[0] != "Day:") and (day_temp > 1):
            if len(temp) >= 4:
                try:
                    if temp[0] == "I":
                        syn_bp_day.append([int(temp[3]), int(temp[2])])
                        day.append(day_temp)
                    else:
                        syn_ep_day.append([int(temp[3]), int(temp[2])])
                        day.append(day_temp)
                except ValueError:
                    continue

    if day_temp == 27:
        syn_bp.append(syn_bp_day)
        syn_ep.append(syn_ep_day)

    return syn_bp, syn_ep

def generate_and_evaluate(param_file, real_bp, real_ep):
    """
    Runs the actual L-system simulation (lpfg) to generate a plant and calculates the real cost.
    Requires 'lpfg' and 'project' executables to be in the path or current directory.
    """
    output_dir = "data/surrogate"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Run LPFG (L-System Generator)
    # Note: Lsystem files are in 'lsystem/' folder relative to execution root
    # param_file should be absolute or relative to execution root
    lpfg_cmd = [
        "lpfg",
        "-w", "306", "256",
        "lsystem/lsystem.l",
        "lsystem/view.v",
        "lsystem/materials.mat",
        "lsystem/contours.cset",
        "lsystem/functions.fset",
        "lsystem/functions.tset",
        param_file
    ]
    
    log_file = os.path.join(output_dir, "lpfg_log.txt")
    with open(log_file, "w") as f_log:
        try:
            subprocess.run(lpfg_cmd, stdout=f_log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as e:
            print(f"LPFG execution failed. Check if 'lpfg' is installed/reachable. Error: {e}")
            return float('inf')
        except FileNotFoundError:
             print("LPFG executable not found. Please ensure 'lpfg' is in your PATH.")
             return float('inf')

    # 2. Run Project (Endpoint/Branchpoint Extractor)
    # Reads 'leafposition.dat' which lpfg generates in current directory
    if not os.path.exists("project"):
        print("Compiling project executable...")
        os.system("g++ -o project -Wall -Wextra lsystem/project.cpp -lm")
    
    if os.path.exists("leafposition.dat"):
        output_txt = os.path.join(output_dir, "output.txt")
        # Ensure 'project' is executable
        if not os.access("./project", os.X_OK):
             os.chmod("./project", 0o755)
             
        cmd = f"./project 2454 2056 leafposition.dat > {output_txt}"
        os.system(cmd)
        
        # Move leafposition.dat to data folder for record keeping
        shutil.move("leafposition.dat", os.path.join(output_dir, "leafposition.dat"))
        
        # 3. Read Output and Calculate Cost
        syn_bp, syn_ep = read_syn_plant_surrogate(output_txt)
        
        total_cost = 0.0
        # Determine strictness of comparison (min length of both)
        num_days = min(len(syn_bp), len(real_bp))
        
        if num_days == 0:
            return float('inf') # Failed generation
            
        for i in range(num_days):
            # calculate_cost takes (day_syn_bp, day_syn_ep, real_bp, real_ep)
            # Note: syn_bp[i] contains points for day i
            total_cost += calculate_cost(syn_bp[i], syn_ep[i], real_bp[i], real_ep[i])
            
        return total_cost
        
    else:
        print("Error: leafposition.dat was not generated by lpfg.")
        return float('inf')

def optimize_for_model(model, model_type, real_bp_batch, real_ep_batch, restarts, steps, lr, real_bp, real_ep):
    """Runs the optimization loop for a single model"""
    
    best_cost = float('inf')
    best_params = None
    best_real_cost = float('inf')
    
    print(f"Starting optimization ({restarts} restarts, {steps} steps)...")
    
    for restart in range(restarts):
        # Initialize Optimizer Network
        opt_net = OptimizerNet()
        optimizer = optim.Adam(opt_net.parameters(), lr=lr)
        
        # Batch of inputs for the optimizer net (can be just constant 1s or noise)
        dummy_input = torch.randn(1, 1) 
        
        # start_time = time.time()
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # 1. Generate Parameters
            pred_params = opt_net(dummy_input)
            
            # 2. Evaluate Cost using Surrogate
            if model_type == "sinkhorn":
                # Sinkhorn model might return (assignment, cost) or just cost depending on implementation
                # Based on previous file read, it returns denormalized cost directly in forward()
                predicted_cost = model(pred_params, real_bp_batch, real_ep_batch)
            else:
                predicted_cost = model(pred_params, real_bp_batch, real_ep_batch)
            
            # 3. Loss = Minimize Cost
            loss = predicted_cost.mean()
            
            loss.backward()
            optimizer.step()
            
        # Check result of this restart
        with torch.no_grad():
            final_params = opt_net(dummy_input)
            final_cost = model(final_params, real_bp_batch, real_ep_batch).item()
            
            # Convert tensors to list for file writing
            params_list = final_params.squeeze().tolist()
            
            # Verify with Real L-System Simulation
            print(f"  Restart {restart+1}: Surrogate Cost {final_cost:.4f}. Verifying with LPFG...")
            temp_param_file = f"temp_opt_params_{restart}.vset"
            build_parameter_file(temp_param_file, params_list)
            
            try:
                # generate_and_evaluate needs to run lpfg which produces leafposition.dat
                # We need to be careful about file collisions if running multiple optimizations, 
                # but here we are sequential.
                real_sim_cost = generate_and_evaluate(temp_param_file, real_bp, real_ep)
                print(f"  -> Real L-System Cost: {real_sim_cost:.4f}")
            except Exception as e:
                print(f"  -> Simulation Failed: {e}")
                real_sim_cost = float('inf')
            
            # Cleanup temp file
            if os.path.exists(temp_param_file):
                os.remove(temp_param_file)

            # We track best based on SURROGATE cost (since that's what we optimized)
            # But the user can inspect Real Cost to validate the surrogate.
            # Alternatively, we could save the best *Real* cost, but that defeats the purpose of the surrogate 
            # if we are just doing random restarts and checking real cost. 
            # The surrogate guides the gradient descent. 
            
            if final_cost < best_cost:
                best_cost = final_cost
                best_params = params_list
                best_real_cost = real_sim_cost
                print(f"  ** New Best Surrogate Cost found! **")

    return best_params, best_cost, best_real_cost

def main():
    # Helper: Ensure LPFG is in PATH
    lpfg_path = os.path.expanduser("~/PhytomorphicNN/vlab/bin")
    if lpfg_path not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + lpfg_path
        
    parser = get_parser()
    args = parser.parse_args()
    
    # 1. Load Real Plant Data
    print("Reading real plant structure...")
    real_bp, real_ep = read_real_plants()
    real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep)
    
    # 2. Gather Models to Process (Combine CLI args and Config)
    models_to_process = []
    
    # Configuration lists
    run_dirs = list(TARGET_RUN_DIRS)
    model_paths = list(TARGET_MODEL_PATHS)
    
    # Args override or append? Let's treat them as additional inputs
    if args.run_dir:
        run_dirs.append(args.run_dir)
    if args.model_path:
        model_paths.append(args.model_path)
        
    # Process Directories
    for run_dir in run_dirs:
        if os.path.exists(run_dir):
            for root, dirs, files in os.walk(run_dir):
                if "best_model.pt" in files:
                    models_to_process.append(os.path.join(root, "best_model.pt"))
        else:
            print(f"Warning: Run directory {run_dir} not found.")

    # Process Individual Files
    for mp in model_paths:
        if os.path.exists(mp):
            if mp not in models_to_process:
                models_to_process.append(mp)
        else:
            print(f"Warning: Model file {mp} not found.")
            
    # Remove duplicates
    models_to_process = list(set(models_to_process))

    if not models_to_process:
        print("No models found to process. Please check USER CONFIGURATION or command line arguments.")
        parser.print_help()
        return

    # 3. Process Each Model
    
    # Determine output directory
    if args.output_dir == OUTPUT_DIR:
        # Default behavior: create a new timestamped run folder
        date_str = datetime.now().strftime("%m%d%y")
        run_base_name = f"Run_{date_str}"
        output_dir = os.path.join(OUTPUT_DIR, run_base_name)
        
        # Incremental naming to avoid collision
        if os.path.exists(output_dir):
            i = 1
            while os.path.exists(f"{output_dir}_{i}"):
                i += 1
            output_dir = f"{output_dir}_{i}"
    else:
        # User specified directory
        output_dir = args.output_dir
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving optimizer results to: {output_dir}")
    
    print(f"\nFound {len(models_to_process)} models to optimize.")
    
    for model_path in models_to_process:
        print("\n" + "=" * 60)
        model, model_type = load_model(model_path)
        if model is None:
            continue
            
        restarts = args.restarts if args.restarts != 10 else NUM_RESTARTS
        steps = args.steps if args.steps != 1000 else NUM_STEPS
        lr = args.lr if args.lr != 0.01 else LEARNING_RATE
            
        best_params, best_surrogate_cost, best_real_cost = optimize_for_model(
            model, model_type, real_bp_batch, real_ep_batch, restarts, steps, lr,
            real_bp, real_ep
        )
        
        if best_params is not None:
            # Save Results
            model_name = os.path.basename(os.path.dirname(model_path)) # e.g. baseline_mlp
            # run_name is used in filename to identify source model run, not output run
            run_name = os.path.basename(os.path.dirname(os.path.dirname(model_path))) # e.g. Run_030426
            
            output_csv = os.path.join(output_dir, f"optimized_{run_name}_{model_name}.csv")
            
            with open(output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["predicted_min_cost", "real_sim_cost"] + [f"param_{i}" for i in range(len(best_params))]
                writer.writerow(header)
                row = [best_surrogate_cost, best_real_cost] + list(best_params)
                writer.writerow(row)
                
            print(f"Optimization Complete for {model_name}.")
            print(f"  Predicted Cost: {best_surrogate_cost:.4f}")
            print(f"  Real Simulation Cost: {best_real_cost:.4f}")
            print(f"Results saved to: {output_csv}")
            # print(f"Best Parameters: {best_params}")
        
    print("\n" + "=" * 60)
    print("Done.")

if __name__ == "__main__":
    main()

