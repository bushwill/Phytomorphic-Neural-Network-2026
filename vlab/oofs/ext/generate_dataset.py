import os
import time
import csv
import torch
import shutil
import argparse
import numpy as np
import uuid
import sys
from datetime import datetime
from scipy.stats import qmc, norm

# Set umask to 0 so all created files/dirs are readable/writable by everyone (777/666)
# This allows the host user (pzu426) to modify files created by Docker root user
os.umask(0)

import plant_comparison_nn
from utils_nn import generate_plant, read_syn_plant, build_parameter_file
from plant_comparison_nn import read_real_plants, calculate_cost

# Default Dataset Generation Parameters
USE_LHS = True

DEFAULT_PLANT = "Plant_063-32"   # Target real plant folder name
DEFAULT_TRAIN_SIZE = 5000
DEFAULT_VAL_SIZE = 100
DEFAULT_TEST_SIZE = 1000

# Constants for distributions
# [Min, Max]
# Ranges derived from lsystem/lsystem.pnl where available
PARAM_RANGES = [
    (1, 20),       # max_phytomers (Discrete Uniform 1-20)
    (0.1, 10.0),   # plastochron (1-100 / 10)
    (-180.0, 180.0), # plant_roll_angle (Assumed full range)
    (0.0, 180.0),  # plant_down_angle (Assumed 0-180 for down angle)
    (0.0, 270.0),  # branch_angle (0-270)
    (0.1, 10.0),   # leaf_len (1-100 / 10)
    (0.01, 1.0),   # exp_leaf_wid (1-100 / 100)
    (0.01, 2.0),   # leaf_wid (1-200 / 100)
    (0.0, 180.0),  # leaf_bend_scale (Assumed range)
    (0.0, 360.0),  # leaf_twist_scale (Assumed range)
    (0.0, 2.0),    # node_len (IntLen 0-20 / 10. Note: lsystem says 0-20, utils says 0.7. Range 0-2 seems right)
    (0.01, 2.0),   # int_wid (1-200 / 100)
    (0.0, 1.0)     # exp_int_rad (0-100 / 100)
]

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def generate_lhs_samples(n_samples):
    """
    Generate parameter samples using Latin Hypercube Sampling with Uniform Distributions.
    """
    sampler = qmc.LatinHypercube(d=len(PARAM_RANGES))
    sample = sampler.random(n=n_samples)
    
    # Scale samples to parameter ranges
    scaled_samples = qmc.scale(sample, [r[0] for r in PARAM_RANGES], [r[1] for r in PARAM_RANGES])
    
    # Discretize `max_phytomers` (First parameter) to integer
    scaled_samples[:, 0] = np.round(scaled_samples[:, 0])
    
    return scaled_samples

def generate_random_samples(n_samples):
    """
    Generate parameter samples using Random Sampling (Gaussian/Normal distribution centered on heuristics).
    This mimics build_random_parameter_file from utils_nn but returns an array instead of writing file.
    """
    samples = np.zeros((n_samples, len(PARAM_RANGES)))
    
    for i in range(n_samples):
        # Logic from utils_nn.build_random_parameter_file
        # nran(mean, std)
        
        max_phy = norm.rvs(10., 1.)
        plast = norm.rvs(3., 0.1)
        
        # Chirality handling logic from original script
        chirality = 1.
        if np.random.uniform(0., 1.) < 0.5:
            chirality = -1.
            
        plant_roll_angle = norm.rvs(chirality * 90., 10.0)
        plant_down_angle = norm.rvs(0., 4.0)
        
        # Mapping remaining params based on utils_nn
        branch_angle = norm.rvs(135., 5.)
        leaf_len = norm.rvs(5., 1.)
        exp_leaf_wid = norm.rvs(0.5, 0.01)
        leaf_wid = norm.rvs(1., 0.1)
        leaf_bend_scale = norm.rvs(90., 3.)
        leaf_twist_scale = norm.rvs(180., 3.)
        node_len = norm.rvs(0.7, 0.05)
        int_wid = norm.rvs(0.9, 0.01)
        exp_int_rad = norm.rvs(0.5, 0.01)
        
        samples[i] = [
            max_phy, plast, plant_roll_angle, plant_down_angle, branch_angle,
            leaf_len, exp_leaf_wid, leaf_wid, leaf_bend_scale, leaf_twist_scale,
            node_len, int_wid, exp_int_rad
        ]
        
    return samples

def generate_dataset_split(split_name, size, real_bp, real_ep, output_dir, use_lhs=True):
    """
    Generates a dataset split (Train, Val, or Test).
    """
    split_start_time = time.time()
    print(f"\nGenerating {split_name} dataset ({size} samples)...")
    if use_lhs:
        print("Method: Latin Hypercube Sampling (Uniform)")
    else:
        print("Method: Random Sampling (Gaussian)")
    
    # Create directories
    split_dir = os.path.join(output_dir, split_name)
    structures_dir = os.path.join(split_dir, "structures")
    if not os.path.exists(structures_dir):
        os.makedirs(structures_dir)
    
    csv_file = os.path.join(output_dir, f"{split_name}.csv")
    
    # Initialize CSV with header
    header = ["id", "cost"] + [f"param_{i}" for i in range(13)]
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
    # Generate ALL parameters upfront
    if use_lhs:
        all_params = generate_lhs_samples(size)
    else:
        all_params = generate_random_samples(size)
    
    # Verify max_phytomers is not too small (absolute minimum 1)
    all_params[:, 0] = np.maximum(all_params[:, 0], 1.0)
    
    unique_id = uuid.uuid4().hex[:8]
    temp_param_file = f"temp_dataset_params_{split_name}_{unique_id}.vset"
    temp_output_dir = f"temp_dataset_output_{split_name}_{unique_id}"
    
    os.makedirs(temp_output_dir, exist_ok=True)
    
    try:
        valid_samples_count = 0
        
        for i in range(size):
            params = all_params[i]
            
            # 1. Write Parameter File
            build_parameter_file(temp_param_file, params)
            
            # 2. Generate Synthetic Plant (L-system)
            try:
                generate_plant(temp_param_file, temp_output_dir)
                
                # 3. Read the Generated Structure
                output_txt_path = os.path.join(temp_output_dir, "output.txt")
                if not os.path.exists(output_txt_path):
                     continue
                     
                syn_bp, syn_ep = read_syn_plant(output_txt_path)
                
                # 4. Calculate True Cost vs Real Plant
                total_cost = 0.0
                num_days = min(len(syn_bp), len(real_bp))
                
                if num_days == 0:
                    total_cost = 1e6 # Penalty
                else:
                    for day in range(num_days):
                        day_cost = calculate_cost(syn_bp[day], syn_ep[day], real_bp[day], real_ep[day])
                        total_cost += day_cost
                
                # 5. Save Data
                params_list = params.tolist()
                
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    row = [i, total_cost] + params_list
                    writer.writerow(row)
                    
                structure_data = {
                    "bp": syn_bp,
                    "ep": syn_ep,
                    "params": torch.tensor(params, dtype=torch.float32),
                    "cost": torch.tensor(total_cost, dtype=torch.float32)
                }
                torch.save(structure_data, os.path.join(structures_dir, f"structure_{i}.pt"))
                
                valid_samples_count += 1
                if (i+1) % 50 == 0:
                    print(f"  Processed {i+1}/{size} samples.")
                    
            except Exception as e:
                continue
                
            # Cleanup per iteration to keep workspace clean
            if os.path.exists(temp_param_file):
                try:
                    os.remove(temp_param_file)
                except OSError:
                    pass
    finally:
        # Cleanup
        if os.path.exists(temp_param_file):
            try:
                os.remove(temp_param_file)
            except OSError: 
                pass
        if os.path.exists(temp_output_dir):
            try:
                shutil.rmtree(temp_output_dir)
            except OSError:
                pass
                
    split_end_time = time.time()
    split_duration = split_end_time - split_start_time
    print(f"Finished {split_name} dataset. Valid samples: {valid_samples_count}/{size}. Time: {split_duration:.2f}s")

def main():
    
    # Generate output directory based on current date
    current_date = datetime.now().strftime("%m%d%y")
    
    # Auto-incrementing Dataset Folder
    base_name = f"Run {current_date}"
    counter = 0
    candidate_name = base_name
    
    while os.path.exists(os.path.join("Datasets", candidate_name)):
        counter += 1
        candidate_name = f"{base_name}_{counter}"
        
    DEFAULT_OUT = os.path.join("Datasets", candidate_name)

    parser = argparse.ArgumentParser(description="Generate datasets for PhytomorphicNN")
    parser.add_argument("--plant", type=str, default=DEFAULT_PLANT, help=f"Name of the real plant folder (default: {DEFAULT_PLANT})")
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE, help=f"Number of training samples (default: {DEFAULT_TRAIN_SIZE})")
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE, help=f"Number of validation samples (default: {DEFAULT_VAL_SIZE})")
    parser.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE, help=f"Number of test samples (default: {DEFAULT_TEST_SIZE})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT, help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--lhs", action="store_true", default=True, help="Use Latin Hypercube Sampling (default: True)")
    parser.add_argument("--random", action="store_true", help="Use Random (Gaussian) Sampling instead of LHS")
    
    args = parser.parse_args()
    
    # Determine sampling method
    if args.random:
        USE_LHS = False
    elif args.lhs:
        USE_LHS = True
        
    start_time = time.time()
    print(f"Target Real Plant: {args.plant}")
    print(f"Sampling Method: {'Latin Hypercube (Uniform)' if USE_LHS else 'Random (Gaussian)'}")
    
    plant_comparison_nn.real_plant_name = args.plant
    plant_comparison_nn.plant_image_path = os.path.join(plant_comparison_nn.plant_images_path, args.plant)
    
    print("Reading real plant data...")
    try:
        real_bp, real_ep = read_real_plants()
        print(f"Successfully loaded real plant data ({len(real_bp)} days).")
    except Exception as e:
        print(f"Failed to read real plant data: {e}")
        return

    # Create main output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    log_file = os.path.join(args.output_dir, "dataset_generation_log.txt")
    sys.stdout = DualLogger(log_file)
    print(f"Logging to {log_file}")
        
    # Generate Splits
    if args.train_size > 0:
        generate_dataset_split("Train", args.train_size, real_bp, real_ep, args.output_dir, use_lhs=USE_LHS)
    
    if args.val_size > 0:
        generate_dataset_split("Validation", args.val_size, real_bp, real_ep, args.output_dir, use_lhs=USE_LHS)
        
    if args.test_size > 0:
        generate_dataset_split("Test", args.test_size, real_bp, real_ep, args.output_dir, use_lhs=USE_LHS)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Dataset Generation Time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")

    # Generate description.txt
    desc_path = os.path.join(args.output_dir, "description.txt")
    print(f"Creating description file: {desc_path}")
    try:
        with open(desc_path, "w") as f:
            f.write(f"Dataset Description\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Generator Script: generate_dataset_lhs.py (Unified)\n")
            f.write(f"Sampling Method: {'Latin Hypercube (LHS) - Uniform' if USE_LHS else 'Random - Gaussian (Heuristic)'}\n")
            f.write(f"Target Plant: {args.plant}\n")
            f.write(f"Output Directory: {args.output_dir}\n\n")
            f.write("=== Split Sizes ===\n")
            f.write(f"Total Generation Time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)\n")
            f.write(f"Train: {args.train_size}\n")
            f.write(f"Validation: {args.val_size}\n")
            f.write(f"Test: {args.test_size}\n")
            f.write(f"Total Samples: {args.train_size + args.val_size + args.test_size}\n")
            if USE_LHS:
                f.write("\n=== Param Ranges (LHS) ===\n")
                f.write(f"{PARAM_RANGES}\n")
    except Exception as e:
        print(f"Error creating description file: {e}")
        
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()
