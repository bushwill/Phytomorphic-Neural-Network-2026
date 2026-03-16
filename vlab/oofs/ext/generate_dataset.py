"""
Dataset Generation Script for PhytomorphicNN.
Generates synthetic plant structures using L-systems with sampled parameters.
Supports Latin Hypercube Sampling (LHS) and Random Gaussian Sampling.
"""

import os
import sys
import csv
import time
import shutil
import uuid
import logging
import argparse
import numpy as np
import torch
from datetime import datetime
from scipy.stats import qmc, norm

# Ensure project modules are in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from plant_comparison_nn import read_real_plants, calculate_cost, plant_images_path
    import plant_comparison_nn
    from utils_nn import generate_plant, read_syn_plant, build_parameter_file
except ImportError:
    print("Error: Could not import project modules. Ensure they are in the python path.")
    sys.exit(1)

# --- Configuration ---
DEFAULT_PLANT = "Plant_063-32"
DEFAULT_TRAIN_SIZE = 10000
DEFAULT_VAL_SIZE = 100
DEFAULT_TEST_SIZE = 3000

# Param Ranges (Min, Max)
# Based on usage in lsystem/lsystem.pnl
PARAM_RANGES = [
    (1, 20),         # 0: max_phytomers
    (0.1, 10.0),     # 1: plastochron
    (-180.0, 180.0), # 2: plant_roll_angle
    (0.0, 180.0),    # 3: plant_down_angle
    (0.0, 270.0),    # 4: branch_angle
    (0.1, 10.0),     # 5: leaf_len
    (0.01, 1.0),     # 6: exp_leaf_wid
    (0.01, 2.0),     # 7: leaf_wid
    (0.0, 180.0),    # 8: leaf_bend_scale
    (0.0, 360.0),    # 9: leaf_twist_scale
    (0.0, 2.0),      # 10: node_len
    (0.01, 2.0),     # 11: int_wid
    (0.0, 1.0)       # 12: exp_int_rad
]

def setup_logging(log_file):
    """Configures logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def generate_lhs_samples(n_samples):
    """Generate samples using Latin Hypercube Sampling (Uniform)."""
    sampler = qmc.LatinHypercube(d=len(PARAM_RANGES))
    sample = sampler.random(n=n_samples)
    scaled = qmc.scale(sample, [r[0] for r in PARAM_RANGES], [r[1] for r in PARAM_RANGES])
    scaled[:, 0] = np.round(scaled[:, 0]) # max_phytomers is integer
    np.maximum(scaled[:, 0], 1.0, out=scaled[:, 0]) # Ensure >= 1
    return scaled

def generate_gaussian_samples(n_samples):
    """Generate samples using Gaussian distributions (Heuristic)."""
    samples = np.zeros((n_samples, len(PARAM_RANGES)))
    for i in range(n_samples):
        # Heuristics mimicking utils_nn.build_random_parameter_file
        chirality = -1.0 if np.random.random() < 0.5 else 1.0
        
        row = [
            norm.rvs(10., 1.),                  # max_phytomers
            norm.rvs(3., 0.1),                  # plastochron
            norm.rvs(chirality * 90., 10.0),    # plant_roll_angle
            norm.rvs(0., 4.0),                  # plant_down_angle
            norm.rvs(135., 5.),                 # branch_angle
            norm.rvs(5., 1.),                   # leaf_len
            norm.rvs(0.5, 0.01),                # exp_leaf_wid
            norm.rvs(1., 0.1),                  # leaf_wid
            norm.rvs(90., 3.),                  # leaf_bend_scale
            norm.rvs(180., 3.),                 # leaf_twist_scale
            norm.rvs(0.7, 0.05),                # node_len
            norm.rvs(0.9, 0.01),                # int_wid
            norm.rvs(0.5, 0.01)                 # exp_int_rad
        ]
        samples[i] = row
        
    # Clip to valid ranges just in case
    for dim, (low, high) in enumerate(PARAM_RANGES):
        np.clip(samples[:, dim], low, high, out=samples[:, dim])
        
    samples[:, 0] = np.round(samples[:, 0])
    return samples

def generate_split(split_name, size, real_data, output_dir, use_lhs=True):
    """Generates a dataset split (Train/Val/Test)."""
    if size <= 0:
        return

    logging.info(f"\nGenerating {split_name} split ({size} samples)...")
    
    split_dir = os.path.join(output_dir, split_name)
    structures_dir = os.path.join(split_dir, "structures")
    os.makedirs(structures_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{split_name}.csv")
    csv_header = ["id", "cost"] + [f"param_{i}" for i in range(len(PARAM_RANGES))]
    
    # Init CSV
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_header)
        
    # Generate Parameters
    if use_lhs:
        params_array = generate_lhs_samples(size)
    else:
        params_array = generate_gaussian_samples(size)

    real_bp, real_ep = real_data
    valid_count = 0
    start_time = time.time()
    
    # Process Loop
    for i in range(size):
        params = params_array[i]
        
        # Temporary unique workspace
        uid = uuid.uuid4().hex[:8]
        temp_param_file = f"temp_{split_name}_{uid}.vset"
        temp_out_dir = f"temp_out_{split_name}_{uid}"
        os.makedirs(temp_out_dir, exist_ok=True)
        
        try:
            # 1. Generate L-System Structure
            build_parameter_file(temp_param_file, params)
            generate_plant(temp_param_file, temp_out_dir)
            
            output_txt = os.path.join(temp_out_dir, "output.txt")
            if not os.path.exists(output_txt):
                continue
                
            syn_bp, syn_ep = read_syn_plant(output_txt)
            
            # 2. Calculate Cost
            total_cost = 0.0
            num_days = min(len(syn_bp), len(real_bp))
            
            if num_days == 0:
                total_cost = 1e6 # Penalty for empty
            else:
                for day in range(num_days):
                    total_cost += calculate_cost(syn_bp[day], syn_ep[day], real_bp[day], real_ep[day])

            # 3. Save Results
            # Save Structure Tensor
            struct_data = {
                "bp": syn_bp,
                "ep": syn_ep,
                "params": torch.tensor(params, dtype=torch.float32),
                "cost": torch.tensor(total_cost, dtype=torch.float32)
            }
            torch.save(struct_data, os.path.join(structures_dir, f"structure_{i}.pt"))
            
            # Save CSV Row
            with open(csv_path, "a", newline="") as f:
                row = [i, total_cost] + params.tolist()
                csv.writer(f).writerow(row)
                
            valid_count += 1
            if (i + 1) % 50 == 0:
                logging.info(f"  Processed {i+1}/{size} samples.")

        except Exception as e:
            logging.warning(f"  Sample {i} failed: {e}")
        finally:
            # Clean up temp files
            if os.path.exists(temp_param_file):
                os.remove(temp_param_file)
            if os.path.exists(temp_out_dir):
                shutil.rmtree(temp_out_dir)

    logging.info(f"Finished {split_name}. Valid: {valid_count}/{size}. Time: {time.time() - start_time:.2f}s")
    return valid_count

def main():
    parser = argparse.ArgumentParser(description="Generate Plant Surrogate Datasets.")
    parser.add_argument("--plant", default=DEFAULT_PLANT, help="Target real plant name")
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--output_dir", default=None, help="Base output directory")
    parser.add_argument("--method", choices=["lhs", "random"], default="lhs", help="Sampling method")
    args = parser.parse_args()
    
    # 0. Setup Directories
    base_dir = args.output_dir
    if base_dir is None:
        date_str = datetime.now().strftime("%m%d%y")
        candidate = f"Run_{date_str}"
        counter = 0
        while os.path.exists(os.path.join("Datasets", candidate)):
            counter += 1
            candidate = f"Run_{date_str}_{counter}"
        base_dir = os.path.join("Datasets", candidate)
        
    os.makedirs(base_dir, exist_ok=True)
    setup_logging(os.path.join(base_dir, "generation_log.txt"))
    
    logging.info(f"=== Dataset Generation Started ===")
    logging.info(f"Target Plant: {args.plant}")
    logging.info(f"Output Directory: {base_dir}")
    logging.info(f"Method: {args.method.upper()}")
    
    # 1. Load Real Plant Data
    try:
        plant_comparison_nn.real_plant_name = args.plant
        # Fix for path construction if module var is relative
        if not os.path.isabs(plant_comparison_nn.plant_images_path):
             # Assume relative to module loc or CWD. Let's use as-is but ensure it exists?
             pass
        
        real_bp, real_ep = read_real_plants()
        logging.info(f"Loaded real plant data ({len(real_bp)} days).")
        real_data = (real_bp, real_ep)
    except Exception as e:
        logging.error(f"Failed to load real plant data: {e}")
        return

    # 2. Generate Splits
    use_lhs = (args.method == "lhs")
    
    generate_split("Train", args.train_size, real_data, base_dir, use_lhs)
    generate_split("Validation", args.val_size, real_data, base_dir, use_lhs)
    generate_split("Test", args.test_size, real_data, base_dir, use_lhs)
    
    # 3. Create Description File
    with open(os.path.join(base_dir, "description.txt"), "w") as f:
        f.write(f"Dataset: {os.path.basename(base_dir)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Plant: {args.plant}\n")
        f.write(f"Sizes: Train={args.train_size}, Val={args.val_size}, Test={args.test_size}\n")
        f.write(f"Ranges: {PARAM_RANGES}\n")

    logging.info("Generation Complete.")

if __name__ == "__main__":
    # Allow host access to created files
    os.umask(0)
    main()
