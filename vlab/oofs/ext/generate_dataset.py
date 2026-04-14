"""
Dataset Generation Script for PhytomorphicNN.
Generates synthetic plant structures using L-systems with sampled parameters.
Sampling uses the same random parameter distribution as Nazifa.
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
from numpy.random import normal as nran
from numpy.random import uniform as uran

# Ensure project modules are in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils_nn import read_real_plants, calculate_cost, plant_images_path, generate_plant, read_syn_plant, build_parameter_file
    import utils_nn as plant_comparison_nn
except ImportError:
    print("Error: Could not import project modules. Ensure they are in the python path.")
    sys.exit(1)

# --- USER CONFIGURATION ---
# Dataset Generation Parameters
PLANT_NAME = "Plant_063-32"   # Target real plant name
TRAIN_SIZE = 10000            # Number of training samples
VAL_SIZE = 100                # Number of validation samples
TEST_SIZE = 3000              # Number of test samples
BASE_OUTPUT_DIR = "Datasets"  # Base directory for generated datasets
SAMPLING_METHOD = "random"

# --- END USER CONFIGURATION ---

DEFAULT_PLANT = PLANT_NAME
DEFAULT_TRAIN_SIZE = TRAIN_SIZE
DEFAULT_VAL_SIZE = VAL_SIZE
DEFAULT_TEST_SIZE = TEST_SIZE

PARAM_NAMES = [
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

def configure_output_file_logging(output_dir, run_label):
    """Route stdout/stderr to a persistent log file when attached to a TTY."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{run_label}_terminal_output.log")
    stream = open(log_path, "a", buffering=1)

    if sys.stdout.isatty() or sys.stderr.isatty():
        notice = f"[Logging] Redirecting stdout/stderr to {log_path}"
        try:
            os.write(1, (notice + "\n").encode("utf-8", errors="replace"))
        except OSError:
            pass

        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
        sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)
        print(notice)

    return log_path

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

def generate_nazifa_random_samples(n_samples):
    """Generate samples using the exact Nazifa random parameter distribution."""
    samples = np.zeros((n_samples, len(PARAM_NAMES)))
    for i in range(n_samples):
        # Match utils_nn.build_random_parameter_file exactly.
        chirality = -1.0 if uran(0.0, 1.0) < 0.5 else 1.0

        row = [
            nran(10.0, 1.0),
            nran(3.0, 0.1),
            nran(chirality * 90.0, 10.0),
            nran(0.0, 4.0),
            nran(135.0, 5.0),
            nran(5.0, 1.0),
            nran(0.5, 0.01),
            nran(1.0, 0.1),
            nran(90.0, 3.0),
            nran(180.0, 3.0),
            nran(0.7, 0.05),
            nran(0.9, 0.01),
            nran(0.5, 0.01),
        ]
        samples[i] = row

    return samples

def generate_split(split_name, size, real_data, output_dir):
    """Generates a dataset split (Train/Val/Test)."""
    if size <= 0:
        return

    logging.info(f"\nGenerating {split_name} split ({size} samples)...")
    
    split_dir = os.path.join(output_dir, split_name)
    structures_dir = os.path.join(split_dir, "structures")
    os.makedirs(structures_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{split_name}.csv")
    csv_header = ["id", "cost"] + [f"param_{i}" for i in range(len(PARAM_NAMES))]
    
    # Init CSV
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_header)
        
    # Generate parameters using Nazifa's random distribution.
    params_array = generate_nazifa_random_samples(size)

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
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 0. Setup Directories
    base_dir = args.output_dir
    if base_dir is None:
        date_str = datetime.now().strftime("%m%d%y")
        candidate = f"Run_{date_str}"
        counter = 0
        datasets_root = os.path.join(script_dir, BASE_OUTPUT_DIR)
        while os.path.exists(os.path.join(datasets_root, candidate)):
            counter += 1
            candidate = f"Run_{date_str}_{counter}"
        base_dir = os.path.join(datasets_root, candidate)
    elif not os.path.isabs(base_dir):
        # Keep user-provided relative paths inside this project by default.
        base_dir = os.path.join(script_dir, base_dir)
        
    os.makedirs(base_dir, exist_ok=True)
    log_path = configure_output_file_logging(base_dir, os.path.basename(base_dir))
    setup_logging(os.path.join(base_dir, "generation_log.txt"))
    
    logging.info(f"=== Dataset Generation Started ===")
    logging.info(f"Target Plant: {args.plant}")
    logging.info(f"Output Directory: {base_dir}")
    logging.info(f"Terminal Log: {log_path}")
    logging.info("Method: RANDOM (Nazifa distribution)")
    
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
    generate_split("Train", args.train_size, real_data, base_dir)
    generate_split("Validation", args.val_size, real_data, base_dir)
    generate_split("Test", args.test_size, real_data, base_dir)
    
    # 3. Create Description File
    with open(os.path.join(base_dir, "description.txt"), "w") as f:
        f.write(f"Dataset: {os.path.basename(base_dir)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("Method: random (Nazifa distribution)\n")
        f.write(f"Plant: {args.plant}\n")
        f.write(f"Sizes: Train={args.train_size}, Val={args.val_size}, Test={args.test_size}\n")
        f.write("Distribution:\n")
        f.write("max_phytomers ~ N(10, 1)\n")
        f.write("plastochron ~ N(3, 0.1)\n")
        f.write("plant_roll_angle ~ N(chirality*90, 10), chirality in {-1,+1} with p=0.5\n")
        f.write("plant_down_angle ~ N(0, 4)\n")
        f.write("branch_angle ~ N(135, 5)\n")
        f.write("leaf_len ~ N(5, 1)\n")
        f.write("exp_leaf_wid ~ N(0.5, 0.01)\n")
        f.write("leaf_wid ~ N(1, 0.1)\n")
        f.write("leaf_bend_scale ~ N(90, 3)\n")
        f.write("leaf_twist_scale ~ N(180, 3)\n")
        f.write("node_len ~ N(0.7, 0.05)\n")
        f.write("int_wid ~ N(0.9, 0.01)\n")
        f.write("exp_int_rad ~ N(0.5, 0.01)\n")

    logging.info("Generation Complete.")

if __name__ == "__main__":
    # Allow host access to created files
    os.umask(0)
    main()
