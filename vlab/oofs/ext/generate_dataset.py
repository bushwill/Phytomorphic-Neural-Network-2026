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
import utils_nn
from utils_nn import read_real_plants, calculate_cost, plant_images_path, generate_plant, read_syn_plant, build_parameter_file
import utils_nn as plant_comparison_nn
import subprocess
import concurrent.futures

# Prevent docker generated files from locking out host user
os.umask(0)

# Ensure project modules are in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Dataset Generation Parameters
PLANT_NAME = "Plant_063-32"   # Target real plant name
TRAIN_SIZE = 10000            # Number of training samples
VAL_SIZE = 100                # Number of validation samples
TEST_SIZE = 3000              # Number of test samples
BASE_OUTPUT_DIR = "Datasets"  # Base directory for generated datasets
SAMPLING_METHOD = "random"

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

def generate_random_samples(n_samples):
    """Generate samples using the exact Nazifa random parameter distribution."""
    samples = np.zeros((n_samples, len(PARAM_NAMES)))
    for i in range(n_samples):
        # Matches utils_nn.build_random_parameter_file
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

def _process_sample(args):
    i, params, split_name, real_bp, real_ep, structures_dir, lsystem_tmp_root = args

    uid = uuid.uuid4().hex[:8]
    worker_ws = None
    try:
        # Create temporary file for LPFG execution to isolate parallel runs
        worker_ws = os.path.join(lsystem_tmp_root, f"worker_{uid}")
        os.makedirs(worker_ws, exist_ok=True)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        lsystem_base = os.path.join(script_dir, "lsystem")

        # Copy essential LPFG files into the temp file
        for item in os.listdir(lsystem_base):
            src = os.path.join(lsystem_base, item)
            if os.path.isfile(src) and not item.endswith('.o') and not item.endswith('.so'):
                shutil.copy2(src, os.path.join(worker_ws, item))

        # Compile project.cpp in the worker workspace if it doesn't have it (necessary for LPFG)
        if not os.path.exists(os.path.join(worker_ws, "project")):
            ret = os.system(f"g++ -o {os.path.join(worker_ws, 'project')} -Wall -Wextra {os.path.join(worker_ws, 'project.cpp')} -lm")
            if ret != 0:
                return i, False, "project.cpp compilation failed"

        temp_param_file = os.path.join(worker_ws, f"temp_{split_name}_{uid}.vset")
        temp_out_dir = os.path.join(worker_ws, f"temp_out_{split_name}_{uid}")
        os.makedirs(temp_out_dir, exist_ok=True)

        # 1. Generate L-System Structure
        # Generate parameter file
        build_parameter_file(temp_param_file, params)
        
        # Build lpfg command components
        abs_output_dir = os.path.abspath(temp_out_dir)
        abs_param_file = os.path.abspath(temp_param_file)
        
        # Function to get lsystem file paths from temp file
        def ls(f): return os.path.join(worker_ws, f)

        # Build lpfg command components 
        lpfg_args = [
            "lpfg", 
            "-w", "306", "256", 
            ls("lsystem.l"), 
            ls("view.v"), 
            ls("materials.mat"), 
            ls("contours.cset"), 
            ls("functions.fset"), 
            ls("functions.tset"), 
            abs_param_file
        ]

        # Run lpfg inside the isolated temp_out_dir
        log_file = os.path.join(abs_output_dir, "lpfg_log.txt")
        with open(log_file, "w") as f_log:
            process = subprocess.Popen(lpfg_args, cwd=abs_output_dir, stdout=f_log, stderr=subprocess.STDOUT)
            process.wait()

        # Run project executable explicitly from worker workspace
        project_exe = os.path.join(worker_ws, "project")
        leafval_file = "leafposition.dat" 
        output_txt = os.path.join(abs_output_dir, "output.txt")
        
        with open(output_txt, "w") as f_out:
            p_args = [project_exe, "2454", "2056", leafval_file]
            p_proc = subprocess.Popen(p_args, cwd=abs_output_dir, stdout=f_out)
            p_proc.wait()
        
        if not os.path.exists(output_txt):
            return i, False, "output.txt not generated"
            
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
        
        row = [i, total_cost] + params.tolist()
        return i, True, row

    except Exception as e:
        return i, False, str(e)
    finally:
        # Clean up isolated worker workspace
        if worker_ws and os.path.exists(worker_ws):
            shutil.rmtree(worker_ws, ignore_errors=True)

def cleanup_worker_dirs(lsystem_tmp_root):
    """Remove leftover worker directories under an lsystem tmp split folder."""
    removed = 0
    failed = 0
    if not os.path.isdir(lsystem_tmp_root):
        return removed, failed

    for name in os.listdir(lsystem_tmp_root):
        if not name.startswith("worker_"):
            continue
        worker_path = os.path.join(lsystem_tmp_root, name)
        if not os.path.isdir(worker_path):
            continue
        try:
            shutil.rmtree(worker_path)
            removed += 1
        except Exception as e:
            failed += 1
            logging.warning(f"  Failed to remove stale worker directory {worker_path}: {e}")

    return removed, failed

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
        
    params_array = generate_random_samples(size)

    real_bp, real_ep = real_data
    valid_count = 0
    start_time = time.time()
    
    # Keep L-system temp artifacts inside the dedicated lsystem workspace.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lsystem_tmp_root = os.path.join(script_dir, "lsystem", "tmp", split_name)
    os.makedirs(lsystem_tmp_root, exist_ok=True)

    # Remove leftovers from interrupted runs before launching new workers.
    removed, failed = cleanup_worker_dirs(lsystem_tmp_root)
    if removed > 0 or failed > 0:
        logging.info(f"  Pre-run worker cleanup in {lsystem_tmp_root}: removed={removed}, failed={failed}")

    args_list = [(i, params_array[i], split_name, real_bp, real_ep, structures_dir, lsystem_tmp_root) for i in range(size)]

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i, success, result in executor.map(_process_sample, args_list):
                if success:
                    # Save CSV Row serially to avoid exact parallel append conflicts
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow(result)
                    valid_count += 1
                else:
                    logging.warning(f"  Sample {i} failed: {result}")
                
                if (i + 1) % 50 == 0:
                    logging.info(f"  Processed {i+1}/{size} samples.")
    finally:
        # Best-effort cleanup for any crash/interruption leftovers.
        removed, failed = cleanup_worker_dirs(lsystem_tmp_root)
        if removed > 0 or failed > 0:
            logging.info(f"  Post-run worker cleanup in {lsystem_tmp_root}: removed={removed}, failed={failed}")

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
    utils_nn.configure_output_file_logging(base_dir, os.path.basename(base_dir))
    setup_logging(os.path.join(base_dir, "generation_log.txt"))
    
    logging.info(f"=== Dataset Generation Started ===")
    logging.info(f"Target Plant: {args.plant}")
    logging.info(f"Output Directory: {base_dir}")
    logging.info(f"Method: RANDOM")
    
    # 1. Load Real Plant Data
    try:
        # Let read_real_plants handle the plant name
        # Fix for path construction if module var is relative
        if not os.path.isabs(plant_comparison_nn.plant_images_path):
             # Assume relative to module loc or CWD. Let's use as-is but ensure it exists?
             pass
        
        real_bp, real_ep = read_real_plants(args.plant)
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
        f.write("Method: random\n")
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
    main()
