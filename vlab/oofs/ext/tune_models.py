"""
Hyperparameter Tuning Script.

Runs multiple configurations of Surrogate Models with varying hyperparameters
(Learning Rate, Batch Size, Epochs) to identify optimal settings.
"""

import os
import time
import argparse
import itertools
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Import model definitions
import model_hungarian as baseline_model
import model_sinkhorn as sinkhorn_model
import model_mlp as benchmark_mlp
import utils_nn

# Import training worker from existing pipeline
from train_models import train_model_worker, aggregate_results, prepare_real_plant_batch

# --- User Configuration ---
DATASET_NAME = "Run 021926"
PLANT_NAME = "Plant_063-32"
NUM_REPLICATES = 2

# Define Hyperparameter Search Space
# Each model will be run with each combination of parameters below
HP_GRID = {
    # Models to tune (choose keys carefully)
    "models": ["baseline_hierarchical", "sinkhorn_hierarchical"],
    
    # Tuning Parameters (Cartesian Product)
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size": [16, 32],
    "epochs": [10]  # Keep low for tuning, increase for final training
}

# Base Configuration for Models
MODEL_BASE_CONFIGS = {
    "baseline_hierarchical": {
        "dataset_class": baseline_model.PlantDataset,
        "model_class": baseline_model.HierarchicalPlantSurrogateNet,
        "loss_fn": baseline_model.hierarchical_loss_function,
        "module": baseline_model,
    },
    "sinkhorn_hierarchical": {
        "dataset_class": sinkhorn_model.PlantDataset,
        "model_class": sinkhorn_model.HierarchicalPlantSurrogateNet,
        "loss_fn": sinkhorn_model.hierarchical_loss_function,
        "module": sinkhorn_model
    }
}

def load_data(dataset_name="Run 031326", plant_name="Plant_063-32"):
    """
    Loads dataset stats and real plant structure for comparison.
    Adapted from train_models.py main()
    """
    # Assuming the script is in ext/
    # And Datasets/ is at ../../../../Datasets/
    # Let's be safer with paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Dataset is in the current directory (ext/Datasets)
    dataset_dir = os.path.join(current_dir, "Datasets", dataset_name)
    
    # Validation
    if not os.path.exists(dataset_dir):
         print(f"Error: Dataset not found at {dataset_dir}")
         # Check via ls
         print(f"Contents of {current_dir}:")
         print(os.listdir(current_dir))
         return None

    train_csv = os.path.join(dataset_dir, "Train.csv")
    val_csv = os.path.join(dataset_dir, "Validation.csv")
    test_csv = os.path.join(dataset_dir, "Test.csv")
    
    print(f"Loading Data from: {dataset_dir}")
    
    # Load stats for normalization
    import pandas as pd
    try:
        df = pd.read_csv(train_csv)
    except FileNotFoundError:
        print(f"Error: Could not find Train.csv in {dataset_dir}")
        return None
    
    # Check column count to distinguish standard MLP vs Hierarchical expectations
    # Assuming params start at col 2 and end at 15 (13 params)
    params = df.iloc[:, 2:15].values
    costs = df.iloc[:, 1].values
    
    input_mean = np.mean(params, axis=0)
    input_std = np.std(params, axis=0) + 1e-6
    output_mean = np.mean(costs)
    output_std = np.std(costs) + 1e-6
    
    # Load Real Plant Structure
    # utils_nn.read_real_plants() uses hardcoded relative path "./Real Plants/"
    # which works if running script from ext directory.
    print(f"Loading Real Plant Structure (via utils_nn)...")
    try:
         # Note: read_real_plants() takes no arguments and uses global path relative to CWD
         real_bp, real_ep = utils_nn.read_real_plants()
         
         # Convert using train_models helper for consistency
         real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep, use_multiprocessing=False)

    except Exception as e:
         print(f"Error reading real plant: {e}")
         print("Using dummy plant structure (zeros)")
         # create dummy
         # real_bp list of 26 arrays of size (50,3)
         real_bp = [np.zeros((50,3)) for _ in range(26)]
         real_ep = [np.zeros((50,3)) for _ in range(26)]
         real_bp_batch = torch.zeros(26, 50, 3)
         real_ep_batch = torch.zeros(26, 50, 3)

    return {
        "input_mean": input_mean, "input_std": input_std,
        "output_mean": output_mean, "output_std": output_std,
        "real_bp": real_bp, "real_ep": real_ep,
        "real_bp_batch": real_bp_batch, "real_ep_batch": real_ep_batch,
        "train_csv": train_csv, "val_csv": val_csv, "test_csv": test_csv,
        "dataset_dir": dataset_dir 
    }

def main():
    print("--- Hyperparameter Tuning Script ---")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Plant: {PLANT_NAME}")
    print(f"Replicates: {NUM_REPLICATES}")
    
    # 1. Setup Data
    data = load_data(DATASET_NAME, PLANT_NAME)
    if data is None:
        return
        
    train_ds_args = (data["train_csv"], None)
    val_ds_args = (data["val_csv"], None)
    test_ds_args = (data["test_csv"], None)
    
    # 2. Generate Configurations
    keys = ["learning_rate", "batch_size", "epochs"]
    values = [HP_GRID[k] for k in keys]
    combinations = list(itertools.product(*values))
    
    run_configs = []
    
    for model_name in HP_GRID["models"]:
        base_config = MODEL_BASE_CONFIGS.get(model_name)
        if not base_config:
            print(f"Warning: No base config for {model_name}, skipping.")
            continue
            
        for lr, bs, ep in combinations:
            config = base_config.copy()
            config["learning_rate"] = lr
            config["batch_size"] = bs
            config["epochs"] = ep
            
            # Create a unique name for this run
            slug = f"{model_name}_lr{lr}_bs{bs}_ep{ep}"
            config["name"] = slug
            run_configs.append(config)
            
    print(f"Generated {len(run_configs)} configurations to test.")
    
    # 3. Execution Loop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_dir = os.path.join(data["dataset_dir"], f"Tuning_Run_{timestamp}")
    os.makedirs(tuning_dir, exist_ok=True)
    
    print(f"Output Directory: {tuning_dir}")
    
    # Save HP Grid for reference
    with open(os.path.join(tuning_dir, "hp_grid.txt"), "w") as f:
         f.write(str(HP_GRID))

    # --- Generate description.txt ---
    # Matching the format requested in train_models.py
    with open(os.path.join(tuning_dir, "description.txt"), "w") as f:
        f.write(f"Run ID: Tuning_Run_{timestamp}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Ref Plant: {PLANT_NAME}\n")
        f.write("========================================\n")
        f.write("Configuration:\n")
        f.write(f"  Replicates: {NUM_REPLICATES}\n")
        f.write(f"  Type: Hyperparameter Tuning\n")
        f.write(f"  Grid: {HP_GRID}\n")
        f.write("========================================\n")
        f.write("Statistics (Normalization):\n")
        # Ensure full numpy arrays are printed
        np_print_opts = np.get_printoptions()
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        f.write(f"  Input Mean: {data['input_mean']}\n")
        f.write(f"  Input Std:  {data['input_std']}\n")
        np.set_printoptions(**np_print_opts) # Restore options
        f.write(f"  Cost Mean:  {data['output_mean']:.6f}\n")
        f.write(f"  Cost Std:   {data['output_std']:.6f}\n")
        f.write("========================================\n")
        f.write("Models Tested:\n")
        for m in HP_GRID['models']:
             f.write(f"  - {m}\n")
    
    pipeline_start = time.time()
    
    # Execution
    for i, config in enumerate(run_configs):
        print(f"\n--- Starting Configuration {i+1}/{len(run_configs)}: {config['name']} ---")
        
        for rep in range(1, NUM_REPLICATES + 1):
             # Ensure types for train_model_worker
             # It expects numpy arrays for stats, not Tensors (it converts them internally)
             # load_data returns numpy arrays, so we are good.
            
            train_model_worker(
                config, 
                tuning_dir,
                data["input_mean"], data["input_std"], 
                data["output_mean"], data["output_std"],
                data["real_bp_batch"], data["real_ep_batch"],
                data["real_bp"], data["real_ep"],
                train_ds_args, val_ds_args, test_ds_args,
                replicate_id=rep,
                use_multiprocessing=False # Avoid nesting multiprocessing if possible
            )

    print(f"\nTuning Complete. Duration: {time.time() - pipeline_start:.2f}s")
    
    try:
        aggregate_results(tuning_dir)
        print(f"Results aggregated in {tuning_dir}/summary_results.csv")
    except Exception as e:
        print(f"Aggregation Error: {e}") 

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
