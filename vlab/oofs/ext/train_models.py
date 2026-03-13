"""
Universal Training Pipeline for Plant Surrogate Models.
Trains multiple model architectures on a specified dataset run.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set umask to 0 so all created files/dirs are readable/writable by everyone (777/666)
# This allows the host user (pzu426) to modify files created by Docker root user
os.umask(0)

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# Import Model Architectures
# Ensure these files are in the python path or same directory
import surrogate_nn_dataset as baseline_model
import surrogate_nn_dataset_sinkhorn as sinkhorn_model
from plant_comparison_nn import read_real_plants
from utils_nn import log_training_stats

import torch.multiprocessing as mp

import argparse

# --- CONFIGURATION ---
DEFAULT_DATASET = "Run 030926"  # Default dataset folder name
USE_MULTIPROCESSING = True      # Set True to train models in parallel (CPU only recommended)
WORKERS_PER_MODEL = 0           # DataLoader workers (0 = main process)
FORCE_CPU = True               # Set True to force CPU usage

# Model Configurations
MODELS_TO_TRAIN = [
    {
        "name": "baseline_mlp",
        "dataset_class": baseline_model.PlantDataset,
        "model_class": baseline_model.HierarchicalPlantSurrogateNet,
        "loss_fn": baseline_model.hierarchical_loss_function,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10
    },
    {
        "name": "sinkhorn_transformer",
        "dataset_class": sinkhorn_model.PlantDataset,
        "model_class": sinkhorn_model.HierarchicalPlantSurrogateNet,
        "loss_fn": sinkhorn_model.hierarchical_loss_function,
        "batch_size": 32,
        "learning_rate": 1e-4, # Initial LR
        "epochs": 10
    }
]

parser = argparse.ArgumentParser(description="Train plant surrogate models.")
parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"Name of the dataset folder (default: '{DEFAULT_DATASET}')")
args, unknown = parser.parse_known_args()

DATASET_RUN = args.dataset  # The folder name in Datasets/

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "Datasets", DATASET_RUN)
OUTPUT_ROOT = os.path.join(BASE_DIR, "Training Data")
DATE_STR = datetime.now().strftime("%m%d%y")

# --- UTILS ---

def prepare_real_plant_batch(real_bp, real_ep, max_points=50):
    """Convert real plant data to fixed-size tensors (Shared memory for multiproc)"""
    num_days = len(real_bp)
    bp_batch = torch.zeros(1, num_days, max_points, 2)
    ep_batch = torch.zeros(1, num_days, max_points, 2)
    for day in range(num_days):
        if len(real_bp[day]) > 0:
            count = min(len(real_bp[day]), max_points)
            bp_batch[0, day, :count, :] = torch.tensor(real_bp[day][:count], dtype=torch.float32)
        if len(real_ep[day]) > 0:
            count = min(len(real_ep[day]), max_points)
            ep_batch[0, day, :count, :] = torch.tensor(real_ep[day][:count], dtype=torch.float32)
            
    # For multiprocessing, sharing memory avoids pickling overhead
    if USE_MULTIPROCESSING:
        bp_batch.share_memory_()
        ep_batch.share_memory_()
        
    return bp_batch, ep_batch

def train_one_epoch(model, loader, optimizer, real_bp_batch, real_ep_batch, real_bp_raw, real_ep_raw, model_config, training_log_csv=None, epoch_num=1):
    model.train()
    total_loss = 0.0
    
    # Check if model handles ghost probabilities (Sinkhorn does, Baseline doesn't)
    is_sinkhorn = "sinkhorn" in model_config["name"]
    loss_fn = model_config["loss_fn"]

    batch_idx = 0
    total_batches = len(loader)
    
    for params, costs in loader:
        optimizer.zero_grad()
        bs = params.size(0)
        
        # Normalize targets for training stability
        if hasattr(model, 'output_mean') and hasattr(model, 'output_std'):
             norm_costs = (costs - model.output_mean) / model.output_std
        else:
             norm_costs = costs
        
        # Prepare batch features
        curr_bp = real_bp_batch.repeat(bs, 1, 1, 1)
        curr_ep = real_ep_batch.repeat(bs, 1, 1, 1)
        
        # Forward Pass
        pred_cost = model(params, curr_bp, curr_ep)
        
        # Normalize prediction for loss calculation
        if hasattr(model, 'output_mean') and hasattr(model, 'output_std'):
             pred_cost_norm = (pred_cost - model.output_mean) / model.output_std
        else:
             pred_cost_norm = pred_cost
        
        # Structure Generation (for auxiliary loss)
        norm_params = (params - model.input_mean) / model.input_std
        
        # Handle structure gen call safely across different models
        bp_syn, bp_probs, ep_syn, ep_probs = None, None, None, None
        if hasattr(model, 'structure_gen'):
            bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen(norm_params)

        # Calculate Loss
        # We pass NORMALIZED costs to the loss function to balance with auxiliary losses
        loss, _, _, _ = loss_fn(
            pred_cost_norm, norm_costs, bp_syn, bp_probs, ep_syn, ep_probs, real_bp_raw, real_ep_raw
        )
             
        # Sinkhorn Temperature Clamping
        if is_sinkhorn:
             if hasattr(model, 'sinkhorn_net') and hasattr(model.sinkhorn_net, 'log_temperature'):
                 model.sinkhorn_net.log_temperature.data.clamp_(min=-5.0)

        loss.backward()
        
        # Gradient Clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
            current_loss = loss.item()
            print(f"[{model_config['name']}] Epoch {epoch_num} Batch {batch_idx + 1}/{total_batches} Loss: {current_loss:.4f}", flush=True)
            
            if training_log_csv:
                with open(training_log_csv, "a") as f:
                    f.write(f"{epoch_num},{batch_idx + 1},{current_loss:.6f},,,,\n")
        
        batch_idx += 1
        
    return total_loss / len(loader)

def validate(model, loader, real_bp_batch, real_ep_batch):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for params, costs in loader:
            bs = params.size(0)
            curr_bp = real_bp_batch.repeat(bs, 1, 1, 1)
            curr_ep = real_ep_batch.repeat(bs, 1, 1, 1)
            
            # Prediction in Normalized Space
            pred = model(params, curr_bp, curr_ep)
            
            # De-normalize Prediction and Target for Interpretable Metrics
            # Target (costs) are coming from dataset, which we normalized
            # But the dataset class usually returns them raw? 
            # Wait, the dataset class returns RAW costs.
            # The MODEL likely outputs normalized predictions if it was trained that way.
            # But here `model` is passed output_mean/std.
            # The model's forward() typically handles the internal normalization if implemented in HierarchicalPlantSurrogateNet.
            # However, standard practice is:
            # - Dataset returns RAW values.
            # - Model Input: normalized inside/outside.
            # - Model Output: normalized.
            # - Loss: calculated on normalized values.
            
            # Let's align with the training loop:
            # Training loop calls loss_fn(pred_cost, costs, ...)
            # If `costs` are raw, then `pred_cost` must be raw-scale or loss handles it?
            # Standard Surrogates usually predict normalized values.
            
            # Assuming current implementation:
            # Pred is normalized. Target is raw.
            # We need to de-normalize Pred to compare.
            
            all_preds.append(pred)
            all_targets.append(costs)
            
    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0).squeeze()
    all_targets = torch.cat(all_targets, dim=0).squeeze()
    
    # Check if model outputs normalized or real scale.
    # Based on current implementation of HierarchicalPlantSurrogateNet, forward() returns Real Scale (denormalized).
    # Dataset returns Real Scale costs.
    # So we compare directly without further transformation.
    
    real_preds = all_preds
    real_targets = all_targets

    # Calculate Metrics on REAL SCALE
    mse = F.mse_loss(real_preds, real_targets).item()
    mae = F.l1_loss(real_preds, real_targets).item()
    
    # Calculate Normalized Loss (to match training loss scale)
    val_loss = mse
    if hasattr(model, 'output_mean') and hasattr(model, 'output_std'):
         norm_preds = (real_preds - model.output_mean) / model.output_std
         norm_targets = (real_targets - model.output_mean) / model.output_std
         val_loss = F.mse_loss(norm_preds, norm_targets).item()
    
    # Relative Error
    rel_err = torch.abs(real_preds - real_targets) / (torch.abs(real_targets) + 1e-8)
    mean_rel_err = rel_err.mean().item()
    median_rel_err = rel_err.median().item()
    
    # R-squared (R2)
    target_mean = torch.mean(real_targets)
    ss_tot = torch.sum((real_targets - target_mean) ** 2)
    ss_res = torch.sum((real_targets - real_preds) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    r2 = r2.item()
    
    # Accuracy within 1% and 5% threshold
    acc_1pct = (rel_err < 0.01).float().mean().item() * 100
    acc_5pct = (rel_err < 0.05).float().mean().item() * 100
            
    return {
        "loss": val_loss,
        "real_mse": mse,
        "mae": mae,
        "rel_err": mean_rel_err,
        "median_rel_err": median_rel_err,
        "r2": r2,
        "acc_1pct": acc_1pct,
        "acc_5pct": acc_5pct
    }

def train_model_worker(config, run_dir, input_mean, input_std, output_mean, output_std, real_bp_batch, real_ep_batch, real_bp, real_ep, train_ds_args, val_ds_args, test_ds_args):
    """Worker function for multiprocessing training"""
    model_name = config["name"]
    print(f"\n[Worker] Starting {model_name}...")
    
    # Needs to re-instantiate datasets/loaders inside worker process on Linux
    # because passing DataLoaders across processes is problematic
    # Using the dataset class directly from config module
    PlantDataset = config["dataset_class"]
    
    train_ds = PlantDataset(*train_ds_args)
    val_ds = PlantDataset(*val_ds_args)
    test_ds = PlantDataset(*test_ds_args)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    print(f"[{model_name}] Dataset loaded. Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Setup Dirs
    model_dir = os.path.join(run_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    log_csv = os.path.join(model_dir, "training_log.csv")
    best_model_path = os.path.join(model_dir, "best_model.pt")
    
    # Init Model
    ModelClass = config["model_class"]
    model = ModelClass(
        input_mean=input_mean, 
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std
    )
    
    # Enable memory sharing for model parameters if needed
    if USE_MULTIPROCESSING:
        model.share_memory()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Init Log
    with open(log_csv, "w", buffering=1) as f:
        f.write("epoch,batch,train_loss,val_loss,val_mae,val_rel_err,val_rel_err_median,val_r2,val_acc_1pct,val_acc_5pct,time_sec\n")
        
    best_val_loss = float('inf')
    
    model_start_time = time.time()
    epoch_start = time.time()
    print(f"[{model_name}] Starting training loop for {config['epochs']} epochs...")
    
    for epoch in range(config["epochs"]):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, 
            real_bp_batch, real_ep_batch, 
            real_bp, real_ep,
            config,
            training_log_csv=log_csv,
            epoch_num=epoch+1
        )
        
        # Validate
        val_metrics = validate(model, val_loader, real_bp_batch, real_ep_batch)
        
        # Log
        elapsed = time.time() - epoch_start
        print(f"[{model_name}] Epoch {epoch+1}/{config['epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f} | R2: {val_metrics['r2']:.4f}")
        
        with open(log_csv, "a") as f:
            f.write(f"{epoch+1},ALL,{train_loss:.6f},{val_metrics['loss']:.6f},{val_metrics['mae']:.6f},"
                    f"{val_metrics['rel_err']:.6f},{val_metrics['median_rel_err']:.6f},{val_metrics['r2']:.6f},"
                    f"{val_metrics['acc_1pct']:.2f},{val_metrics['acc_5pct']:.2f},{elapsed:.2f}\n")
        
        # Save Best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), best_model_path)
            
        epoch_start = time.time()

    model_end_time = time.time()
    total_training_duration = model_end_time - model_start_time
    print(f"[Worker] Finished {model_name}. Best: {best_val_loss:.4f}. Duration: {total_training_duration:.2f}s")
    
    # Evaluation
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = validate(model, test_loader, real_bp_batch, real_ep_batch)
    
    results_txt = os.path.join(model_dir, "test_results.txt")
    with open(results_txt, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {DATE_STR}\n")
        
        # Dataset Info
        try:
            dataset_path = os.path.dirname(train_ds_args[0])
            dataset_name = os.path.basename(dataset_path)
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Split Sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}\n")
            f.write(f"Target Cost Mean: {output_mean:.4f}\n")
            f.write(f"Target Cost Std: {output_std:.4f}\n")
        except:
            f.write("Dataset info unavailable\n")

        f.write("=== Test Results ===\n")
        f.write(f"Test Norm Loss (MSE): {test_metrics['loss']:.4f}\n")
        f.write(f"Test Real MSE: {test_metrics['real_mse']:.4f}\n")
        f.write(f"Test Real MAE: {test_metrics['mae']:.4f}\n")
        
        # Add sanity check consistency metrics
        if output_std > 0:
            implied_rmse = (test_metrics['loss'] ** 0.5) * output_std
            f.write(f"Implied Real RMSE (from Norm Loss): {implied_rmse:.4f}\n")
            
        f.write(f"Test R2 Score: {test_metrics['r2']:.4f}\n")
        f.write(f"Mean Relative Error: {test_metrics['rel_err']:.4f}\n")
        f.write(f"Median Relative Error: {test_metrics['median_rel_err']:.4f}\n")
        f.write(f"Accuracy (<1% error): {test_metrics['acc_1pct']:.2f}%\n")
        f.write(f"\n=== Timing ===\n")
        f.write(f"Total Training Duration: {total_training_duration:.2f} seconds ({total_training_duration/3600:.2f} hours)\n")
        f.write(f"Start Time: {datetime.fromtimestamp(model_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time: {datetime.fromtimestamp(model_end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        f.write(f"Accuracy (<5% error): {test_metrics['acc_5pct']:.2f}%\n")
        
    print(f"[Worker] Saved results for {model_name}")

def main():
    if FORCE_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[Config] Force CPU enabled. CUDA_VISIBLE_DEVICES set to empty string.")

    # Use 'spawn' for CUDA compatibility later, but 'fork' is default/faster on CPU Linux

    # If errors occur, try mp.set_start_method('spawn')
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Auto-incrementing Run Name
    base_name = f"Run_{DATE_STR}"
    counter = 0
    candidate_name = base_name

    while os.path.exists(os.path.join(OUTPUT_ROOT, candidate_name)):
        counter += 1
        candidate_name = f"{base_name}_{counter}"

    RUN_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, candidate_name)
    if not os.path.exists(RUN_OUTPUT_DIR):
        try:
            os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
        except OSError:
            pass

    print(f"=== Starting Training Pipeline: {DATE_STR} ===")
    print(f"Dataset Run: {DATASET_RUN}")
    print(f"Multiprocessing: {USE_MULTIPROCESSING}")
    
    pipeline_start = time.time()
    
    # 1. Load Real Plant Data (Main Process)
    print("\nReading Real Plant Structure...")
    real_bp, real_ep = read_real_plants()
    real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep)
    
    # Shared Stats Calculation
    # We load one dataset just to get the stats
    train_csv = os.path.join(DATASETS_DIR, "Train.csv")
    val_csv = os.path.join(DATASETS_DIR, "Validation.csv")
    test_csv = os.path.join(DATASETS_DIR, "Test.csv")
    
    print(f"Calculating shared stats from {train_csv}...")
    # Use baseline dataset class for stats
    ds_temp = baseline_model.PlantDataset(train_csv)
    input_mean = ds_temp.params.mean(axis=0)
    input_std = ds_temp.params.std(axis=0) + 1e-8
    output_mean = ds_temp.costs.mean()
    output_std = ds_temp.costs.std() + 1e-8
    
    # --- Create Description File ---
    desc_path = os.path.join(RUN_OUTPUT_DIR, "description.txt")
    try:
        with open(desc_path, "w") as f:
            f.write(f"Training Run: {os.path.basename(RUN_OUTPUT_DIR)}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Source: {DATASET_RUN}\n")
            f.write(f"Dataset Path: {DATASETS_DIR}\n\n")
            
            f.write("=== Dataset Statistics (Normalization) ===\n")
            f.write(f"Number of training samples: {len(ds_temp)}\n")
            f.write(f"Input Parameters Mean:\n{input_mean}\n")
            f.write(f"Input Parameters Std:\n{input_std}\n")
            f.write(f"Cost Mean: {output_mean:.6f}\n")
            f.write(f"Cost Std: {output_std:.6f}\n\n")
            
            f.write("=== Models Configuration ===\n")
            for m in MODELS_TO_TRAIN:
                f.write(f"Model: {m['name']}\n")
                f.write(f"  Batch Size: {m['batch_size']}\n")
                f.write(f"  Learning Rate: {m['learning_rate']}\n")
                f.write(f"  Epochs: {m['epochs']}\n")
                f.write("-" * 30 + "\n")
            
            # Append Dataset Generation Log if available
            log_candidates = ["lhs_generation_log.txt", "generation_log.txt"]
            found_log = False
            for log_name in log_candidates:
                log_path = os.path.join(DATASETS_DIR, log_name)
                if os.path.exists(log_path):
                    f.write(f"\n=== Dataset Generation Log ({log_name}) ===\n")
                    try:
                        with open(log_path, "r") as log_f:
                            f.write(log_f.read())
                        found_log = True
                        break # Only include one log if multiple exist
                    except Exception as e:
                        f.write(f"Error reading log file: {e}\n")
            
            if not found_log:
                f.write("\nNo dataset generation log found.\n")

        print(f"Created description file: {desc_path}")
    except Exception as e:
        print(f"Warning: Could not create description file: {e}")

    # Arguments for Dataset creation inside workers
    # (csv_file, root_dir=None)
    train_ds_args = (train_csv, None)
    val_ds_args = (val_csv, None)
    test_ds_args = (test_csv, None)
    
    if USE_MULTIPROCESSING:
        print(f"Spawning {len(MODELS_TO_TRAIN)} parallel processes...")
        processes = []
        for config in MODELS_TO_TRAIN:
            p = mp.Process(target=train_model_worker, args=(
                config, RUN_OUTPUT_DIR, input_mean, input_std, output_mean, output_std,
                real_bp_batch, real_ep_batch, real_bp, real_ep,
                train_ds_args, val_ds_args, test_ds_args
            ))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
    else:
        print("Running sequentially...")
        for config in MODELS_TO_TRAIN:
            train_model_worker(
                config, RUN_OUTPUT_DIR, input_mean, input_std, output_mean, output_std,
                real_bp_batch, real_ep_batch, real_bp, real_ep,
                train_ds_args, val_ds_args, test_ds_args
            )

    pipeline_end = time.time()
    pipeline_duration = pipeline_end - pipeline_start
    print(f"\nAll models finished training. Total Pipeline Duration: {pipeline_duration:.2f} seconds ({pipeline_duration/3600:.2f} hours)")

    try:
        with open(desc_path, "a") as f:
            f.write(f"\n=== Training Pipeline Timing ===\n")
            f.write(f"Total Duration: {pipeline_duration:.2f} seconds ({pipeline_duration/3600:.2f} hours)\n")
            f.write(f"Start Time: {datetime.fromtimestamp(pipeline_start).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {datetime.fromtimestamp(pipeline_end).strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"Could not update description file with timing: {e}")

if __name__ == "__main__":
    main()
