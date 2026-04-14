"""
Universal Training Pipeline for Plant Surrogate Models.

Features:
- Multi-Model Support: Trains MLP, Hungarian, and Sinkhorn models in one run.
- Multiprocessing: Uses parallel workers for replicate runs.
- Automatic Normalization: Computes and registers input/output statistics.
- Unified Outputs: Writes checkpoints, logs, metrics.json, and summary CSVs.

Process:
1. Load Real Plant Data (Target Structure).
2. Load Training/Validation Datasets (Synthetic Param-Cost pairs).
3. Compute Normalization Statistics (Mean/Std of params and costs).
4. Launch worker processes for each model replicate.
5. Train loop: forward -> loss -> backprop -> optimize.
6. Run post-training evaluation figures for each trained model.
"""

import os
import sys
import time
import argparse
import shutil
import json
import numpy as np
from datetime import datetime
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.nn.functional as F

# --- Import Model Architectures ---
import model_hungarian
import model_sinkhorn
import model_mlp
import utils_nn as plant_comparison_nn
from utils_nn import read_real_plants

# --- USER CONFIGURATION ---
DATASET_NAME = "Run 031326"  # Source folder in Datasets/
PLANT_NAME = "Plant_063-32"  # Real plant reference for structure comparison
NUM_REPLICATES = 3           # Independent runs per model type
NUM_EPOCHS = 10              # Training duration
BATCH_SIZE = 32              # Batch size (Tuned)
LEARNING_RATE = 5e-4         # Learning Rate (Tuned)
USE_MULTIPROCESSING = True   # Parallel training

# Model Registry
MODELS_TO_TRAIN = [
    {
        "name": "baseline",
        "dataset_class": model_mlp.PlantDataset,
        "model_class": model_mlp.BenchmarkSurrogateNet,
        "loss_fn": model_mlp.benchmark_loss_function,  
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "module": model_mlp 
    },
    {
        "name": "hungarian",
        "dataset_class": model_hungarian.PlantDataset,
        "model_class": model_hungarian.HierarchicalPlantSurrogateNet,
        "loss_fn": model_hungarian.hierarchical_loss_function,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "module": model_hungarian
    },
    {
        "name": "sinkhorn",
        "dataset_class": model_sinkhorn.PlantDataset,
        "model_class": model_sinkhorn.HierarchicalPlantSurrogateNet,
        "loss_fn": model_sinkhorn.hierarchical_loss_function,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "module": model_sinkhorn
    }
]

os.umask(0)

def configure_output_file_logging(output_dir, run_label):
    """
    Route stdout/stderr to a persistent run log file when attached to a TTY.

    This prevents worker processes from stalling on terminal backpressure while
    keeping script usage unchanged (e.g. `python3 train_models.py`).
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{run_label}_terminal_output.log")

    # Line-buffered append keeps output streaming to disk for long runs.
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train plant surrogate models.")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME, 
                        help=f"Dataset folder (default: '{DATASET_NAME}')")
    parser.add_argument("--plant", type=str, default=PLANT_NAME,
                        help=f"Real plant name (default: '{PLANT_NAME}')")
    parser.add_argument("--replicates", type=int, default=NUM_REPLICATES, 
                        help=f"Replicates per model (default: {NUM_REPLICATES})")
    parser.add_argument("--no-multiprocessing", action="store_true", default=not USE_MULTIPROCESSING, 
                        help="Disable multiprocessing")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip automatic post-training evaluation step")
    return parser.parse_known_args()[0]

# --- UTILS ---

def prepare_real_plant_batch(real_bp, real_ep, max_points=50, use_multiprocessing=True):
    """
    Pre-processes real plant data into fixed-size tensors for batching.
    Moves data to shared memory if multiprocessing is enabled.
    
    Args:
        real_bp (list): List of daily branch point lists [Day -> Points].
        real_ep (list): List of daily end point lists.
    """
    num_days = len(real_bp)
    bp_batch = torch.zeros(1, num_days, max_points, 2)
    ep_batch = torch.zeros(1, num_days, max_points, 2)
    
    for day in range(num_days):
        # Fill tensors from lists, truncating to max_points
        if len(real_bp[day]) > 0:
            count = min(len(real_bp[day]), max_points)
            bp_batch[0, day, :count, :] = torch.tensor(real_bp[day][:count], dtype=torch.float32)
        if len(real_ep[day]) > 0:
            count = min(len(real_ep[day]), max_points)
            ep_batch[0, day, :count, :] = torch.tensor(real_ep[day][:count], dtype=torch.float32)
            
    if use_multiprocessing:
        bp_batch.share_memory_()
        ep_batch.share_memory_()
        
    return bp_batch, ep_batch

def train_one_epoch(model, loader, optimizer, real_bp_batch, real_ep_batch, 
                   real_bp_raw, real_ep_raw, model_config, training_log_csv=None, epoch_num=1):
    """
    Executes one full epoch of training.
    
    Steps:
    1. Zero Gradients.
    2. Forward Pass (Predict Cost).
    3. Forward Pass (Generate Structure - purely for regularization loss).
    4. Compute Composite Loss.
    5. Backprop & Optimize.
    6. Log progress.
    """
    model.train()
    total_loss = 0.0
    
    is_sinkhorn = "sinkhorn" in model_config["name"]
    loss_fn = model_config["loss_fn"]
    
    batch_idx = 0
    total_batches = len(loader)
    
    for params, costs in loader:
        optimizer.zero_grad()
        bs = params.size(0)
        
        # --- Normalization Note ---
        # The Model handles 'internal' normalization.
        # However, for Loss Calculation, we need targets (Costs) to be on the same scale 
        # as the normalized Model Output.
        
        # 1. Normalize Targets (True Costs)
        if hasattr(model, 'output_mean') and hasattr(model, 'output_std'):
             norm_costs = (costs - model.output_mean) / model.output_std
        else:
             norm_costs = costs
        
        # 2. Main Forward Pass (Prediction)
        curr_bp = real_bp_batch.repeat(bs, 1, 1, 1)
        curr_ep = real_ep_batch.repeat(bs, 1, 1, 1)
        
        pred_cost = model(params, curr_bp, curr_ep)
        
        # 3. Normalize Prediction (if model output was denormalized inside forward)
        if hasattr(model, 'output_mean') and hasattr(model, 'output_std'):
             pred_cost_norm = (pred_cost - model.output_mean) / model.output_std
        else:
             pred_cost_norm = pred_cost
        
        # 4. Structure Generation (Auxiliary)
        # We need the generated structure stats for regularization losses
        norm_params = (params - model.input_mean) / model.input_std
        bp_syn, bp_probs, ep_syn, ep_probs = None, None, None, None
        
        if hasattr(model, 'structure_gen'):
            bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen(norm_params)

        # 5. Compute Loss
        loss, _, _, _ = loss_fn(
            pred_cost_norm, norm_costs, bp_syn, bp_probs, ep_syn, ep_probs, params, real_bp_raw, real_ep_raw
        )
             
        # Stability: Clamp Sinkhorn Temperature
        if is_sinkhorn and hasattr(model, 'sinkhorn_net') and hasattr(model.sinkhorn_net, 'log_temperature'):
             model.sinkhorn_net.log_temperature.data.clamp_(min=-5.0)

        # 6. Optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 7. Log
        if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
            current_loss = loss.item()
            print(f"[{model_config['name']}] Epoch {epoch_num} Batch {batch_idx + 1}/{total_batches} Loss: {current_loss:.4f}", flush=True)
            if training_log_csv:
                with open(training_log_csv, "a") as f:
                    f.write(f"{epoch_num},{batch_idx + 1},{current_loss:.6f},,,,\n")
        
        batch_idx += 1
        
    return total_loss / len(loader)

def validate(model, loader, real_bp_batch, real_ep_batch):
    """
    Run evaluation on validation set.
    Returns standard metrics (R2, MSE, MAE, Accuracy).
    """
    model.eval()
    all_preds_list = []
    all_targets_list = []
    
    with torch.no_grad():
        for params, costs in loader:
            bs = params.size(0)
            curr_bp = real_bp_batch.repeat(bs, 1, 1, 1)
            curr_ep = real_ep_batch.repeat(bs, 1, 1, 1)
            
            # Forward
            pred = model(params, curr_bp, curr_ep) # Returns Real-Scale Cost
            
            all_preds_list.append(pred)
            all_targets_list.append(costs)
            
    all_preds = torch.cat(all_preds_list, dim=0).squeeze()
    all_targets = torch.cat(all_targets_list, dim=0).squeeze()
    
    # Metrics
    mse = F.mse_loss(all_preds, all_targets).item()
    mae = F.l1_loss(all_preds, all_targets).item()
    
    # Compute Normalized Loss equivalent to Training Loss for fair comparison
    val_loss = mse
    if hasattr(model, 'output_mean') and hasattr(model, 'output_std'):
         norm_preds = (all_preds - model.output_mean) / model.output_std
         norm_targets = (all_targets - model.output_mean) / model.output_std
         val_loss = F.mse_loss(norm_preds, norm_targets).item()
    
    # Relative Error
    rel_err = torch.abs(all_preds - all_targets) / (torch.abs(all_targets) + 1e-8)
    mean_rel_err = rel_err.mean().item()
    median_rel_err = rel_err.median().item()
    
    # R-Squared
    target_mean = torch.mean(all_targets)
    ss_tot = torch.sum((all_targets - target_mean) ** 2)
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Accuracy thresholds
    acc_1pct = (rel_err < 0.01).float().mean().item() * 100
    acc_5pct = (rel_err < 0.05).float().mean().item() * 100
            
    return {
        "loss": val_loss,
        "real_mse": mse,
        "mae": mae,
        "rel_err": mean_rel_err,
        "median_rel_err": median_rel_err,
        "r2": r2.item(),
        "acc_1pct": acc_1pct,
        "acc_5pct": acc_5pct
    }

def train_model_worker(config, run_dir, input_mean, input_std, output_mean, output_std, 
                      real_bp_batch, real_ep_batch, real_bp, real_ep, 
                      train_ds_args, val_ds_args, test_ds_args, replicate_id=1,
                      total_replicates=1, use_multiprocessing=True):
    """
    Train one model instance and save its outputs into a single folder.
    """

    # Seeding for reproducibility
    current_seed = 42 + replicate_id
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)

    model_base_name = config["name"]
    if total_replicates > 1:
        unique_run_name = f"{model_base_name}_Rep_{replicate_id}"
    else:
        unique_run_name = model_base_name
    print(f"\n[Worker] Starting {unique_run_name} (Seed: {current_seed})...")
    
    # Load Datasets (Local to process)
    PlantDataset = config["dataset_class"]
    train_ds = PlantDataset(*train_ds_args)
    val_ds = PlantDataset(*val_ds_args)
    test_ds = PlantDataset(*test_ds_args)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # Output Paths
    if total_replicates > 1:
        model_dir = os.path.join(run_dir, model_base_name, f"Rep_{replicate_id}")
    else:
        model_dir = os.path.join(run_dir, model_base_name)
    os.makedirs(model_dir, exist_ok=True)
    
    log_csv = os.path.join(model_dir, "training_log.csv")
    best_model_path = os.path.join(model_dir, "best_model.pt")
    metrics_json_path = os.path.join(model_dir, "metrics.json")
    
    # Initialize Model
    ModelClass = config["model_class"]
    model = ModelClass(
        input_mean=input_mean, 
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std
    )
    
    if use_multiprocessing:
        model.share_memory()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Optional Scheduler
    scheduler = None
    if "module" in config and hasattr(config["module"], "get_scheduler"):
        scheduler = config["module"].get_scheduler(optimizer)
    
    # Initialize Log File
    with open(log_csv, "w", buffering=1) as f:
        f.write("epoch,batch,train_loss,val_loss,val_mae,val_rel_err,val_rel_err_median,val_r2,val_acc_1pct,val_acc_5pct,time_sec\n")
        
    best_val_loss = float('inf')
    model_start_time = time.time()
    
    # Training Loop
    ep_start = time.time()
    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, 
            real_bp_batch, real_ep_batch, 
            real_bp, real_ep,
            config,
            training_log_csv=log_csv,
            epoch_num=epoch+1
        )
        
        val_metrics = validate(model, val_loader, real_bp_batch, real_ep_batch)
        
        # Scheduler Step
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
                if (epoch + 1) % 5 == 0:
                     print(f"[{unique_run_name}] LR: {optimizer.param_groups[0]['lr']:.2e}")
            else:
                scheduler.step()
                
        # Logging
        elapsed = time.time() - ep_start
        print(f"[{unique_run_name}] Epoch {epoch+1}/{config['epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f} | R2: {val_metrics['r2']:.4f}")
        
        with open(log_csv, "a") as f:
            f.write(f"{epoch+1},ALL,{train_loss:.6f},{val_metrics['loss']:.6f},{val_metrics['mae']:.6f},"
                    f"{val_metrics['rel_err']:.6f},{val_metrics['median_rel_err']:.6f},{val_metrics['r2']:.6f},"
                    f"{val_metrics['acc_1pct']:.2f},{val_metrics['acc_5pct']:.2f},{elapsed:.2f}\n")
        
        # Save Best Model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), best_model_path)
            
        ep_start = time.time()

    model_end_time = time.time()
    total_duration = model_end_time - model_start_time
    print(f"[Worker] Finished {unique_run_name}. Best Val Loss: {best_val_loss:.4f}")
    
    # Final evaluation on the best checkpoint.
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = validate(model, test_loader, real_bp_batch, real_ep_batch)
    
    # Canonical metrics artifact for summaries and downstream reporting.
    metrics_payload = {
        "model_name": model_base_name,
        "replicate": int(replicate_id),
        "has_replicates": total_replicates > 1,
        "metrics": {
            "test_norm_mse": float(test_metrics['loss']),
            "test_real_mse": float(test_metrics['real_mse']),
            "test_mae": float(test_metrics['mae']),
            "test_r2": float(test_metrics['r2']),
            "mean_relative_error": float(test_metrics['rel_err']),
            "median_relative_error": float(test_metrics['median_rel_err']),
            "accuracy_lt_1pct": float(test_metrics['acc_1pct']),
            "accuracy_lt_5pct": float(test_metrics['acc_5pct'])
        },
        "best_val_loss": float(best_val_loss),
        "total_training_duration_sec": float(total_duration),
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[Worker] Metrics saved for {unique_run_name}")

def aggregate_results(run_dir):
    """Build per-run and per-model summaries from canonical metrics.json files."""
    summary_path = os.path.join(run_dir, "summary_results.csv")
    summary_model_path = os.path.join(run_dir, "summary_by_model.csv")
    records = []
    
    print(f"Aggregating results from {run_dir}...")
    for root, _, files in os.walk(run_dir):
        if "metrics.json" not in files:
            continue

        metrics_path = os.path.join(root, "metrics.json")
        try:
            with open(metrics_path, "r") as f:
                payload = json.load(f)
            m = payload.get("metrics", {})
            records.append({
                "path": os.path.relpath(root, run_dir),
                "model_name": payload.get("model_name", "unknown"),
                "replicate": int(payload.get("replicate", 1)),
                "test_norm_mse": float(m.get("test_norm_mse", np.nan)),
                "test_real_mse": float(m.get("test_real_mse", np.nan)),
                "test_mae": float(m.get("test_mae", np.nan)),
                "test_r2": float(m.get("test_r2", np.nan)),
                "mean_relative_error": float(m.get("mean_relative_error", np.nan)),
                "median_relative_error": float(m.get("median_relative_error", np.nan)),
                "accuracy_lt_1pct": float(m.get("accuracy_lt_1pct", np.nan)),
                "accuracy_lt_5pct": float(m.get("accuracy_lt_5pct", np.nan)),
                "best_val_loss": float(payload.get("best_val_loss", np.nan)),
                "total_training_duration_sec": float(payload.get("total_training_duration_sec", np.nan)),
            })
        except Exception as e:
            print(f"Skipping {metrics_path}: {e}")

    if not records:
        print("No metrics.json files found.")
        return

    records.sort(key=lambda x: (x["model_name"], x["replicate"]))

    detail_fields = [
        "model_name", "replicate", "path", "test_norm_mse", "test_real_mse", "test_mae",
        "test_r2", "mean_relative_error", "median_relative_error",
        "accuracy_lt_1pct", "accuracy_lt_5pct", "best_val_loss", "total_training_duration_sec"
    ]
    with open(summary_path, "w") as f:
        f.write(",".join(detail_fields) + "\n")
        for r in records:
            f.write(",".join(str(r.get(k, "")) for k in detail_fields) + "\n")

    # Model-level aggregation (mean/std) across replicates for easy retrieval.
    grouped = {}
    for r in records:
        grouped.setdefault(r["model_name"], []).append(r)

    metric_keys = [
        "test_norm_mse", "test_real_mse", "test_mae", "test_r2",
        "mean_relative_error", "median_relative_error",
        "accuracy_lt_1pct", "accuracy_lt_5pct", "best_val_loss", "total_training_duration_sec"
    ]

    agg_fields = ["model_name", "n_runs"]
    for k in metric_keys:
        agg_fields.extend([f"{k}_mean", f"{k}_std"])

    with open(summary_model_path, "w") as f:
        f.write(",".join(agg_fields) + "\n")
        for model_name, rows in sorted(grouped.items()):
            out = [model_name, str(len(rows))]
            for k in metric_keys:
                vals = np.array([rr[k] for rr in rows], dtype=np.float64)
                out.append(f"{np.mean(vals):.8f}")
                out.append(f"{np.std(vals):.8f}")
            f.write(",".join(out) + "\n")

    print(f"Summary saved to {summary_path}")
    print(f"Model summary saved to {summary_model_path}")

def run_post_training_evaluation(run_dir, input_mean, input_std, output_mean, output_std,
                                 real_bp_batch, real_ep_batch, test_csv):
    """
    Runs evaluate_models on all trained replicates and writes figure artifacts
    directly into each replicate directory under Training Data.
    """
    try:
        import evaluate_models
    except Exception as e:
        print(f"Post-training evaluation skipped: could not import evaluate_models ({e})")
        return

    data = {
        "input_mean": input_mean,
        "input_std": input_std,
        "output_mean": output_mean,
        "output_std": output_std,
        "real_bp_batch": real_bp_batch,
        "real_ep_batch": real_ep_batch,
        "test_csv": test_csv,
    }

    print("\nStarting post-training evaluation (figures -> Training Data replicate folders)...")
    evaluated = 0
    for root, _, files in os.walk(run_dir):
        if "best_model.pt" in files and "training_log.csv" in files:
            evaluate_models.evaluate_run(
                root,
                data,
                output_path=root,
                include_metrics_artifacts=False
            )
            evaluated += 1

    if evaluated == 0:
        print("Post-training evaluation: no replicate folders found.")
    else:
        print(f"Post-training evaluation complete. Replicates evaluated: {evaluated}")

def main():
    args = parse_arguments()
    
    # Configure Plant
    plant_comparison_nn.real_plant_name = args.plant
    if not plant_comparison_nn.plant_images_path.endswith(os.sep):
        plant_comparison_nn.plant_images_path += os.sep
    plant_comparison_nn.plant_image_path = plant_comparison_nn.plant_images_path + args.plant
    
    dataset_run = args.dataset
    replicates_count = args.replicates
    use_multi = not args.no_multiprocessing
    run_eval_after_training = not args.skip_evaluation
    
    # Determine Output Directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, "Datasets", dataset_run)
    output_root = os.path.join(base_dir, "Training Data")
    
    date_str = datetime.now().strftime("%m%d%y")
    run_name = f"Run_{date_str}"
    
    # Handle Name Collisions
    counter = 0
    candidate = run_name
    while os.path.exists(os.path.join(output_root, candidate)):
        counter += 1
        candidate = f"{run_name}_{counter}"
    
    run_output_dir = os.path.join(output_root, candidate)
    os.makedirs(run_output_dir, exist_ok=True)

    log_path = configure_output_file_logging(run_output_dir, candidate)
    
    print(f"=== Training Pipeline: {date_str} ===")
    print(f"Dataset: {dataset_run}")
    print(f"Output: {run_output_dir}")
    print(f"Replicates: {replicates_count}")
    print(f"Terminal Log: {log_path}")
    
    # Load Data & Stats
    print("Loading Real Plants...")
    real_bp, real_ep = read_real_plants()
    real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep, use_multiprocessing=use_multi)
    
    train_csv = os.path.join(datasets_dir, "Train.csv")
    val_csv = os.path.join(datasets_dir, "Validation.csv")
    test_csv = os.path.join(datasets_dir, "Test.csv")
    
    # Calculate Stats for Normalization (Using Baseline class as helper)
    ds_temp = model_hungarian.PlantDataset(train_csv)
    input_mean = ds_temp.params.mean(axis=0)
    input_std = ds_temp.params.std(axis=0) + 1e-8
    output_mean = ds_temp.costs.mean()
    output_std = ds_temp.costs.std() + 1e-8
    
    # Save Description
    with open(os.path.join(run_output_dir, "description.txt"), "w") as f:
        f.write(f"Run ID: {candidate}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_run}\n")
        f.write(f"Ref Plant: {args.plant}\n")
        f.write("========================================\n")
        f.write("Configuration:\n")
        f.write(f"  Replicates: {replicates_count}\n")
        f.write(f"  Epochs: {NUM_EPOCHS}\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {LEARNING_RATE}\n")
        f.write("========================================\n")
        f.write("Statistics (Normalization):\n")
        # Ensure full numpy arrays are printed
        np_print_opts = np.get_printoptions()
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        f.write(f"  Input Mean: {input_mean}\n")
        f.write(f"  Input Std:  {input_std}\n")
        np.set_printoptions(**np_print_opts) # Restore options
        f.write(f"  Cost Mean:  {output_mean:.6f}\n")
        f.write(f"  Cost Std:   {output_std:.6f}\n")
        f.write("========================================\n")
        f.write("Models:\n")
        for m in MODELS_TO_TRAIN:
            f.write(f"  - {m['name']} (Loss: {m['loss_fn'].__name__})\n")
        
    # Launch Workers
    train_ds_args = (train_csv, None)
    val_ds_args = (val_csv, None)
    test_ds_args = (test_csv, None)
    
    pipeline_start = time.time()
    
    if use_multi:
        processes = []
        for rep in range(1, replicates_count + 1):
            for config in MODELS_TO_TRAIN:
                p = mp.Process(target=train_model_worker, args=(
                    config, run_output_dir, input_mean, input_std, output_mean, output_std,
                    real_bp_batch, real_ep_batch, real_bp, real_ep,
                    train_ds_args, val_ds_args, test_ds_args, rep, replicates_count, True
                ))
                p.start()
                processes.append(p)
        for p in processes:
            p.join()
    else:
        for rep in range(1, replicates_count + 1):
            for config in MODELS_TO_TRAIN:
                train_model_worker(
                    config, run_output_dir, input_mean, input_std, output_mean, output_std,
                    real_bp_batch, real_ep_batch, real_bp, real_ep,
                    train_ds_args, val_ds_args, test_ds_args, rep, replicates_count, False
                )
                
    full_duration = time.time() - pipeline_start
    print(f"\nPipeline Finished. Duration: {full_duration:.2f}s")
    
    try:
        aggregate_results(run_output_dir)
    except Exception as e:
        print(f"Aggregation failed: {e}")

    if run_eval_after_training:
        try:
            run_post_training_evaluation(
                run_output_dir,
                input_mean,
                input_std,
                output_mean,
                output_std,
                real_bp_batch,
                real_ep_batch,
                test_csv,
            )
        except Exception as e:
            print(f"Post-training evaluation failed: {e}")
    else:
        print("Post-training evaluation skipped (--skip-evaluation).")

if __name__ == "__main__":
    main()
