"""
Universal Training Pipeline for Plant Surrogate Models.
Trains multiple model architectures on a specified dataset run using PyTorch.
Supports multiprocessing, experiment replication, and automatic logging.
"""

import os
import time
import argparse
import shutil
import numpy as np
from datetime import datetime
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.nn.functional as F

# --- Import Model Architectures ---
# Each module should provide: PlantDataset, HierarchicalPlantSurrogateNet, hierarchical_loss_function
import surrogate_nn_dataset as baseline_model
import surrogate_nn_dataset_sinkhorn as sinkhorn_model
import surrogate_nn_dataset_scheduler as baseline_model_scheduler
import surrogate_nn_dataset_sinkhorn_scheduler as sinkhorn_model_scheduler
from plant_comparison_nn import read_real_plants

# --- CONFIGURATION ---
DEFAULT_DATASET = "Run 031026"

# Define the models to be trained in this session
MODELS_TO_TRAIN = [
    {
        "name": "baseline_mlp_scheduler",
        "dataset_class": baseline_model_scheduler.PlantDataset,
        "model_class": baseline_model_scheduler.HierarchicalPlantSurrogateNet,
        "loss_fn": baseline_model_scheduler.hierarchical_loss_function,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
        "module": baseline_model_scheduler  # Used to access get_scheduler
    },
    {
        "name": "sinkhorn_scheduler",
        "dataset_class": sinkhorn_model_scheduler.PlantDataset,
        "model_class": sinkhorn_model_scheduler.HierarchicalPlantSurrogateNet,
        "loss_fn": sinkhorn_model_scheduler.hierarchical_loss_function,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
        "module": sinkhorn_model_scheduler
    }
]

# Set umask to 0 to ensure created files are accessible by host (when running in Docker)
os.umask(0)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train plant surrogate models.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, 
                        help=f"Name of the dataset folder in Datasets/ (default: '{DEFAULT_DATASET}')")
    parser.add_argument("--replicates", type=int, default=1, 
                        help="Number of replicates to run for each model configuration (default: 1)")
    parser.add_argument("--no-multiprocessing", action="store_true", 
                        help="Disable multiprocessing (run sequentially)")
    return parser.parse_known_args()[0]

# --- UTILS ---

def prepare_real_plant_batch(real_bp, real_ep, max_points=50, use_multiprocessing=True):
    """
    Convert real plant data (list of arrays) to fixed-size tensors.
    Optionally places tensors in shared memory for multiprocessing.
    """
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
            
    if use_multiprocessing:
        bp_batch.share_memory_()
        ep_batch.share_memory_()
        
    return bp_batch, ep_batch

def train_one_epoch(model, loader, optimizer, real_bp_batch, real_ep_batch, 
                   real_bp_raw, real_ep_raw, model_config, training_log_csv=None, epoch_num=1):
    """Executes one epoch of training."""
    model.train()
    total_loss = 0.0
    
    is_sinkhorn = "sinkhorn" in model_config["name"]
    loss_fn = model_config["loss_fn"]
    
    batch_idx = 0
    total_batches = len(loader)
    
    for params, costs in loader:
        optimizer.zero_grad()
        bs = params.size(0)
        
        # 1. Normalize targets (Costs) used for loss calculation
        # The model usually outputs 'Real' scale, so we normalize for numerical stability in loss
        if hasattr(model, 'output_mean') and hasattr(model, 'output_std'):
             norm_costs = (costs - model.output_mean) / model.output_std
        else:
             norm_costs = costs
        
        # 2. Forward Pass
        # Pass expanded batches of real plant structure for comparison
        curr_bp = real_bp_batch.repeat(bs, 1, 1, 1)
        curr_ep = real_ep_batch.repeat(bs, 1, 1, 1)
        
        pred_cost = model(params, curr_bp, curr_ep)
        
        # 3. Normalize prediction for loss calculation 
        # (Assuming model returns Real Scale cost)
        if hasattr(model, 'output_mean') and hasattr(model, 'output_std'):
             pred_cost_norm = (pred_cost - model.output_mean) / model.output_std
        else:
             pred_cost_norm = pred_cost
        
        # 4. Structure Generation (Auxiliary Output for Regularization)
        # Normalize inputs for the structure generator
        norm_params = (params - model.input_mean) / model.input_std
        
        bp_syn, bp_probs, ep_syn, ep_probs = None, None, None, None
        if hasattr(model, 'structure_gen'):
            bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen(norm_params)

        # 5. Calculate Loss
        loss, _, _, _ = loss_fn(
            pred_cost_norm, norm_costs, bp_syn, bp_probs, ep_syn, ep_probs, real_bp_raw, real_ep_raw
        )
             
        # Sinkhorn Specific: Clamp temperature to prevent numerical instability
        if is_sinkhorn and hasattr(model, 'sinkhorn_net') and hasattr(model.sinkhorn_net, 'log_temperature'):
             model.sinkhorn_net.log_temperature.data.clamp_(min=-5.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
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
    Validate model performance on a dataset loader.
    Returns dictionary of metrics including R2, MSE, MAE.
    """
    model.eval()
    all_preds_list = []
    all_targets_list = []
    
    with torch.no_grad():
        for params, costs in loader:
            bs = params.size(0)
            curr_bp = real_bp_batch.repeat(bs, 1, 1, 1)
            curr_ep = real_ep_batch.repeat(bs, 1, 1, 1)
            
            pred = model(params, curr_bp, curr_ep)
            
            all_preds_list.append(pred)
            all_targets_list.append(costs)
            
    all_preds = torch.cat(all_preds_list, dim=0).squeeze()
    all_targets = torch.cat(all_targets_list, dim=0).squeeze()
    
    # Calculate Metrics (Real Scale)
    mse = F.mse_loss(all_preds, all_targets).item()
    mae = F.l1_loss(all_preds, all_targets).item()
    
    # Normalized Loss (for comparison with training loss)
    val_loss = mse
    if hasattr(model, 'output_mean') and hasattr(model, 'output_std'):
         norm_preds = (all_preds - model.output_mean) / model.output_std
         norm_targets = (all_targets - model.output_mean) / model.output_std
         val_loss = F.mse_loss(norm_preds, norm_targets).item()
    
    # Relative Error
    rel_err = torch.abs(all_preds - all_targets) / (torch.abs(all_targets) + 1e-8)
    mean_rel_err = rel_err.mean().item()
    median_rel_err = rel_err.median().item()
    
    # R2 Score
    target_mean = torch.mean(all_targets)
    ss_tot = torch.sum((all_targets - target_mean) ** 2)
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Accuracy
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
                      train_ds_args, val_ds_args, test_ds_args, replicate_id=1, use_multiprocessing=True):
    """
    Worker function to train a single model replicate.
    Designed to run in a separate process.
    """
    model_base_name = config["name"]
    unique_run_name = f"{model_base_name}_Rep_{replicate_id}"
    print(f"\n[Worker] Starting {unique_run_name}...")
    
    # Re-instantiate Datasets inside worker (avoid pickling issues with DataLoaders)
    PlantDataset = config["dataset_class"]
    train_ds = PlantDataset(*train_ds_args)
    val_ds = PlantDataset(*val_ds_args)
    test_ds = PlantDataset(*test_ds_args)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # Setup Output Directories
    model_dir = os.path.join(run_dir, model_base_name, f"Rep_{replicate_id}")
    os.makedirs(model_dir, exist_ok=True)
    
    log_csv = os.path.join(model_dir, "training_log.csv")
    best_model_path = os.path.join(model_dir, "best_model.pt")
    results_txt = os.path.join(model_dir, "test_results.txt")
    
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
    
    # Initialize Scheduler
    scheduler = None
    if "module" in config and hasattr(config["module"], "get_scheduler"):
        scheduler = config["module"].get_scheduler(optimizer)
        print(f"[{unique_run_name}] Scheduler Active: {type(scheduler).__name__}")
    
    # Init Log File
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
    
    # Final Evaluation (Load Best Model)
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = validate(model, test_loader, real_bp_batch, real_ep_batch)
    
    # Save Results
    with open(results_txt, "w") as f:
        f.write(f"Model: {model_base_name} (Replicate {replicate_id})\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("=== Test Results ===\n")
        f.write(f"Test Norm Loss (MSE): {test_metrics['loss']:.4f}\n")
        f.write(f"Test Real MSE: {test_metrics['real_mse']:.4f}\n")
        f.write(f"Test R2 Score: {test_metrics['r2']:.4f}\n")
        f.write(f"Mean Relative Error: {test_metrics['rel_err']:.4f}\n")
        f.write(f"Median Relative Error: {test_metrics['median_rel_err']:.4f}\n")
        f.write(f"Accuracy (<1% error): {test_metrics['acc_1pct']:.2f}%\n")
        f.write(f"Accuracy (<5% error): {test_metrics['acc_5pct']:.2f}%\n")
        f.write(f"Total Training Duration: {total_duration:.2f}s\n")

    print(f"[Worker] Results saved for {unique_run_name}")

def aggregate_results(run_dir):
    """Compiles test results from all replicates into a single CSV."""
    summary_path = os.path.join(run_dir, "summary_results.csv")
    records = []
    
    print(f"Aggregating results from {run_dir}...")
    for root, dirs, files in os.walk(run_dir):
        if "test_results.txt" in files:
            res_path = os.path.join(root, "test_results.txt")
            record = {"Path": os.path.relpath(root, run_dir)}
            try:
                with open(res_path, "r") as f:
                    for line in f:
                        if ":" in line:
                            key, val = line.split(":", 1)
                            key, val = key.strip(), val.strip()
                            clean_val = val.rstrip("%s") # Remove units like %, s
                            try:
                                record[key] = float(clean_val)
                            except ValueError:
                                record[key] = val
                                
                # Parse Model Name/Replicate from file content or path if needed
                if "Model" in record:
                    full = str(record["Model"])
                    if "(Replicate" in full:
                        parts = full.split("(Replicate")
                        record["Model Name"] = parts[0].strip()
                        record["Replicate"] = int(parts[1].replace(")", "").strip())
                    else:
                        record["Model Name"] = full
                        record["Replicate"] = 1
                records.append(record)
            except Exception as e:
                print(f"Skipping {res_path}: {e}")

    if not records:
        print("No results found.")
        return

    records.sort(key=lambda x: (x.get("Model Name", ""), x.get("Replicate", 0)))
    
    fieldnames = ["Model Name", "Replicate", "Test R2 Score", "Test Real MSE", 
                  "Test Norm Loss (MSE)", "Mean Relative Error", "Accuracy (<1% error)", 
                  "Total Training Duration"]
    
    # Add extra keys found
    known_keys = set(fieldnames + ["Path"])
    extra_keys = sorted([k for k in records[0].keys() if k not in known_keys])
    fieldnames.extend(extra_keys)
    
    with open(summary_path, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for r in records:
            row = [str(r.get(k, "")) for k in fieldnames]
            f.write(",".join(row) + "\n")
            
    print(f"Summary saved to {summary_path}")

def main():
    args = parse_arguments()
    dataset_run = args.dataset
    replicates_count = args.replicates
    use_multi = not args.no_multiprocessing
    
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
    
    print(f"=== Training Pipeline: {date_str} ===")
    print(f"Dataset: {dataset_run}")
    print(f"Output: {run_output_dir}")
    print(f"Replicates: {replicates_count}")
    
    # Load Data & Stats
    print("Loading Real Plants...")
    real_bp, real_ep = read_real_plants()
    real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep, use_multi)
    
    train_csv = os.path.join(datasets_dir, "Train.csv")
    val_csv = os.path.join(datasets_dir, "Validation.csv")
    test_csv = os.path.join(datasets_dir, "Test.csv")
    
    # Calculate Stats for Normalization (Using Baseline class as helper)
    ds_temp = baseline_model.PlantDataset(train_csv)
    input_mean = ds_temp.params.mean(axis=0)
    input_std = ds_temp.params.std(axis=0) + 1e-8
    output_mean = ds_temp.costs.mean()
    output_std = ds_temp.costs.std() + 1e-8
    
    # Save Description
    with open(os.path.join(run_output_dir, "description.txt"), "w") as f:
        f.write(f"Run: {candidate}\nDataset: {dataset_run}\n")
        f.write(f"Stats - Input Mean: {input_mean[:3]}...\n") # Abbreviated
        f.write(f"Stats - Cost Mean: {output_mean:.2f}, Std: {output_std:.2f}\n")
        
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
                    train_ds_args, val_ds_args, test_ds_args, rep, True
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
                    train_ds_args, val_ds_args, test_ds_args, rep, False
                )
                
    full_duration = time.time() - pipeline_start
    print(f"\nPipeline Finished. Duration: {full_duration:.2f}s")
    
    try:
        aggregate_results(run_output_dir)
    except Exception as e:
        print(f"Aggregation failed: {e}")

if __name__ == "__main__":
    main()
