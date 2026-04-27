"""
End-to-End Hyperparameter Tuning and Convergence Script.

Replaces standard validation-loss monitoring with true surrogate optimization utility.
Tunes models by evaluating how well they guide a lightweight optimizer to a true L-system structure.
"""

import os
import sys
import time
import argparse
import itertools
import numpy as np
import random
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp

# Import model definitions
try:
    import model_base
    import model_sinkhorn as sinkhorn_model
    import model_mlp as benchmark_mlp
    import utils_nn

    # Import training worker utilities
    from train_models import (
        train_one_epoch,
        validate,
        aggregate_results,
        prepare_real_plant_batch,
        configure_output_file_logging,
    )

    # Import optimizer utilities
    from optimizer_script import (
        optimize_params_for_model,
        run_simulation_verification,
        evaluate_real_cost,
        cleanup_empty_verify_dirs
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    import sys
    sys.exit(1)

# Keep generated files/dirs editable and removable.
os.umask(0)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def attach_transcript(log_path):
    transcript = open(log_path, "a", buffering=1)
    sys.stdout = TeeStream(sys.stdout, transcript)
    sys.stderr = TeeStream(sys.stderr, transcript)
    return transcript

# Define Hyperparameter Search Space
DEFAULT_HP_GRID = {
    "learning_rate": [1e-3, 5e-4],
    "batch_size": [16, 32],
}

MODEL_BASE_CONFIGS = {
    "sinkhorn": {
        "dataset_class": model_base.PlantDataset,
        "model_class": sinkhorn_model.HierarchicalPlantSurrogateNet,
        "loss_fn": model_base.hierarchical_loss_function,
        "module": sinkhorn_model,
        "internal_type": "sinkhorn"
    },
    "sinkhorn_full": {
        "dataset_class": model_base.PlantDataset,
        "model_class": sinkhorn_model.HierarchicalPlantSurrogateNet,
        "loss_fn": model_base.hierarchical_loss_function,
        "module": sinkhorn_model,
        "internal_type": "sinkhorn",
        "model_kwargs": {"use_encoder": True, "use_scaler": True, "use_aggregator": True}
    },
    "sinkhorn_no_encoder": {
        "dataset_class": model_base.PlantDataset,
        "model_class": sinkhorn_model.HierarchicalPlantSurrogateNet,
        "loss_fn": model_base.hierarchical_loss_function,
        "module": sinkhorn_model,
        "internal_type": "sinkhorn",
        "model_kwargs": {"use_encoder": False, "use_scaler": True, "use_aggregator": True}
    },
    "sinkhorn_no_scaler": {
        "dataset_class": model_base.PlantDataset,
        "model_class": sinkhorn_model.HierarchicalPlantSurrogateNet,
        "loss_fn": model_base.hierarchical_loss_function,
        "module": sinkhorn_model,
        "internal_type": "sinkhorn",
        "model_kwargs": {"use_encoder": True, "use_scaler": False, "use_aggregator": True}
    },
    "sinkhorn_no_aggregator": {
        "dataset_class": model_base.PlantDataset,
        "model_class": sinkhorn_model.HierarchicalPlantSurrogateNet,
        "loss_fn": model_base.hierarchical_loss_function,
        "module": sinkhorn_model,
        "internal_type": "sinkhorn",
        "model_kwargs": {"use_encoder": True, "use_scaler": True, "use_aggregator": False}
    },
    "mlp": {
        "dataset_class": benchmark_mlp.PlantDataset,
        "model_class": benchmark_mlp.BenchmarkSurrogateNet,
        "loss_fn": benchmark_mlp.benchmark_loss_function,
        "module": benchmark_mlp,
        "internal_type": "mlp"
    }
}

class LightweightOptArgs:
    """Mock arguments to pass into the optimizer_script functions."""
    def __init__(self, restarts=3, steps=250, lr=0.1, param_jitter_std=0.01, log_every=1000):
        self.restarts = restarts
        self.steps = steps
        self.lr = lr
        self.param_jitter_std = param_jitter_std
        self.log_every = log_every

def load_data(dataset_name, plant_name):
    """Loads dataset stats and real plant structure for comparison."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "Datasets", dataset_name)
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset not found at {dataset_dir}")
        return None

    train_csv = os.path.join(dataset_dir, "Train.csv")
    val_csv = os.path.join(dataset_dir, "Validation.csv")
    test_csv = os.path.join(dataset_dir, "Test.csv")
    
    import pandas as pd
    try:
        df = pd.read_csv(train_csv)
    except FileNotFoundError:
        print(f"Error: Could not find Train.csv in {dataset_dir}")
        return None
    
    params = df.iloc[:, 2:15].values
    costs = df.iloc[:, 1].values
    
    input_mean = np.mean(params, axis=0)
    input_std = np.std(params, axis=0) + 1e-6
    output_mean = np.mean(costs)
    output_std = np.std(costs) + 1e-6
    
    print(f"Loading Real Plant Structure '{plant_name}'...")
    try:
        real_bp, real_ep = utils_nn.read_real_plants()
        real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep, use_multiprocessing=False)
    except Exception as e:
        print(f"Error reading real plant: {e}")
        print("Using dummy plant structure (zeros)")
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

def end_to_end_tuning_worker(config, run_dir, data, train_ds_args, val_ds_args, test_ds_args, args, replicate_id=1):
    """
    Train a model instance. Every epoch, evaluate its optimization utility via a lightweight surrogate run.
    Early stopping is based strictly on the Real LPFG Verification Cost.
    """
    # Reproducibility
    current_seed = 42 + replicate_id
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    random.seed(current_seed)

    unique_run_name = f"{config['name']}_Rep_{replicate_id}"
    model_dir = os.path.join(run_dir, config["name"], f"Rep_{replicate_id}")
    os.makedirs(model_dir, exist_ok=True)
    
    transcript_path = os.path.join(model_dir, "terminal.log")
    transcript_handle = attach_transcript(transcript_path)

    def log_status(message):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {message}")

    log_status(f"[Worker] Starting {unique_run_name} | Seed={current_seed} | Plant={args.plant} | Dataset={args.dataset} | LR={config['learning_rate']} | Batch={config['batch_size']} | Fraction={args.dataset_fraction}")
    log_status(f"[Worker] Transcript: {transcript_path}")

    PlantDataset = config["dataset_class"]
    train_ds = PlantDataset(*train_ds_args)
    val_ds = PlantDataset(*val_ds_args)
    test_ds = PlantDataset(*test_ds_args)

    # Convergence testing with reduced training subset
    if args.dataset_fraction < 1.0:
        num_samples = int(len(train_ds) * args.dataset_fraction)
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        train_ds = Subset(train_ds, indices[:num_samples])
        print(f"[{unique_run_name}] Convergence test: using {num_samples} training samples ({args.dataset_fraction*100:.1f}%)")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    log_csv = os.path.join(model_dir, "tuning_log.csv")
    best_model_path = os.path.join(model_dir, "best_model.pt")
    log_status(f"[Worker] Active output dir: {model_dir}")
    
    # Initialize Model
    ModelClass = config["model_class"]
    model_kwargs = config.get("model_kwargs", {})
    model = ModelClass(
        input_mean=data["input_mean"], input_std=data["input_std"],
        output_mean=data["output_mean"], output_std=data["output_std"],
        **model_kwargs
    )
        
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    opt_args = LightweightOptArgs(restarts=args.opt_restarts, steps=args.opt_steps)
    model_opt_type = config.get("internal_type", "mlp")

    # Write training log CSV header
    with open(log_csv, "w") as f:
        f.write("epoch,train_loss,val_mse,val_r2,opt_surrogate_cost,opt_real_lpfg_cost,time_sec\n")
        
    best_lpfg_cost = float('inf')
    best_val_r2 = -float('inf')
    patience_counter = 0
    total_epochs_trained = 0
    ep_start = time.time()
    
    for epoch in range(args.epochs):
        total_epochs_trained += 1
        
        # 1. Standard Training Epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, 
            data["real_bp_batch"], data["real_ep_batch"], 
            data["real_bp"], data["real_ep"],
            config, training_log_csv=None, epoch_num=epoch+1
        )
        
        # 2. Standard Validation
        val_metrics = validate(model, val_loader, data["real_bp_batch"], data["real_ep_batch"])
        
        # 3. Lightweight End-to-End Evaluation
        model.eval()
        model.to('cpu')
        
        best_params, opt_cost, _ = optimize_params_for_model(
            model, model_opt_type, data["real_bp_batch"].to('cpu'), data["real_ep_batch"].to('cpu'), opt_args
        )
        
        # Verify physically with LPFG
        verify_dir = os.path.join(model_dir, "Verify")
        os.makedirs(verify_dir, exist_ok=True)
        syn_bp, syn_ep = run_simulation_verification(best_params, output_dir=verify_dir)
        
        real_lpfg_cost = float('inf')
        if syn_bp is not None:
            real_lpfg_cost = evaluate_real_cost(syn_bp, syn_ep, data["real_bp"], data["real_ep"])
        
        elapsed = time.time() - ep_start
        print(f"[{unique_run_name}] Ep {epoch+1}/{args.epochs} | "
              f"Val R2: {val_metrics['r2']:.4f} (Best: {max(best_val_r2, val_metrics['r2']):.4f}) | "
              f"LPFG Cost: {real_lpfg_cost:.4f} (Best: {min(best_lpfg_cost, real_lpfg_cost):.4f})")
        
        with open(log_csv, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_metrics['loss']:.6f},{val_metrics['r2']:.6f},"
                    f"{opt_cost:.6f},{real_lpfg_cost:.6f},{elapsed:.2f}\n")

        # 4. Early Stopping strictly on Real LPFG Cost
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']

        if real_lpfg_cost < best_lpfg_cost:
            best_lpfg_cost = real_lpfg_cost
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[{unique_run_name}] * New best LPFG score achieved! Saving model.")
        else:
            patience_counter += 1
            print(f"[{unique_run_name}] LPFG score did not improve. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"[{unique_run_name}] Early stopping triggered at epoch {epoch+1}!")
                break
                
        ep_start = time.time()
        
    cleanup_empty_verify_dirs(model_dir)
    log_status(f"[Worker] Finished {unique_run_name}. Best LPFG Score: {best_lpfg_cost:.4f} | Best R2: {best_val_r2:.4f}")
    transcript_handle.flush()

    # Log final summary line for this replicate so we don't have to rewrite aggregate_results() completely
    summary_csv = os.path.join(run_dir, "tuning_summary.csv")
    write_header = not os.path.exists(summary_csv)
    
    with open(summary_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "replicate", "learning_rate", "batch_size", 
                             "dataset_fraction", "epochs_trained", "best_val_r2", "best_lpfg_cost"])
        writer.writerow([
            config["name"], replicate_id, config["learning_rate"], config["batch_size"],
            args.dataset_fraction, total_epochs_trained, f"{best_val_r2:.6f}", f"{best_lpfg_cost:.6f}"
        ])

def main():
    parser = argparse.ArgumentParser(description="End-to-End Surrogate Tuning")
    parser.add_argument("--dataset", type=str, default="first_hp_tuning", help="Dataset folder name")
    parser.add_argument("--plant", type=str, default="Plant_063-32", help="Target plant structure (must match utils_nn)")
    parser.add_argument("--models", type=str, nargs="+", default=["mlp", "sinkhorn"], help="Space-separated list of models")
    parser.add_argument("--replicates", type=int, default=2, help="Replicates per config")
    parser.add_argument("--dataset-fraction", type=float, default=1.0, help="Fraction of training subset to test convergence")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs allowed before forced stop")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping based on real LPFG Cost")
    parser.add_argument("--opt-restarts", type=int, default=3, help="Optimizer restarts per evaluation")
    parser.add_argument("--opt-steps", type=int, default=200, help="Optimizer steps per evaluation restart")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3, 5e-4], help="Space-separated list of learning rates to test")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32], help="Space-separated list of batch sizes to test")
    args = parser.parse_args()

    print("--- End-to-End Tuning & Convergence ---")
    
    data = load_data(args.dataset, args.plant)
    if data is None: 
        return
        
    train_ds_args = (data["train_csv"], None)
    val_ds_args = (data["val_csv"], None)
    test_ds_args = (data["test_csv"], None)
    
    keys = ["learning_rate", "batch_size"]
    values = [args.learning_rates, args.batch_sizes]
    combinations = list(itertools.product(*values))
    
    run_configs = []
    
    # args.models is already a list due to nargs="+"
    target_models = [m.strip().lower() for m in args.models]
    
    alias_map = {
        "mlp": "mlp", 
        "sinkhorn": "sinkhorn", 
        "hungarian": "baseline_hierarchical"
    }

    for input_name in target_models:
        model_name = alias_map.get(input_name, input_name)
        if model_name not in MODEL_BASE_CONFIGS:
            print(f"Skipping unknown model: {model_name}")
            continue
            
        base_config = MODEL_BASE_CONFIGS[model_name]
        for lr, bs in combinations:
            config = base_config.copy()
            config["learning_rate"] = lr
            config["batch_size"] = bs
            config["name"] = f"{input_name}_lr{lr}_bs{bs}"
            run_configs.append(config)
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Store in "Hyperparameter Tuning" directory instead of intertwining with datasets
    base_tuning_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hyperparameter Tuning")
    models_str = "-".join(target_models)
    dir_name = f"Tuning_{timestamp}_{args.dataset}_{models_str}"
    tuning_dir = os.path.join(base_tuning_dir, dir_name)
    os.makedirs(tuning_dir, exist_ok=True)
    transcript_path = os.path.join(tuning_dir, "terminal.log")
    transcript_handle = attach_transcript(transcript_path)
    print(f"[Run] Transcript: {transcript_path}")
    
    # Save a highly detailed description of this entire setup
    with open(os.path.join(tuning_dir, "description.txt"), "w") as f:
        f.write(f"Run ID: Tuning_Run_{timestamp}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Ref Plant: {args.plant}\n")
        f.write("========================================\n")
        f.write("Configuration:\n")
        f.write(f"  Replicates: {args.replicates}\n")
        f.write(f"  Dataset Fraction: {args.dataset_fraction}\n")
        f.write(f"  Early Stopping Patience: {args.patience} epochs\n")
        f.write(f"  Max Epochs: {args.epochs}\n")
        f.write(f"  Eval Optimizer Restarts: {args.opt_restarts}\n")
        f.write(f"  Eval Optimizer Steps: {args.opt_steps}\n")
        f.write("========================================\n")
        f.write("Statistics (Normalization):\n")
        np_print_opts = np.get_printoptions()
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        f.write(f"  Input Mean: {data['input_mean']}\n")
        f.write(f"  Input Std:  {data['input_std']}\n")
        np.set_printoptions(**np_print_opts)
        f.write(f"  Cost Mean:  {data['output_mean']:.6f}\n")
        f.write(f"  Cost Std:   {data['output_std']:.6f}\n")
        f.write("========================================\n")

    total_jobs = len(run_configs) * args.replicates
    completed_jobs = 0

    print(f"[Run] Tuning directory: {tuning_dir}")
    print(f"[Run] Terminal transcript: {transcript_path}")
    print(f"[Run] Total jobs: {total_jobs} ({len(run_configs)} configs x {args.replicates} replicates)")

    for idx, config in enumerate(run_configs):
        msg = f"--- Starting Grid Config {idx+1}/{len(run_configs)}: {config['name']} (LR={config['learning_rate']}, BS={config['batch_size']}) ---"
        print(f"\n{msg}")
        for rep in range(1, args.replicates + 1):
            completed_jobs += 1
            job_msg = f"[Run] Job {completed_jobs}/{total_jobs} -> {config['name']} Rep {rep}/{args.replicates}"
            print(job_msg)
            end_to_end_tuning_worker(
                config, tuning_dir, data,
                train_ds_args, val_ds_args, test_ds_args,
                args, replicate_id=rep
            )
            
    print(f"\nTuning Complete. Results aggregated in: {os.path.join(tuning_dir, 'tuning_summary.csv')}")
    print(f"Run terminal transcript: {transcript_path}")
    transcript_handle.flush()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
