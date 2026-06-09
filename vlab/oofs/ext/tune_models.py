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
import shutil
import numpy as np
import pandas as pd
import random
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp

# Keep the optimized 13-parameter L-system names aligned with dataset generation.
LSYSTEM_PARAM_NAMES = [
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


def _params_to_list(best_params):
    if best_params is None:
        return []
    if torch.is_tensor(best_params):
        return [float(value) for value in best_params.detach().cpu().view(-1).tolist()]
    return [float(value) for value in best_params]


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
    "sinkhorn_hollow": {
        "dataset_class": model_base.PlantDataset,
        "model_class": sinkhorn_model.HierarchicalPlantSurrogateNet,
        "loss_fn": model_base.hierarchical_loss_function,
        "module": sinkhorn_model,
        "internal_type": "sinkhorn",
        "model_kwargs": {"use_encoder": False, "use_scaler": False, "use_aggregator": False}
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

def evaluate_loader_r2(model, loader, real_bp_batch, real_ep_batch):
    """Evaluate a loader and return the vectors used for R2 plus summary stats."""
    model.eval()
    all_preds_list = []
    all_targets_list = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                params, costs, _ = batch
            else:
                params, costs = batch

            batch_size = params.size(0)
            curr_bp = real_bp_batch.repeat(batch_size, 1, 1, 1)
            curr_ep = real_ep_batch.repeat(batch_size, 1, 1, 1)

            pred = model(params, curr_bp, curr_ep)
            all_preds_list.append(torch.atleast_1d(pred).detach().cpu())
            all_targets_list.append(torch.atleast_1d(costs).detach().cpu())

    all_preds = torch.cat(all_preds_list, dim=0).squeeze()
    all_targets = torch.cat(all_targets_list, dim=0).squeeze()

    target_mean = torch.mean(all_targets)
    ss_tot = torch.sum((all_targets - target_mean) ** 2)
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        "preds": all_preds,
        "targets": all_targets,
        "target_mean": target_mean,
        "ss_tot": ss_tot,
        "ss_res": ss_res,
        "r2": r2,
    }

def reset_verify_dir(model_dir, unique_run_name):
    """Remove any stale Verify folder before starting a replicate."""
    verify_dir = os.path.join(model_dir, "Verify")
    if os.path.isdir(verify_dir):
        print(f"[{unique_run_name}] Resetting stale Verify folder: {verify_dir}")
        shutil.rmtree(verify_dir)
    os.makedirs(verify_dir, exist_ok=True)
    return verify_dir

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

    # If a previous attempt left verification artifacts behind, clear them so this run starts clean.
    verify_dir = reset_verify_dir(model_dir, unique_run_name)

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
    best_val_model_path = os.path.join(model_dir, "best_val_model.pt")
    final_model_path = os.path.join(model_dir, "final_model.pt")
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
        
    best_lpfg_cost = float('nan') if args.train_only else float('inf')
    best_lpfg_params = None
    best_lpfg_surrogate_cost = float('nan') if args.train_only else float('inf')
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
        
        if not args.train_only:
            # 3. Lightweight End-to-End Evaluation
            model.eval()
            model.to('cpu')

            best_params, opt_cost, _ = optimize_params_for_model(
                model, model_opt_type, data["real_bp_batch"].to('cpu'), data["real_ep_batch"].to('cpu'), opt_args
            )

            # Verify physically with LPFG
            syn_bp, syn_ep = run_simulation_verification(best_params, output_dir=verify_dir)

            real_lpfg_cost = float('inf')
            if syn_bp is not None:
                real_lpfg_cost = evaluate_real_cost(syn_bp, syn_ep, data["real_bp"], data["real_ep"])
        else:
            best_params = None
            opt_cost = float('nan')
            real_lpfg_cost = float('nan')

        elapsed = time.time() - ep_start
        if args.train_only:
            print(f"[{unique_run_name}] Ep {epoch+1}/{args.epochs} | "
                  f"Val R2: {val_metrics['r2']:.4f} (Best: {max(best_val_r2, val_metrics['r2']):.4f})")
        else:
            print(f"[{unique_run_name}] Ep {epoch+1}/{args.epochs} | "
                  f"Val R2: {val_metrics['r2']:.4f} (Best: {max(best_val_r2, val_metrics['r2']):.4f}) | "
                  f"LPFG Cost: {real_lpfg_cost:.4f} (Best: {min(best_lpfg_cost, real_lpfg_cost):.4f})")
        
        with open(log_csv, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_metrics['loss']:.6f},{val_metrics['r2']:.6f},"
                    f"{opt_cost:.6f},{real_lpfg_cost:.6f},{elapsed:.2f}\n")

        # 4. Early Stopping strictly on Real LPFG Cost
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            # Save a snapshot of the model with the best validation R2
            try:
                torch.save(model.state_dict(), best_val_model_path)
                print(f"[{unique_run_name}] Saved best-val model (R2={best_val_r2:.4f}) -> {best_val_model_path}")
            except Exception:
                pass

        if not args.train_only and real_lpfg_cost < best_lpfg_cost:
            best_lpfg_cost = real_lpfg_cost
            best_lpfg_params = best_params.detach().cpu().view(-1).clone()
            best_lpfg_surrogate_cost = float(opt_cost)
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[{unique_run_name}] * New best LPFG score achieved! Saving model.")
        elif not args.train_only:
            patience_counter += 1
            print(f"[{unique_run_name}] LPFG score did not improve. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"[{unique_run_name}] Early stopping triggered at epoch {epoch+1}!")
                break
                
        ep_start = time.time()
        
    torch.save(model.state_dict(), final_model_path)
    print(f"[{unique_run_name}] Saved final model -> {final_model_path}")

    cleanup_empty_verify_dirs(model_dir)
    log_status(f"[Worker] Finished {unique_run_name}. Best LPFG Score: {best_lpfg_cost:.4f} | Best R2: {best_val_r2:.4f}")
    transcript_handle.flush()

    # Final test-set R2 report with the exact vectors used in the calculation.
    # Prefer the final model for post-training evaluation so the reported score
    # matches the fully trained checkpoint, not an earlier best checkpoint.
    model_to_load = None
    if os.path.exists(final_model_path):
        model_to_load = final_model_path
    elif os.path.exists(best_val_model_path):
        model_to_load = best_val_model_path
    elif os.path.exists(best_model_path):
        model_to_load = best_model_path

    if model_to_load is None:
        print(f"[{unique_run_name}] No saved model found to evaluate test R2.")
        transcript_handle.flush()
        return

    model.load_state_dict(torch.load(model_to_load, map_location="cpu"))
    test_loader_report = evaluate_loader_r2(
        model.cpu(),
        test_loader,
        data["real_bp_batch"].to("cpu"),
        data["real_ep_batch"].to("cpu"),
    )
    test_r2_report_path = os.path.join(model_dir, "test_r2_report.txt")
    sample_count = min(10, test_loader_report["preds"].numel())
    with open(test_r2_report_path, "w") as rf:
        rf.write(f"model={config['name']}\n")
        rf.write(f"replicate={replicate_id}\n")
        rf.write(f"dataset={args.dataset}\n")
        rf.write(f"plant={args.plant}\n")
        rf.write(f"loaded_model={os.path.basename(model_to_load)}\n")
        rf.write(f"split=test\n")
        rf.write(f"num_points={test_loader_report['targets'].numel()}\n")
        rf.write(f"target_mean={float(test_loader_report['target_mean']):.10f}\n")
        rf.write(f"ss_tot={float(test_loader_report['ss_tot']):.10f}\n")
        rf.write(f"ss_res={float(test_loader_report['ss_res']):.10f}\n")
        rf.write(f"r2={float(test_loader_report['r2']):.10f}\n")
        rf.write("\nfirst_vectors\n")
        rf.write("pred,target,diff\n")
        for idx in range(sample_count):
            pred_value = float(test_loader_report["preds"][idx])
            target_value = float(test_loader_report["targets"][idx])
            rf.write(f"{pred_value:.10f},{target_value:.10f},{(target_value - pred_value):.10f}\n")

    if best_lpfg_params is not None and not args.train_only:
        optimized_params_path = os.path.join(model_dir, "optimized_params.csv")
        with open(optimized_params_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model",
                "replicate",
                "dataset",
                "plant",
                "dataset_fraction",
                "epochs_trained",
                "best_val_r2",
                "best_lpfg_cost",
                "best_lpfg_surrogate_cost",
                *LSYSTEM_PARAM_NAMES,
            ])
            writer.writerow([
                config["name"],
                replicate_id,
                args.dataset,
                args.plant,
                args.dataset_fraction,
                total_epochs_trained,
                f"{best_val_r2:.6f}",
                f"{best_lpfg_cost:.6f}",
                f"{best_lpfg_surrogate_cost:.6f}",
                *[f"{value:.10f}" for value in _params_to_list(best_lpfg_params)],
            ])
            print(f"[{unique_run_name}] Saved optimized L-system parameters -> {optimized_params_path}")

    # Log final summary line for this replicate so we don't have to rewrite aggregate_results() completely
    summary_csv = os.path.join(run_dir, "tuning_summary.csv")
    write_header = not os.path.exists(summary_csv)
    
    summary_lpfg_cost = "" if args.train_only else f"{best_lpfg_cost:.6f}"
    summary_lpfg_surrogate_cost = "" if args.train_only else f"{best_lpfg_surrogate_cost:.6f}"

    with open(summary_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "model",
                "replicate",
                "learning_rate",
                "batch_size",
                "dataset_fraction",
                "epochs_trained",
                "best_val_r2",
                "best_lpfg_cost",
                "best_lpfg_surrogate_cost",
                *[f"opt_{name}" for name in LSYSTEM_PARAM_NAMES],
            ])
        writer.writerow([
            config["name"],
            replicate_id,
            config["learning_rate"],
            config["batch_size"],
            args.dataset_fraction,
            total_epochs_trained,
            f"{best_val_r2:.6f}",
            summary_lpfg_cost,
            summary_lpfg_surrogate_cost,
            *[f"{value:.10f}" for value in _params_to_list(best_lpfg_params)],
        ])
    print(f"[{unique_run_name}] Updated tuning summary -> {summary_csv}")
    # Create a DONE marker so resumed runs can skip completed replicates
    try:
        done_path = os.path.join(model_dir, "DONE")
        with open(done_path, "w") as df:
            df.write(f"Completed: {datetime.now().isoformat()}\n")
    except Exception:
        pass

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
    parser.add_argument("--resume", action="store_true", help="If set, skip replicates already completed in the target tuning directory")
    parser.add_argument("--tuning-dir", type=str, default=None, help="Path to an existing tuning directory to resume or append results into")
    parser.add_argument("--train-only", action="store_true", help="Skip per-epoch optimization and only train the surrogate models")
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
            # Include fraction and epoch config in name to avoid overwriting different convergence tests
            frac_percent = int(args.dataset_fraction * 100)
            config["name"] = f"{input_name}_lr{lr}_bs{bs}_frac{frac_percent}pct_ep{args.epochs}_pat{args.patience}"
            run_configs.append(config)
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Store in "Hyperparameter Tuning" directory instead of intertwining with datasets
    base_tuning_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hyperparameter Tuning")
    models_str = "-".join(target_models)
    if args.tuning_dir:
        tuning_dir = args.tuning_dir
        os.makedirs(tuning_dir, exist_ok=True)
    else:
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

    def replicate_completed(run_dir, model_name, replicate_id):
        """Return True if the replicate appears completed.

        Checks (in order):
        - entry exists in tuning_summary.csv for this model+replicate
        - DONE marker file exists in the replicate dir

        Note: we intentionally do NOT consider the presence of `best_model.pt` alone
        as proof of completion because interrupted runs may have written partial
        artifacts. Only a summary entry or explicit DONE marker indicate a
        completed replicate.
        """
        summary_csv = os.path.join(run_dir, "tuning_summary.csv")
        if os.path.exists(summary_csv):
            try:
                with open(summary_csv, "r", newline="") as sf:
                    reader = csv.DictReader(sf)
                    for row in reader:
                        try:
                            rep = int(row.get("replicate", -1))
                        except Exception:
                            rep = -1
                        if row.get("model") == model_name and rep == replicate_id:
                            return True
            except Exception:
                pass

        model_dir = os.path.join(run_dir, model_name, f"Rep_{replicate_id}")
        done_marker = os.path.join(model_dir, "DONE")
        if os.path.exists(done_marker):
            return True

        return False

    print(f"[Run] Tuning directory: {tuning_dir}")
    print(f"[Run] Terminal transcript: {transcript_path}")
    print(f"[Run] Total jobs: {total_jobs} ({len(run_configs)} configs x {args.replicates} replicates)")

    for idx, config in enumerate(run_configs):
        msg = f"--- Starting Grid Config {idx+1}/{len(run_configs)}: {config['name']} (LR={config['learning_rate']}, BS={config['batch_size']}) ---"
        print(f"\n{msg}")
        for rep in range(1, args.replicates + 1):
            completed_jobs += 1
            # If resuming, check whether this replicate already completed (summary, DONE, or best model)
            if args.resume and replicate_completed(tuning_dir, config["name"], rep):
                print(f"[Run] Resume: skipping already-completed {config['name']} Rep {rep}")
                continue

            # If resuming and the replicate dir exists but is NOT marked complete,
            # remove it entirely so we restart training deterministically from epoch 1.
            model_dir_path = os.path.join(tuning_dir, config["name"], f"Rep_{rep}")
            if args.resume and os.path.isdir(model_dir_path) and not replicate_completed(tuning_dir, config["name"], rep):
                print(f"[Run] Resume: found incomplete artifacts for {config['name']} Rep {rep}, resetting folder: {model_dir_path}")
                try:
                    shutil.rmtree(model_dir_path)
                except Exception as e:
                    print(f"[Run] Warning: failed to remove {model_dir_path}: {e}")
                os.makedirs(model_dir_path, exist_ok=True)
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
