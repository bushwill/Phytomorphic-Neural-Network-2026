"""
Evaluate Trained Surrogate Models.

This module generates evaluation figures for trained models.
It reads training logs, computes parameter importance, and visualizes performance metrics.

Key Features:
- Loss & Metric Convergence Plots (Train vs Val)
- Parameter Importance Analysis (Permutation Importance)
- Prediction vs Ground Truth Density/Scatter Plots
- Relative Error Distribution
- Combined Summary Report Image
- Individual Plots Saved to Model Directory

Usage:
    python3 evaluate_models.py --model_run_dir "Training Data/Run_031326" --dataset_name "Run 021926"
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import Model Architectures (Must match training import structure)
import model_hungarian as baseline_model
import model_sinkhorn as sinkhorn_model
import model_mlp as benchmark_mlp
import utils_nn
from train_models import prepare_real_plant_batch

# Set umask to 0 so that files created inside docker are accessible on host
os.umask(0)

# --- Configuration ---
DEFAULT_DATASET_NAME = "Run 021926"
DEFAULT_MODEL_RUN_DIR = "Training Data/Run_033126"

# Parameter names for the 13 inputs (Customize as needed based on your dataset)
PARAM_NAMES = [
    "Param 1", "Param 2", "Param 3", "Param 4", "Param 5", 
    "Param 6", "Param 7", "Param 8", "Param 9", "Param 10", 
    "Param 11", "Param 12", "Param 13"
]

def load_model_config(model_dir):
    """
    Infers model class and configuration from directory name or saved files.
    This is a heuristic since we don't save a full config file yet.
    """
    model_name_lower = os.path.basename(model_dir).lower()
    
    # Check parent dir if current is a Replicate folder (Rep_X)
    if "rep_" in model_name_lower:
         parent_dir = os.path.basename(os.path.dirname(model_dir))
         model_name_lower = parent_dir.lower()
    
    # Check parent dir (e.g., "baseline_mlp") or full path pattern
    # Assuming standard folder structure: Run_X/ModelName/Rep_Y/
    
    config = {}
    
    if "benchmark" in model_name_lower or "mlp" in model_name_lower:
        config["module"] = benchmark_mlp
        config["model_class"] = benchmark_mlp.BenchmarkSurrogateNet
        config["dataset_class"] = benchmark_mlp.PlantDataset
    elif "baseline" in model_name_lower:
        # In this codebase, "baseline" refers to the benchmark MLP model.
        config["module"] = benchmark_mlp
        config["model_class"] = benchmark_mlp.BenchmarkSurrogateNet
        config["dataset_class"] = benchmark_mlp.PlantDataset
    elif "sinkhorn" in model_name_lower:
        config["module"] = sinkhorn_model
        config["model_class"] = sinkhorn_model.HierarchicalPlantSurrogateNet
        config["dataset_class"] = sinkhorn_model.PlantDataset
    elif "hungarian" in model_name_lower:
        config["module"] = baseline_model
        config["model_class"] = baseline_model.HierarchicalPlantSurrogateNet
        config["dataset_class"] = baseline_model.PlantDataset
    else:
        # Default fallback to baseline if unknown
        config["module"] = baseline_model
        config["model_class"] = baseline_model.HierarchicalPlantSurrogateNet
        config["dataset_class"] = baseline_model.PlantDataset
        
    return config

def plot_training_history(df, output_dir):
    """Generates Loss and R2 convergence plots."""
    # Filter for epoch-level stats (batch "ALL")
    epoch_data = df[df['batch'] == 'ALL'].copy()
    epoch_data['epoch'] = epoch_data['epoch'].astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Loss
    axes[0].plot(epoch_data['epoch'], epoch_data['train_loss'], label='Train Loss')
    axes[0].plot(epoch_data['epoch'], epoch_data['val_loss'], label='Val Loss')
    axes[0].set_title("Loss Convergence")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: R2 Score
    axes[1].plot(epoch_data['epoch'], epoch_data['val_r2'], label='Val R2', color='green')
    axes[1].set_title("Validation R2 Score")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_convergence.png"))
    plt.close()
    return fig

def plot_parameter_importance(model, test_loader, device, output_dir, real_bp_batch, real_ep_batch):
    """
    Computes Permutation Importance for the 13 input parameters.
    Measures drop in R2 score when a parameter is shuffled.
    """
    print(f"  > Computing Parameter Importance...") # Progress Marker
    model.eval()
    baseline_loss = 0.0
    criterion = torch.nn.MSELoss()
    
    # Get all data from loader into single batch for importance calc
    all_params = []
    all_targets = []
    
    with torch.no_grad():
        for params, costs in test_loader:
            all_params.append(torch.atleast_1d(params))
            all_targets.append(torch.atleast_1d(costs))
            
    X = torch.cat(all_params).to(device)
    y = torch.cat(all_targets).to(device)
    
    # Expand real plant batches
    bs = X.size(0)
    curr_bp = real_bp_batch.repeat(bs, 1, 1, 1).to(device)
    curr_ep = real_ep_batch.repeat(bs, 1, 1, 1).to(device)
    
    # Limiting Samples for Importance to Speed Up (Optional)
    # If dataset > 1000 samples, take a subset for permutation importance
    if X.size(0) > 2000:
        indices = torch.randperm(X.size(0))[:2000]
        X = X[indices]
        y = y[indices]
        bs = 2000
        curr_bp = curr_bp[:bs]
        curr_ep = curr_ep[:bs]

    # Baseline Performance
    print("    - Computing Baseline...")
    with torch.no_grad():
        preds = model(X, curr_bp, curr_ep)
        baseline_mse = criterion(preds, y).item()
        
    importances = []
    
    print("    - Permuting Parameters", end="", flush=True)
    for i in range(X.size(1)): # For each parameter (13)
        print(".", end="", flush=True)
        X_permuted = X.clone()
        # Shuffle column i
        idx = torch.randperm(bs)
        X_permuted[:, i] = X_permuted[idx, i]
        
        with torch.no_grad():
            preds_perm = model(X_permuted, curr_bp, curr_ep)
            perm_mse = criterion(preds_perm, y).item()
            
        # Importance = Increase in Error
        importance = perm_mse - baseline_mse
        importances.append(importance)
    print(" Done.")
        
    # Plotting
    importances = np.array(importances)
    # Normalize for readability
    importances_norm = importances / np.max(np.abs(importances))
    
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(PARAM_NAMES))
    plt.barh(y_pos, importances_norm, align='center')
    plt.yticks(y_pos, PARAM_NAMES)
    plt.title("Parameter Importance (Permutation Feature Importance)")
    plt.xlabel("Relative Drop in Performance (Normalized)")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "plot_importance.png"))
    
    # Return figure for composite
    fig = plt.gcf()
    plt.close()
    return fig

def plot_predictions(model, test_loader, device, output_dir, real_bp_batch, real_ep_batch):
    """Scatter plot of Predictions vs Ground Truth."""
    model.eval()
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for params, costs in test_loader:
            params = params.to(device)
            bs = params.size(0)
            curr_bp = real_bp_batch.repeat(bs, 1, 1, 1).to(device)
            curr_ep = real_ep_batch.repeat(bs, 1, 1, 1).to(device)
            
            pred = model(params, curr_bp, curr_ep)
            preds_list.append(torch.atleast_1d(pred.cpu()))
            targets_list.append(torch.atleast_1d(costs))
            
    preds = torch.cat(preds_list).numpy().flatten()
    targets = torch.cat(targets_list).numpy().flatten()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.5, s=10)
    
    # Perfect fit line
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel("Ground Truth Cost")
    plt.ylabel("Predicted Cost")
    plt.title("Prediction Accuracy: Predicted vs Real")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, "plot_scatter.png"))
    fig = plt.gcf()
    plt.close()
    
    # Error Hist
    plt.figure(figsize=(10, 6))
    errors = preds - targets
    plt.hist(errors, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Add KDE-like curve using simplistic Gaussian fit for visual aid or just mean/std text
    mu, std = np.mean(errors), np.std(errors)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2)
    plt.plot(x, p, 'k', linewidth=2)
    
    plt.title(f"Error Distribution (Residuals)\nMean: {mu:.4f}, Std: {std:.4f}")
    plt.xlabel("Prediction Error (Predicted - Real)")
    plt.ylabel("Density")
    
    # Add Zero Line (Target)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label="Zero Error")
    plt.legend()
    
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "plot_error_hist.png"))
    plt.close()

    return fig

def create_summary_card(metrics, output_dir):
    """Creates a text-based image summary of key metrics."""
    plt.figure(figsize=(8, 4))
    plt.axis('off')
    
    text_str = "Model Evaluation Summary\n"
    text_str += "========================\n"
    for k, v in metrics.items():
        text_str += f"{k}: {v}\n"
        
    plt.text(0.1, 0.5, text_str, fontsize=12, fontfamily='monospace', va='center')
    plt.savefig(os.path.join(output_dir, "plot_summary_text.png"))
    fig = plt.gcf()
    plt.close()
    return fig

def combine_plots(plots, output_dir):
    """Stitches all generated plots into one master image."""
    # Assuming standard layout: 
    # Row 1: Convergence (Loss/R2)
    # Row 2: Scatter + Importance
    # Row 3: Summary Text
    
    # Note: Requires PIL or distinct handling since matplotlib figs are closed.
    # Re-loading saved images is easier and robust.
    
    try:
        from PIL import Image
        
        files = [
            "plot_convergence.png", 
            "plot_scatter.png",
            "plot_error_hist.png",
            "plot_importance.png",
            "plot_summary_text.png"
        ]
        
        images = []
        for f in files:
            path = os.path.join(output_dir, f)
            if os.path.exists(path):
                images.append(Image.open(path))
                
        if not images:
            return

        # Simple vertical stack for now, or 2x2 grid
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        total_height = sum(heights)
        
        combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for imp in images:
            # center image
            x_offset = (max_width - imp.size[0]) // 2
            combined.paste(imp, (x_offset, y_offset))
            y_offset += imp.size[1]
            
        combined.save(os.path.join(output_dir, "REPORT_FULL.png"))
        print(f"Full report saved to {os.path.join(output_dir, 'REPORT_FULL.png')}")
        
    except ImportError:
        print("PIL/Pillow not installed. Skipping combined image generation.")

def compute_test_metrics(model, test_loader, device, real_bp_batch, real_ep_batch):
    """Computes metrics on the Test Set for the evaluation summary card."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for params, costs in test_loader:
            params = params.to(device)
            bs = params.size(0)
            curr_bp = real_bp_batch.repeat(bs, 1, 1, 1).to(device)
            curr_ep = real_ep_batch.repeat(bs, 1, 1, 1).to(device)
            
            pred = model(params, curr_bp, curr_ep)
            all_preds.append(torch.atleast_1d(pred))
            all_targets.append(torch.atleast_1d(costs.to(device)))
            
    preds = torch.cat(all_preds).squeeze()
    targets = torch.cat(all_targets).squeeze()
    
    # MSE
    mse = F.mse_loss(preds, targets).item()
    
    # MAE
    mae = F.l1_loss(preds, targets).item()

    # Mean Bias (Signed Error)
    mean_bias = (preds - targets).mean().item()

    # R2
    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Relative Error
    rel_err = torch.abs(preds - targets) / (torch.abs(targets) + 1e-8)
    mean_rel_err = rel_err.mean().item()
    
    # Accuracy
    acc_5pct = (rel_err < 0.05).float().mean().item() * 100
    
    return {
        "Test R2": f"{r2.item():.4f}",
        "Test MSE": f"{mse:.4f}",
        "Test MAE": f"{mae:.4f}",
        "Mean Bias": f"{mean_bias:.4f}",
        "Rel Error": f"{mean_rel_err:.4f}",
        "Acc (<5%)": f"{acc_5pct:.2f}%"
    }

def evaluate_run(run_path, data, output_path=None, include_metrics_artifacts=True):
    """Evaluates a single model run (replicate)."""
    if output_path is None:
        output_path = run_path
    os.makedirs(output_path, exist_ok=True)

    print(f"Evaluating: {run_path}")
    print(f"Output to:  {output_path}")
    
    # 1. Load Config
    config = load_model_config(run_path)
    
    # 2. Check for required files
    log_path = os.path.join(run_path, "training_log.csv")
    model_path = os.path.join(run_path, "best_model.pt")
    
    if not (os.path.exists(log_path) and os.path.exists(model_path)):
        # print(f"Skipping {run_path}: Missing log or model file.")
        return

    # 3. Load Training History
    try:
        df = pd.read_csv(log_path)
        plot_training_history(df, output_path)
    except Exception as e:
        print(f"Error processing training log in {run_path}: {e}")
        return
    
    # 4. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PlantDataset = config["dataset_class"]
    ModelClass = config["model_class"]
    
    # Init Model
    model = ModelClass(
        input_mean=data["input_mean"],
        input_std=data["input_std"],
        output_mean=data["output_mean"],
        output_std=data["output_std"]
    ).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights in {run_path}: {e}")
        return

    # 5. Prepare Test Data
    try:
        test_ds = PlantDataset(data["test_csv"])
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
    
    # 6. Generate Analysis
    try:
        plot_predictions(model, test_loader, device, output_path, data['real_bp_batch'].to(device), data['real_ep_batch'].to(device))
        plot_parameter_importance(model, test_loader, device, output_path, data['real_bp_batch'].to(device), data['real_ep_batch'].to(device))
    except Exception as e:
        print(f"Error generating plots in {run_path}: {e}")
    
    # 7. Optional metrics card (disabled when training already owns metrics)
    if include_metrics_artifacts:
        try:
            # Compute metrics on Test Data for the optional summary card.
            metrics = compute_test_metrics(
                model, test_loader, device, 
                data['real_bp_batch'].to(device), 
                data['real_ep_batch'].to(device)
            )
            create_summary_card(metrics, output_path)
        except Exception as e:
            print(f"Could not compute test metrics: {e}")
    
    # 8. Combine
    combine_plots([], output_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Models")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Dataset folder name under Datasets/ (e.g., 'Run 021926')"
    )
    parser.add_argument(
        "--model_run_dir",
        type=str,
        required=False,
        help="Path to trained model run directory (e.g., Training Data/Run_X)"
    )
    # Backward-compatible alias for older commands.
    parser.add_argument(
        "--run_dir",
        type=str,
        required=False,
        help="Deprecated alias for --model_run_dir"
    )
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_run_dir = args.model_run_dir or args.run_dir
    dataset_name = args.dataset_name
    
    # Default to configured model directory if not specified
    if model_run_dir is None:
        model_run_dir = DEFAULT_MODEL_RUN_DIR
    # Resolve relative model path against script directory for consistency.
    if not os.path.isabs(model_run_dir):
        path_check = os.path.join(script_dir, model_run_dir)
        if os.path.exists(path_check):
            model_run_dir = path_check
    
    if not os.path.exists(model_run_dir):
        print(f"Error: Model Run Directory {model_run_dir} not found.")
        return

    run_name = os.path.basename(os.path.normpath(model_run_dir))
    output_base_dir = os.path.join(script_dir, "Evaluation Results")
    log_dir = os.path.join(output_base_dir, run_name)
    utils_nn.configure_output_file_logging(log_dir, "evaluate_models")

    print(f"Model Directory (Weights): {model_run_dir}")

    # Load Data Globals
    current_dir = script_dir
    dataset_dir = os.path.join(current_dir, "Datasets", dataset_name)
    
    # Check for dataset
    if not os.path.exists(os.path.join(dataset_dir, "Train.csv")):
        print(f"Error: Data Source (CSVs) '{dataset_name}' not found in {dataset_dir}")
        return

    print(f"Data Source (CSVs):      {dataset_dir}")
    train_csv = os.path.join(dataset_dir, "Train.csv")
    test_csv = os.path.join(dataset_dir, "Test.csv")
    
    # Calc Norm Stats (Required for model init)
    try:
        df_train = pd.read_csv(train_csv)
        params = df_train.iloc[:, 2:15].values
        costs = df_train.iloc[:, 1].values
        
        input_mean = np.mean(params, axis=0)
        input_std = np.std(params, axis=0) + 1e-6
        output_mean = np.mean(costs)
        output_std = np.std(costs) + 1e-6
    except Exception as e:
        print(f"Error loading dataset stats: {e}")
        return

    # Load Real Plant (Actual Only)
    try:
        real_bp, real_ep = utils_nn.read_real_plants()
        real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep, use_multiprocessing=False)
    except Exception as e:
        print(f"Error: Could not load real plant structure. {e}")
        print("Please ensure 'Real Plants' directory exists and contains valid data.")
        return

    data = {
        "input_mean": input_mean, "input_std": input_std,
        "output_mean": output_mean, "output_std": output_std,
        "real_bp_batch": real_bp_batch, "real_ep_batch": real_ep_batch,
        "test_csv": test_csv
    }

    # Find all replicate folders
    # Pattern: {run_dir}/{model_config_name}/Rep_{n}/
    
    # Define Output directory base
    # Get the directory where script is running to place "Evaluation Results" next to "Training Data"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(script_dir, "Evaluation Results")

    # Walk through directory
    for root, dirs, files in os.walk(model_run_dir):
        if "best_model.pt" in files and "training_log.csv" in files:
            # Determine output path specifically for this model
            # 1. Get path relative to the specific Run Directory provided
            rel_path_from_run = os.path.relpath(root, model_run_dir)
            
            # 2. Construct full output path: Evaluation Results / Run_X / Model / Rep
            output_path = os.path.join(output_base_dir, run_name, rel_path_from_run)
            
            evaluate_run(root, data, output_path=output_path)

if __name__ == "__main__":
    main()
