"""
Convergence Analysis - Post-Processing & Reporting

This module analyzes convergence testing results by:
  1. Parsing tuning_summary.csv files from all Tuning_* directories
  2. Grouping results by model, data fraction, and epoch count
  3. Computing statistics across replicates
  4. Generating comprehensive convergence_summary.txt report

Usage:
    python3 convergence_test.py [--summary-file PATH]

Environment Variables:
    CONVERGENCE_SUMMARY: Path to output summary file (default: convergence_summary.txt)
"""

import os
import sys
import csv
import glob
import argparse
from collections import defaultdict
from statistics import mean, stdev
from datetime import datetime
import re

OPT_PARAM_NAMES = [
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


def infer_dataset_metadata(base_dir):
    """Infer dataset name and split sizes from tuning directory metadata."""
    metadata = {
        "dataset_name": "unknown",
        "train_size": None,
        "val_size": None,
        "test_size": None,
    }

    desc_path = os.path.join(base_dir, "description.txt")
    if os.path.exists(desc_path):
        try:
            with open(desc_path, "r") as f:
                for line in f:
                    if line.startswith("Dataset:"):
                        metadata["dataset_name"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    base_name = os.path.basename(os.path.abspath(base_dir))
    name_source = metadata["dataset_name"] if metadata["dataset_name"] != "unknown" else base_name
    split_match = re.search(r"(?P<train>\d+)_(?P<val>\d+)_(?P<test>\d+)", name_source)

    if split_match:
        metadata["train_size"] = int(split_match.group("train"))
        metadata["val_size"] = int(split_match.group("val"))
        metadata["test_size"] = int(split_match.group("test"))

    return metadata


def format_dataset_label(metadata):
    """Build a concrete dataset label for summary headers."""
    dataset_name = metadata.get("dataset_name", "unknown")
    train_size = metadata.get("train_size")
    val_size = metadata.get("val_size")
    test_size = metadata.get("test_size")

    if train_size is not None and val_size is not None and test_size is not None:
        return (
            f"{dataset_name} "
            f"(Train: {train_size:,} | Val: {val_size:,} | Test: {test_size:,})"
        )
    return dataset_name


def parse_tuning_dirs(base_dir="Hyperparameter Tuning"):
    """
    Parse all tuning result directories (looks for tuning_summary.csv).
    
    Searches for tuning_summary.csv files in base_dir and its subdirectories.
    Works with both traditional Tuning_* structure and direct model folder structures.
    
    Args:
        base_dir (str): Base directory to search for tuning results
        
    Returns:
        dict: Results keyed by (model, fraction, epochs) with lists of replicate data
    """
    results = defaultdict(list)
    
    if not os.path.exists(base_dir):
        return results
    
    # Find all tuning_summary.csv files recursively
    csv_files = glob.glob(os.path.join(base_dir, "**/tuning_summary.csv"), recursive=True)
    
    for tuning_csv in sorted(csv_files):
        if not os.path.exists(tuning_csv):
            continue
        
        # Parse tuning_summary.csv
        with open(tuning_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract model type (before _lr)
                model_name = row['model'].split('_lr')[0]
                frac = float(row['dataset_fraction'])
                epochs = int(row['epochs_trained'])
                replicate = int(row['replicate'])
                
                key = (model_name, frac, epochs)
                results[key].append({
                    'r2': float(row['best_val_r2']),
                    'cost': float(row.get('best_vlab_cost', row.get('best_lpfg_cost'))),
                    'surrogate_cost': float(row.get('best_lpfg_surrogate_cost', 'nan')),
                    'epochs': epochs,
                    'fraction': frac,
                    'learning_rate': float(row['learning_rate']),
                    'batch_size': int(row['batch_size']),
                    'replicate': replicate,
                    'epochs_trained': int(row['epochs_trained']),
                    'lr': float(row['learning_rate']),
                    'bs': int(row['batch_size']),
                    'opt_params': {
                        name: float(row[f'opt_{name}'])
                        for name in OPT_PARAM_NAMES
                        if f'opt_{name}' in row and row[f'opt_{name}'] != ''
                    },
                })
    
    return results


def generate_convergence_summary(results, summary_file, tuning_dir=""):
    """
    Generate comprehensive convergence analysis summary.
    
    Outputs detailed analysis including:
      - Per-model, per-fraction learning curves with R² vectors
      - Per-replicate breakdown with hyperparameters
      - Cross-model comparison tables
      - Key findings and recommendations
    
    Args:
        results (dict): Results from parse_tuning_dirs()
        summary_file (str): Path to output summary file
        
    Returns:
        bool: True if successful
    """
    if not results:
        print("No convergence data found. Run convergence tests first.")
        return False
    
    dataset_meta = infer_dataset_metadata(tuning_dir)
    train_size = dataset_meta.get("train_size")

    with open(summary_file, 'w') as f:
        f.write("="*150 + "\n")
        f.write("CONVERGENCE ANALYSIS SUMMARY - Fraction-Based Learning Curves\n")
        f.write("="*150 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Plant: Plant_023-1\n")
        f.write(f"Dataset: {format_dataset_label(dataset_meta)}\n")
        f.write(f"Configurations tested: {len(results)}\n\n")
        
        # Group by model, fraction, then epochs
        by_model_frac = defaultdict(lambda: defaultdict(list))
        for (model, frac, epochs), data_list in results.items():
            by_model_frac[model][frac].append((epochs, data_list))
        
        # OPTIMIZED HYPERPARAMETER SETS (VLAB cost first, R² retained)
        f.write("="*150 + "\n")
        f.write("OPTIMIZED HYPERPARAMETER SETS\n")
        f.write("="*150 + "\n\n")
        
        models = sorted(set(k[0] for k in results.keys()))
        for model in models:
            f.write(f"\nMODEL: {model.upper()}\n")
            f.write(f"{'-'*150}\n\n")
            
            # Get unique HP sets for this model
            hp_sets = set()
            for (m, frac, epochs), data_list in results.items():
                if m == model:
                    for data in data_list:
                        hp_key = (data['lr'], data['bs'])
                        hp_sets.add(hp_key)
            
            hp_sets = sorted(hp_sets)
            
            # For each HP set, show results across all fractions and epochs
            for lr, bs in hp_sets:
                f.write(f"  Hyperparameter Set: LR={lr:.1e}, BS={bs}\n")
                f.write(f"  {'-'*146}\n")
                f.write(
                    f"  {'Fraction':<12} {'Epochs':<10} {'Avg Cost':<12} {'Std Cost':<12} "
                    f"{'Cost Vector':<28} {'Avg R²':<12} {'Std R²':<12} {'R² Vector':<28}\n"
                )
                f.write(f"  {'-'*146}\n")
                
                fractions = sorted(set(k[1] for k in results.keys() if k[0] == model))
                for frac in fractions:
                    epochs_list = sorted(set(k[2] for k in results.keys() if k[0] == model and k[1] == frac))
                    for epochs in epochs_list:
                        key = (model, frac, epochs)
                        if key in results:
                            data_list = [d for d in results[key] if d['lr'] == lr and d['bs'] == bs]
                            if data_list:
                                r2_vals = sorted([d['r2'] for d in data_list])
                                cost_vals = sorted([d['cost'] for d in data_list])
                                avg_r2 = mean(r2_vals)
                                std_r2 = stdev(r2_vals) if len(r2_vals) > 1 else 0
                                avg_cost = mean(cost_vals)
                                std_cost = stdev(cost_vals) if len(cost_vals) > 1 else 0
                                r2_str = " ".join(f"{v:.4f}" for v in r2_vals)
                                cost_str = " ".join(f"{v:.0f}" for v in cost_vals)
                                
                                frac_pct = f"{frac*100:.0f}%"
                                f.write(
                                    f"  {frac_pct:<12} {epochs:<10} {avg_cost:<12.1f} {std_cost:<12.1f} "
                                    f"{cost_str:<28} {avg_r2:<12.4f} {std_r2:<12.4f} {r2_str:<28}\n"
                                )
                
                f.write("\n")
        
        f.write("\n")
        
        # DETAILED ANALYSIS BY MODEL
        for model in sorted(by_model_frac.keys()):
            f.write("-"*150 + "\n")
            f.write(f"MODEL: {model.upper()}\n")
            f.write("-"*150 + "\n\n")
            
            # By fraction
            for frac in sorted(by_model_frac[model].keys()):
                frac_pct = frac * 100
                data_size = int(train_size * frac) if train_size is not None else None
                f.write(f"\n  {'='*145}\n")
                if data_size is None:
                    f.write(f"  DATA FRACTION: {frac_pct:.0f}% (unknown samples)\n")
                else:
                    f.write(f"  DATA FRACTION: {frac_pct:.0f}% ({data_size:,} samples)\n")
                f.write(f"  {'='*145}\n\n")
                
                # Summary statistics table
                f.write(f"  {'Epochs':<10} {'Avg R²':<12} {'Std R²':<12} {'R² Range':<25} {'R² Values':<50}\n")
                f.write(
                    f"  {'Epochs':<10} {'Avg Cost':<12} {'Std Cost':<12} {'Cost Range':<25} {'Cost Values':<28} "
                    f"{'Avg R²':<12} {'Std R²':<12} {'R² Values':<28}\n"
                )
                f.write(f"  {'-'*140}\n")
                
                epoch_list = sorted(by_model_frac[model][frac])
                for epochs, data_list in epoch_list:
                    r2_vals = sorted([d['r2'] for d in data_list])
                    cost_vals = [d['cost'] for d in data_list]
                    
                    avg_r2 = mean(r2_vals)
                    std_r2 = stdev(r2_vals) if len(r2_vals) > 1 else 0
                    avg_cost = mean(cost_vals)
                    std_cost = stdev(cost_vals) if len(cost_vals) > 1 else 0
                    min_cost = min(cost_vals)
                    max_cost = max(cost_vals)
                    min_r2 = min(r2_vals)
                    max_r2 = max(r2_vals)
                    
                    cost_range = f"[{min_cost:.1f}, {max_cost:.1f}]"
                    cost_str = " ".join(f"{v:.0f}" for v in sorted(cost_vals))
                    r2_range = f"[{min_r2:.4f}, {max_r2:.4f}]"
                    r2_str = " ".join(f"{v:.4f}" for v in r2_vals)
                    
                    f.write(
                        f"  {epochs:<10} {avg_cost:<12.1f} {std_cost:<12.1f} {cost_range:<25} {cost_str:<28} "
                        f"{avg_r2:<12.4f} {std_r2:<12.4f} {r2_str:<28}\n"
                    )
                
                # Detailed per-replicate breakdown
                f.write(f"\n  DETAILED REPLICATE BREAKDOWN:\n")
                f.write(f"  {'-'*140}\n")
                
                for epochs, data_list in epoch_list:
                    f.write(f"\n    Epochs: {epochs}\n")
                    f.write(f"    {'Rep':<5} {'R²':<12} {'Cost':<15} {'LR':<10} {'BS':<5} {'Est Samples':<15}\n")
                    f.write(f"    {'-'*60}\n")
                    
                    for data in sorted(data_list, key=lambda x: x['replicate']):
                        est_samples = int(train_size * frac) if train_size is not None else None
                        est_samples_str = f"{est_samples:,}" if est_samples is not None else "unknown"
                        f.write(f"    {data['replicate']:<5} {data['r2']:<12.4f} {data['cost']:<15.1f} {data['lr']:<10.1e} {data['bs']:<5} {est_samples_str:<15}\n")
                
                f.write(f"\n  OPTIMIZED L-SYSTEM PARAMETER SETS (per replicate):\n")
                f.write(f"  {'-'*140}\n")
                
                for epochs, data_list in epoch_list:
                    f.write(f"\n    Epochs: {epochs}\n")
                    for data in sorted(data_list, key=lambda x: x['replicate']):
                        opt_params = data.get('opt_params', {})
                        if opt_params:
                            param_vector = " | ".join(
                                f"{name}={opt_params[name]:.6f}" for name in OPT_PARAM_NAMES if name in opt_params
                            )
                        else:
                            param_vector = "(not recorded)"
                        f.write(
                            f"    Rep {data['replicate']}: R²={data['r2']:.4f} | Cost={data['cost']:.1f} | {param_vector}\n"
                        )
                
                f.write("\n")
            
            f.write("\n")
        
        # CROSS-MODEL COMPARISON
        f.write("\n" + "="*150 + "\n")
        f.write("CROSS-MODEL COMPARISON BY DATA FRACTION & EPOCH COUNT\n")
        f.write("="*150 + "\n\n")
        
        models = sorted(set(k[0] for k in results.keys()))
        fractions = sorted(set(k[1] for k in results.keys()))
        epoch_counts = sorted(set(k[2] for k in results.keys()))
        
        for frac in fractions:
            frac_pct = frac * 100
            data_size = int(train_size * frac) if train_size is not None else None
            if data_size is None:
                f.write(f"\nData Fraction: {frac_pct:.0f}% (unknown samples)\n")
            else:
                f.write(f"\nData Fraction: {frac_pct:.0f}% ({data_size:,} samples)\n")
            f.write(f"{'-'*140}\n")
            f.write(f"{'Epochs':<10} {'Model':<12} {'Avg R²':<12} {'Std R²':<12} {'Avg Cost':<15} {'Std Cost':<15} {'Min/Max Cost':<20}\n")
            f.write(f"{'-'*140}\n")
            
            for epochs in epoch_counts:
                for model in models:
                    key = (model, frac, epochs)
                    if key in results:
                        data_list = results[key]
                        r2_vals = [d['r2'] for d in data_list]
                        cost_vals = [d['cost'] for d in data_list]
                        
                        avg_r2 = mean(r2_vals)
                        std_r2 = stdev(r2_vals) if len(r2_vals) > 1 else 0
                        avg_cost = mean(cost_vals)
                        std_cost = stdev(cost_vals) if len(cost_vals) > 1 else 0
                        min_cost = min(cost_vals)
                        max_cost = max(cost_vals)
                        
                        cost_range = f"[{min_cost:.0f}, {max_cost:.0f}]"
                        f.write(f"{epochs:<10} {model:<12} {avg_r2:<12.4f} {std_r2:<12.4f} {avg_cost:<15.1f} {std_cost:<15.1f} {cost_range:<20}\n")
                f.write(f"{'-'*140}\n")
        
        # KEY FINDINGS
        f.write("\n" + "="*150 + "\n")
        f.write("KEY FINDINGS & RECOMMENDATIONS:\n")
        f.write("="*150 + "\n\n")
        
        f.write("1. LEARNING CURVES (VLAB Cost primary, R² secondary at Max Epochs):\n\n")
        for model in models:
            f.write(f"   {model.upper()}:\n")
            best_epochs = max(epoch_counts)
            for frac in fractions:
                frac_pct = frac * 100
                key = (model, frac, best_epochs)
                if key in results:
                    r2_vals = [d['r2'] for d in results[key]]
                    cost_vals = [d['cost'] for d in results[key]]
                    avg_r2 = mean(r2_vals)
                    avg_cost = mean(cost_vals)
                    f.write(f"     {frac_pct:>5.0f}% → Cost={avg_cost:.0f}  R²={avg_r2:.4f}\n")
            f.write("\n")
        
        f.write("2. EARLY STOPPING EFFECTIVENESS (% of max epochs used):\n\n")
        for model in models:
            f.write(f"   {model.upper()}:\n")
            for frac in fractions:
                frac_pct = frac * 100
                early_stop_ratios = []
                for epochs in epoch_counts:
                    key = (model, frac, epochs)
                    if key in results:
                        data_list = results[key]
                        avg_early_stop = mean(d['epochs'] / epochs for d in data_list)
                        early_stop_ratios.append((epochs, avg_early_stop))
                
                if early_stop_ratios:
                    avg_ratio = mean(r[1] for r in early_stop_ratios)
                    f.write(f"     {frac_pct:>5.0f}% → {avg_ratio:.1%} of max epochs (early stopping effective: {avg_ratio < 0.6})\n")
            f.write("\n")
        
        f.write("3. MODEL COMPARISON AT FULL DATA (100%, VLAB Cost first):\n\n")
        best_epochs = max(epoch_counts)
        f.write(f"   At {best_epochs} max epochs (best configuration):\n\n")
        f.write(f"   {'Model':<12} {'Avg Cost':<15} {'Best Cost':<15} {'Avg R²':<12} {'Std R²':<12}\n")
        f.write(f"   {'-'*60}\n")
        
        for model in sorted(models):
            key = (model, 1.0, best_epochs)
            if key in results:
                r2_vals = [d['r2'] for d in results[key]]
                cost_vals = [d['cost'] for d in results[key]]
                avg_r2 = mean(r2_vals)
                std_r2 = stdev(r2_vals) if len(r2_vals) > 1 else 0
                avg_cost = mean(cost_vals)
                best_cost = min(cost_vals)
                f.write(f"   {model:<12} {avg_cost:<15.1f} {best_cost:<15.1f} {avg_r2:<12.4f} {std_r2:<12.4f}\n")

            f.write("\n4. MINIMUM DATA FOR CONVERGENCE (>90% R², with VLAB Cost shown):\n\n")
        for model in models:
            f.write(f"   {model.upper()}:\n")
            best_epochs = max(epoch_counts)
            threshold = 0.90
            found_threshold = False
            
            for frac in fractions:
                key = (model, frac, best_epochs)
                if key in results:
                    data_list = results[key]
                    r2_vals = [d['r2'] for d in data_list]
                    cost_vals = [d['cost'] for d in data_list]
                    avg_r2 = mean(r2_vals)
                    avg_cost = mean(cost_vals)
                    
                    if avg_r2 >= threshold and not found_threshold:
                        data_size = int(train_size * frac) if train_size is not None else None
                        if data_size is None:
                            f.write(f"     ✓ {frac*100:.0f}% fraction (unknown samples) → Cost={avg_cost:.0f}, R²={avg_r2:.4f}\n")
                        else:
                            f.write(f"     ✓ {frac*100:.0f}% fraction ({data_size:,} samples) → Cost={avg_cost:.0f}, R²={avg_r2:.4f}\n")
                        found_threshold = True
            
            if not found_threshold:
                f.write(f"     ✗ Does not reach {threshold:.0%} R² even at 100%\n")
            f.write("\n")
        
        f.write("5. RECOMMENDATIONS:\n\n")
        f.write("   • Prioritize lower VLAB cost when selecting final configurations\n")
        f.write("   • Use R² as a secondary model-quality indicator, not the primary objective\n")
        f.write("   • Use minimum fraction that achieves >90% R² for efficient training\n")
        f.write("   • Lower early stopping % ratio indicates better generalization potential\n")
        f.write("   • Monitor if cost increases while R² plateaus (overfitting indicator)\n")
        f.write("   • Use epoch count where early stopping is 40-60% effective (balance)\n")
        f.write("   • Consider model with better cost performance as primary choice\n")
    
    return True


def main():
    """Main entry point for convergence analysis."""
    parser = argparse.ArgumentParser(
        description="Generate convergence analysis summary from tuning results"
    )
    parser.add_argument(
        "--summary-file",
        default=os.environ.get('CONVERGENCE_SUMMARY', 'convergence_summary.txt'),
        help="Path to output summary file (default: from CONVERGENCE_SUMMARY env or convergence_summary.txt)"
    )
    parser.add_argument(
        "--tuning-dir",
        default="Hyperparameter Tuning",
        help="Base directory containing Tuning_* subdirectories (default: Hyperparameter Tuning)"
    )
    
    args = parser.parse_args()
    
    # Parse results
    results = parse_tuning_dirs(args.tuning_dir)
    
    if not results:
        print(f"Error: No convergence data found in {args.tuning_dir}")
        return 1
    
    # Generate summary
    if generate_convergence_summary(results, args.summary_file, args.tuning_dir):
        print(f"✓ Convergence summary: {args.summary_file}")
        return 0
    else:
        print("Error: Failed to generate convergence summary")
        return 1


if __name__ == "__main__":
    sys.exit(main())
