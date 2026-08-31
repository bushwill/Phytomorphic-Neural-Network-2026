"""
Hyperparameter Tuning Evaluation Script

Comprehensive aggregation of tuning results across replicates.
Reads tuning_summary.csv and produces replicate_summary.txt with:
  - Individual R² vectors for each HP set (all replicates)
  - Mean, std, min, max statistics
  - All hyperparameter details (LR, BS, epochs trained)
  - Ranked lists by multiple metrics
  - Cost degradation analysis
  - Per-replicate breakdown
"""

import csv
import os
import sys
from collections import defaultdict
from statistics import mean, stdev


def generate_replicate_summary(tuning_dir, output_file):
    """
    Read tuning_summary.csv and aggregate by HP set with full details.
    
    Args:
        tuning_dir (str): Path to hyperparameter tuning directory
        output_file (str): Path to write replicate_summary.txt
    """
    csv_path = os.path.join(tuning_dir, "tuning_summary.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return False
    
    # Read data with all fields
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print(f"Error: {csv_path} is empty")
        return False
    
    # Group by model (HP set) with full detail
    hp_sets = defaultdict(list)
    for row in rows:
        model = row['model']
        hp_sets[model].append({
            'r2': float(row['best_val_r2']),
            'cost': float(row.get('best_vlab_cost', row.get('best_lpfg_cost'))),
            'replicate': int(row['replicate']),
            'learning_rate': float(row['learning_rate']),
            'batch_size': int(row['batch_size']),
            'epochs_trained': int(row['epochs_trained']),
            'dataset_fraction': float(row['dataset_fraction']),
        })
    
    # Compute comprehensive statistics
    results = []
    for hp_set in sorted(hp_sets.keys()):
        data = hp_sets[hp_set]
        
        r2_vals = sorted([d['r2'] for d in data])
        cost_vals = sorted([d['cost'] for d in data])
        
        mean_r2 = mean(r2_vals)
        std_r2 = stdev(r2_vals) if len(r2_vals) > 1 else 0
        mean_cost = mean(cost_vals)
        std_cost = stdev(cost_vals) if len(cost_vals) > 1 else 0
        
        # Get representative HP values (from first replicate)
        rep_data = data[0]
        
        results.append({
            'hp_set': hp_set,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'min_r2': min(r2_vals),
            'max_r2': max(r2_vals),
            'r2_vals': r2_vals,
            'r2_vector': ' '.join(f"{v:.6f}" for v in r2_vals),
            'mean_cost': mean_cost,
            'std_cost': std_cost,
            'min_cost': min(cost_vals),
            'max_cost': max(cost_vals),
            'cost_vals': cost_vals,
            'cost_vector': ' '.join(f"{v:.2f}" for v in cost_vals),
            'learning_rate': rep_data['learning_rate'],
            'batch_size': rep_data['batch_size'],
            'epochs_trained_avg': mean([d['epochs_trained'] for d in data]),
            'dataset_fraction': rep_data['dataset_fraction'],
            'num_replicates': len(data),
            'replicate_data': data,
        })
    
    # Write comprehensive output
    with open(output_file, 'w') as f:
        dataset_name = os.path.basename(tuning_dir)
        f.write("="*170 + "\n")
        f.write(f"COMPREHENSIVE HYPERPARAMETER TUNING SUMMARY: {dataset_name}\n")
        f.write("="*170 + "\n\n")
        
        # SECTION 1: Summary Table with R² Vectors
        f.write("-"*170 + "\n")
        f.write("SECTION 1: AGGREGATE STATISTICS WITH R² VECTORS\n")
        f.write("-"*170 + "\n\n")
        
        f.write(f"{'HP Set':<30} {'Mean R²':<12} {'Std R²':<12} {'R² Range':<25} {'R² Vector (All Reps)':<70}\n")
        f.write("-"*170 + "\n")
        
        for r in sorted(results, key=lambda x: x['mean_r2'], reverse=True):
            r2_range = f"[{r['min_r2']:.6f}, {r['max_r2']:.6f}]"
            f.write(f"{r['hp_set']:<30} {r['mean_r2']:<12.6f} {r['std_r2']:<12.6f} {r2_range:<25} {r['r2_vector']:<70}\n")
        
        # SECTION 2: Cost Analysis
        f.write("\n" + "-"*170 + "\n")
        f.write("SECTION 2: LPFG COST ANALYSIS WITH COST VECTORS\n")
        f.write("-"*170 + "\n\n")
        
        f.write(f"{'HP Set':<30} {'Mean Cost':<15} {'Std Cost':<15} {'Cost Range':<30} {'Cost Vector (All Reps)':<60}\n")
        f.write("-"*170 + "\n")
        
        for r in sorted(results, key=lambda x: x['mean_cost']):
            cost_range = f"[{r['min_cost']:.1f}, {r['max_cost']:.1f}]"
            f.write(f"{r['hp_set']:<30} {r['mean_cost']:<15.1f} {r['std_cost']:<15.1f} {cost_range:<30} {r['cost_vector']:<60}\n")
        
        # SECTION 3: Hyperparameter Details
        f.write("\n" + "-"*170 + "\n")
        f.write("SECTION 3: HYPERPARAMETER DETAILS & TRAINING METRICS\n")
        f.write("-"*170 + "\n\n")
        
        f.write(f"{'HP Set':<30} {'LR':<12} {'BS':<5} {'Avg Epochs':<12} {'Frac':<8} {'#Reps':<5}\n")
        f.write("-"*170 + "\n")
        
        for r in sorted(results, key=lambda x: x['mean_r2'], reverse=True):
            f.write(f"{r['hp_set']:<30} {r['learning_rate']:<12.1e} {r['batch_size']:<5} {r['epochs_trained_avg']:<12.1f} {r['dataset_fraction']:<8.2f} {r['num_replicates']:<5}\n")
        
        # SECTION 4: Per-Replicate Breakdown
        f.write("\n" + "-"*170 + "\n")
        f.write("SECTION 4: DETAILED PER-REPLICATE BREAKDOWN\n")
        f.write("-"*170 + "\n\n")
        
        for r in sorted(results, key=lambda x: x['mean_r2'], reverse=True):
            f.write(f"\nHP SET: {r['hp_set']}\n")
            f.write(f"  Summary: R²={r['mean_r2']:.6f}±{r['std_r2']:.6f}  Cost={r['mean_cost']:.1f}±{r['std_cost']:.1f}\n")
            f.write(f"  LR={r['learning_rate']:.1e}  BS={r['batch_size']}  Avg Epochs={r['epochs_trained_avg']:.1f}\n")
            f.write(f"\n  {'Rep':<5} {'R²':<12} {'Cost':<15} {'Epochs':<10}\n")
            f.write(f"  {'-'*42}\n")
            
            for rep_data in sorted(r['replicate_data'], key=lambda x: x['replicate']):
                f.write(f"  {rep_data['replicate']:<5} {rep_data['r2']:<12.6f} {rep_data['cost']:<15.1f} {rep_data['epochs_trained']:<10}\n")
        
        # SECTION 5: Ranking by R² (Top 10)
        f.write("\n" + "="*170 + "\n")
        f.write("SECTION 5: TOP 10 HP SETS BY MEAN R² (Validation Accuracy)\n")
        f.write("="*170 + "\n\n")
        
        sorted_by_r2 = sorted(results, key=lambda x: x['mean_r2'], reverse=True)
        f.write(f"{'Rank':<5} {'HP Set':<30} {'Mean R²':<12} {'Std R²':<12} {'Mean Cost':<15} {'Std Cost':<15}\n")
        f.write("-"*170 + "\n")
        
        for i, r in enumerate(sorted_by_r2[:10], 1):
            f.write(f"{i:<5} {r['hp_set']:<30} {r['mean_r2']:<12.6f} {r['std_r2']:<12.6f} {r['mean_cost']:<15.1f} {r['std_cost']:<15.1f}\n")
        
        # SECTION 6: Ranking by Cost (Top 10)
        f.write("\n" + "="*170 + "\n")
        f.write("SECTION 6: TOP 10 HP SETS BY MEAN LPFG COST (Optimization Performance)\n")
        f.write("="*170 + "\n\n")
        
        sorted_by_cost = sorted(results, key=lambda x: x['mean_cost'])
        f.write(f"{'Rank':<5} {'HP Set':<30} {'Mean Cost':<15} {'Std Cost':<15} {'Mean R²':<12} {'Std R²':<12}\n")
        f.write("-"*170 + "\n")
        
        for i, r in enumerate(sorted_by_cost[:10], 1):
            f.write(f"{i:<5} {r['hp_set']:<30} {r['mean_cost']:<15.1f} {r['std_cost']:<15.1f} {r['mean_r2']:<12.6f} {r['std_r2']:<12.6f}\n")
        
        # SECTION 7: Stability Analysis (Low Variance)
        f.write("\n" + "="*170 + "\n")
        f.write("SECTION 7: MOST STABLE HP SETS (Lowest R² Variance)\n")
        f.write("="*170 + "\n\n")
        
        sorted_by_stability = sorted(results, key=lambda x: x['std_r2'])
        f.write(f"{'Rank':<5} {'HP Set':<30} {'Std R²':<12} {'Mean R²':<12} {'Mean Cost':<15} {'R² Range':<25}\n")
        f.write("-"*170 + "\n")
        
        for i, r in enumerate(sorted_by_stability[:10], 1):
            r2_range = f"[{r['min_r2']:.6f}, {r['max_r2']:.6f}]"
            f.write(f"{i:<5} {r['hp_set']:<30} {r['std_r2']:<12.6f} {r['mean_r2']:<12.6f} {r['mean_cost']:<15.1f} {r2_range:<25}\n")
    
    return True


def find_tuning_directories():
    """
    Dynamically discover all hyperparameter tuning directories.
    
    Returns:
        list: Paths to all directories containing tuning_summary.csv
    """
    search_base = "/home/pzu426/PhytomorphicNN/vlab/oofs/ext/Hyperparameter Tuning"
    tuning_dirs = []
    
    if not os.path.exists(search_base):
        print(f"Error: Base directory not found: {search_base}")
        return tuning_dirs
    
    # Find all subdirectories with tuning_summary.csv
    for item in os.listdir(search_base):
        item_path = os.path.join(search_base, item)
        if os.path.isdir(item_path):
            csv_path = os.path.join(item_path, "tuning_summary.csv")
            if os.path.exists(csv_path):
                tuning_dirs.append(item_path)
    
    return sorted(tuning_dirs)


def main():
    """
    Process all hyperparameter tuning directories that need summary generation.
    """
    tuning_dirs = find_tuning_directories()
    
    if not tuning_dirs:
        print("No tuning directories with tuning_summary.csv found")
        return 1
    
    print("="*130)
    print("HYPERPARAMETER TUNING EVALUATION")
    print("="*130)
    print()
    
    # Filter: only process directories without replicate_summary.txt
    dirs_to_process = []
    dirs_already_done = []
    
    for tuning_dir in tuning_dirs:
        output_file = os.path.join(tuning_dir, "replicate_summary.txt")
        if os.path.exists(output_file):
            dirs_already_done.append(tuning_dir)
        else:
            dirs_to_process.append(tuning_dir)
    
    # Report status
    print(f"Found {len(tuning_dirs)} tuning directories:")
    print(f"  - {len(dirs_already_done)} already have replicate_summary.txt")
    print(f"  - {len(dirs_to_process)} need processing")
    print()
    
    if dirs_already_done:
        print("Already processed:")
        for d in dirs_already_done:
            print(f"  ✓ {os.path.basename(d)}")
        print()
    
    if not dirs_to_process:
        print("All tuning directories already have summary files!")
        return 0
    
    print("Processing:")
    success_count = 0
    for tuning_dir in dirs_to_process:
        output_file = os.path.join(tuning_dir, "replicate_summary.txt")
        if generate_replicate_summary(tuning_dir, output_file):
            print(f"  ✓ {os.path.basename(tuning_dir)}")
            success_count += 1
        else:
            print(f"  ✗ {os.path.basename(tuning_dir)}")
    
    print()
    print("="*130)
    if success_count == len(dirs_to_process):
        print(f"SUCCESS: All {success_count} summary files created!")
    else:
        print(f"PARTIAL: {success_count}/{len(dirs_to_process)} summary files created")
    print("="*130)
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
