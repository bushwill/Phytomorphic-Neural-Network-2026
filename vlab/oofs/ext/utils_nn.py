from numpy.random import normal as nran
from numpy.random import uniform as uran
import os
import subprocess
import shutil
from plant_comparison_nn import calculate_cost, read_real_plants
import torch
import numpy as np
import csv
import time as t

def clear_surrogate_dir():
    """Clear and create surrogate directory for clean runs"""
    folder = "data/surrogate"
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

def setup_training_csv(model_name, param_count=13):
    """Setup CSV file for training logs with proper headers"""
    csv_file = model_name + ".csv"
    existing_rows = []
    start_run = 0
    prev_losses = []
    
    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            existing_rows = list(reader)
            start_run = len(existing_rows) - 1  # subtract header
            # Read previous losses (skip header)
            for row in existing_rows[1:]:
                try:
                    prev_losses.append(float(row[5]))  # Use cost_loss column (index 5) for consistency
                except Exception:
                    pass
    
    write_header = not os.path.exists(csv_file)
    if write_header:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            # Standard header for neural network training logs
            writer.writerow(["run #", "datetime", "avg_loss", "avg_loss_change", "total_loss", "cost_loss", "structure_reg", "pred_cost", "true_cost"] + [f"param_{i}" for i in range(param_count)])
    
    return start_run, prev_losses, csv_file

def log_training_step(csv_file, run_number, total_loss_val, cost_loss, structure_reg, pred_cost, true_cost, params, avg_loss, avg_loss_change):
    """Log a single training step to CSV"""
    timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [run_number, timestamp, f"{avg_loss:.4f}", f"{avg_loss_change:.4f}", 
             f"{total_loss_val:.4f}", f"{cost_loss:.4f}", 
             f"{structure_reg:.4f}", f"{pred_cost:.4f}", f"{true_cost:.4f}"] +
            [f"{p:.4f}" for p in params]
        )

def print_training_progress(idx, num_runs, start_run, avg_loss, total_loss_val, cost_loss, accuracy_1000, current_lr, start_time, rel_error=None, pred_cost=None, true_cost=None):
    """Print standardized training progress with meaningful metrics"""
    import sys
    import time
    
    samples_done = idx + 1
    percent = 100.0 * samples_done / num_runs
    elapsed = time.time() - start_time
    samples_left = num_runs - samples_done
    avg_time_per_sample = elapsed / samples_done
    eta = samples_left * avg_time_per_sample
    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
    
    # Calculate relative error if values provided
    if rel_error is None and pred_cost is not None and true_cost is not None:
        rel_error = abs(pred_cost - true_cost) / (abs(true_cost) + 1e-8)
    
    # Build progress string with meaningful metrics
    progress_parts = [
        f"Sample {start_run + idx + 1}",
        f"({percent:.1f}%)",
    ]
    
    if rel_error is not None:
        progress_parts.append(f"rel_err={rel_error:.3f}")
    
    if pred_cost is not None and true_cost is not None:
        progress_parts.append(f"pred={pred_cost:.0f}")
        progress_parts.append(f"true={true_cost:.0f}")
    
    progress_parts.extend([
        f"acc_1000={accuracy_1000:.1f}%",
        f"lr={current_lr:.1e}",
        f"ETA={eta_str}"
    ])
    
    progress_str = " | ".join(progress_parts)
    
    sys.stdout.write(f"\r{progress_str}                    ")
    sys.stdout.flush()

def calculate_intrinsic_cost(bp_data, ep_data):
    """
    Calculate cost based on intrinsic plant structure properties.
    This is a reusable cost function that doesn't require real plant comparison data.
    """
    if not bp_data or not ep_data:
        return 30000.0
    
    total_cost = 0.0
    num_days = len(bp_data)
    
    for day in range(num_days):
        bp_day = bp_data[day] if day < len(bp_data) else []
        ep_day = ep_data[day] if day < len(ep_data) else []
        
        # Calculate structure complexity cost - much more conservative scaling
        num_bp = len(bp_day)
        num_ep = len(ep_day)
        
        # Simple cost based on structure size - scale for realistic L-system output
        structure_cost = (num_bp * 5) + (num_ep * 4)  # Much lower per-point cost
        
        # Minimal additional costs
        if num_ep > 1:
            spread_cost = 10.0  # Small fixed cost
        else:
            spread_cost = 5.0
            
        efficiency_cost = 10.0  # Small fixed cost
            
        daily_cost = structure_cost + spread_cost + efficiency_cost
        total_cost += daily_cost
    
    # Keep it simple - just clamp to reasonable range
    return max(5000.0, min(150000.0, total_cost))

# Remove the hard-coded normalization constants and add a helper function:
def compute_normalization_stats(num_samples = 100, real_bp=None, real_ep=None):
    if real_bp is None or real_ep is None:
        # If no real data is provided, use synthetic data for normalization
        real_bp, real_ep = read_real_plants()
    params_collection = []
    cost_collection = []
    temp_file = "surrogate_params_temp.vset"
    for i in range(num_samples):
        clear_surrogate_dir()
        p = build_random_parameter_file(temp_file)
        c = generate_and_evaluate(temp_file, real_bp, real_ep)
        if np.isfinite(c) and c >= 0:
            params_collection.append(p)
            cost_collection.append(c)
    return (np.mean(params_collection, axis=0), np.std(params_collection, axis=0),
            np.mean(cost_collection), np.std(cost_collection))
    
def generate_plant(param_file, output_dir):
    """
    Generate a plant using lpfg and save results in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lpfg_command = f"lpfg -w 306 256 lsystem/lsystem.l lsystem/view.v lsystem/materials.mat lsystem/contours.cset lsystem/functions.fset lsystem/functions.tset {param_file} > {output_dir}/lpfg_log.txt"
    # Compile project if needed
    if not os.path.exists("project"):
        os.system("g++ -o project -Wall -Wextra lsystem/project.cpp -lm")
    # Run lpfg
    process = subprocess.Popen(['bash', '-c', lpfg_command])
    process.wait()
    os.system(f"./project 2454 2056 leafposition.dat > {output_dir}/output.txt")
    shutil.move("leafposition.dat", f"./{output_dir}")

def read_syn_plant(file_name):
    """
    Read endpoints and branchpoints from a plant output file.
    """
    with open(file_name, "r") as f:
        lines = f.readlines()
    day_temp = 0
    syn_bp = []
    syn_ep = []
    syn_bp_day = []
    syn_ep_day = []
    day = []
    for line in lines:
        temp = line.split(" ")
        if temp[0] == "Day:":
            day_temp = int(temp[1])
            if day_temp > 2:
                syn_bp.append(syn_bp_day)
                syn_ep.append(syn_ep_day)
                syn_bp_day = []
                syn_ep_day = []
        if (temp[0] != "Day:") & (day_temp > 1):
            if temp[0] == "I":
                syn_bp_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)
            else:
                syn_ep_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)
    if day_temp == 27:
        syn_bp.append(syn_bp_day)
        syn_ep.append(syn_ep_day)
    return syn_bp, syn_ep

def generate_and_evaluate_in_dir(param_file, real_bp, real_ep, output_dir, cost_fn):
    """
    Generate a plant in output_dir and evaluate its cost using cost_fn.
    """
    generate_plant(param_file, output_dir)
    syn_bp, syn_ep = read_syn_plant(f"{output_dir}/output.txt")
    cost = 0
    for i in range(min(len(syn_bp), len(real_bp))):
        cost += cost_fn(syn_bp[i], syn_ep[i], real_bp[i], real_ep[i])
    return cost

def build_parameter_file(filename, params):
    with open(filename, "w") as f:
        f.write(f"#define MAX_PHYTOMERS {params[0]}\n")
        f.write(f"#define PLASTOCHRON {params[1]}\n")
        f.write(f"#define PlantRollAng {params[2]}\n")
        f.write(f"#define PlantDownAng {params[3]}\n")
        f.write(f"#define BrAngle {params[4]}\n")
        f.write(f"#define LeafLen {params[5]}\n")
        f.write(f"#define ExpLeafWid {params[6]}\n")
        f.write(f"#define LeafWid {params[7]}\n")
        f.write(f"#define LEAF_BEND_SCALE {params[8]}\n")
        f.write(f"#define LEAF_TWIST_SCALE {params[9]}\n")
        f.write(f"#define IntLen {params[10]}\n")
        f.write(f"#define IntWid {params[11]}\n")
        f.write(f"#define ExpIntRad {params[12]}\n")
        
def build_random_parameter_file(dir_name):
    f = open(dir_name, "w")
    max_phy = nran(10.,1.)
    plast = nran(3.,0.1)
    chirality = 1.
    if uran(0.,1.) < 0.5 :
        chirality = -1.
    plant_roll_angle = nran(chirality * 90.,10.0)
    plant_down_angle = nran(0.,4.0)
    branch_angle = nran(135.,5.)
    leaf_len = nran(5.,1.)
    exp_leaf_wid = nran(0.5,0.01)
    leaf_wid = nran(1.,0.1)
    leaf_bend_scale = nran(90.,3.)
    leaf_twist_scale = nran(180.,3.)
    node_len = nran(0.7,0.05)
    int_wid = nran(0.9,0.01)
    exp_int_rad = nran(0.5,0.01)
    f.write('#define MAX_PHYTOMERS ' + str(max_phy) + '\n')
    f.write('#define PLASTOCHRON ' + str(plast) + '\n')
    f.write('#define PlantRollAng ' + str(plant_roll_angle) + '\n')
    f.write('#define PlantDownAng ' + str(plant_down_angle) + '\n')
    f.write('#define BrAngle ' + str(branch_angle) + '\n')
    f.write('#define LeafLen ' + str(leaf_len) + '\n')
    f.write('#define ExpLeafWid ' + str(exp_leaf_wid) + '\n')
    f.write('#define LeafWid ' + str(leaf_wid) + '\n')
    f.write('#define LEAF_BEND_SCALE ' + str(leaf_bend_scale) + '\n')
    f.write('#define LEAF_TWIST_SCALE ' + str(leaf_twist_scale) + '\n')
    f.write('#define IntLen ' + str(node_len) + '\n')
    f.write('#define IntWid ' + str(int_wid) + '\n')
    f.write('#define ExpIntRad ' + str(exp_int_rad) + '\n')
    f.close()
    return [max_phy, plast, plant_roll_angle, plant_down_angle, branch_angle, leaf_len, exp_leaf_wid, leaf_wid, leaf_bend_scale, leaf_twist_scale, node_len, int_wid, exp_int_rad]

def generate_and_evaluate(param_file, real_bp, real_ep):
    # Run lpfg to generate the synthetic plant
    generateSurrogatePlant(param_file)
    # Read the synthetic plant's endpoints and branchpoints for the latest run
    syn_bp, syn_ep = read_syn_plant_surrogate()
    # Use the first (or only) day's data for cost calculation
    cost = 0
    for i in range(min(len(syn_bp), len(real_bp))):
        cost += calculate_cost(syn_bp[i], syn_ep[i], real_bp[i], real_ep[i])
    return cost

def generateSurrogatePlant(param_file, calculate_cost_fn=None):
    """
    Generate plant using L-system. 
    If calculate_cost_fn is provided, returns the cost.
    Otherwise, just generates the plant files.
    """
    # setup call to lpfg
    # lpfg_command = "lpfg -w 306 256 lsystem.l view.v materials.mat -a anim.a contours.cset functions.fset functions.tset loop_parameters.vset > log.txt"
    lpfg_command = f"lpfg -w 306 256 lsystem/lsystem.l lsystem/view.v lsystem/materials.mat lsystem/contours.cset lsystem/functions.fset lsystem/functions.tset {param_file} > data/surrogate/lpfg_log.txt"

    if not os.path.exists("project"):
        os.system("g++ -o project -Wall -Wextra lsystem/project.cpp -lm")
    
    if not os.path.exists("data/surrogate"):
        os.makedirs("data/surrogate")

    # run lpfg  
    process = subprocess.Popen(['bash', '-c', lpfg_command])
    process.wait()
    os.system(f"./project 2454 2056 leafposition.dat > data/surrogate/output.txt")
    dest_path = "./data/surrogate/leafposition.dat"
    if os.path.exists(dest_path):
        os.remove(dest_path)
    shutil.move("leafposition.dat", dest_path)
    
    # If cost calculation function provided, calculate and return cost
    if calculate_cost_fn is not None:
        syn_bp, syn_ep = read_syn_plant_surrogate()
        return calculate_cost_fn(syn_bp, syn_ep)
def read_syn_plant_surrogate(file_name="data/surrogate/output.txt"):
    f = open(file_name, "r")
    lines = f.readlines()
    day_temp = 0
    syn_bp = []
    syn_ep = []
    syn_bp_day = []
    syn_ep_day = []
    day = []

    for line in lines:
        temp = line.split(" ")
        if temp[0] == "Day:":
            day_temp = int(temp[1])
            if day_temp>2:
                syn_bp.append(syn_bp_day)
                syn_ep.append(syn_ep_day)
                syn_bp_day = []
                syn_ep_day = []
        if (temp[0] != "Day:") & (day_temp > 1):
            if temp[0] == "I":
                syn_bp_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)
            else:
                syn_ep_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)

    if day_temp == 27:
        syn_bp.append(syn_bp_day)
        syn_ep.append(syn_ep_day)

    f.close()

    return syn_bp, syn_ep