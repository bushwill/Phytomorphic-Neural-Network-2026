import os
import pandas as pd
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import skimage.color as color
import sys
import math
import random
import time
import subprocess
import shutil
import csv
import time as t
import torch

from collections import OrderedDict as dict
from random import randrange, uniform
from numpy.random import normal as normal
from numpy.random import normal as nran
from numpy.random import uniform as uran
from scipy.spatial import distance
import skan
from skan import csr
from munkres import Munkres, print_matrix, make_cost_matrix, DISALLOWED
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer


# --- From plant_comparison_nn.py ---

global real_plant_name, size_x, parameter_number, file_path

plant_images_path = "./Real Plants/"
real_plant_name = "Plant_063-32"
plant_image_path = plant_images_path + real_plant_name
size_x = 50  #top best plants we want to know
parameter_number = 12
file_path = "./data/synthetic_images/"


def make_index(cost_):

    m = Munkres()
    indexes = m.compute(cost_)

    total = 0
    leaf_index = []
    for row, column in indexes:
        value = cost_[row][column]
        total += value
        leaf_index.append([row, column, value])

    return leaf_index


def make_matrix(ep_p, bp_p, ep_c, bp_c):

    size = max(len(ep_c), len(ep_p))
    cost_temp = np.zeros((size, size), dtype=float)

    for i in range(0, len(ep_c)):
        for j in range(0, len(ep_p)):
            cost_temp[i, j] = distance.euclidean(ep_c[i], ep_p[j]) + distance.euclidean(bp_c[i], bp_p[j])

    return cost_temp


def parse_dataframe(bin_c):
    if skan.__version__ >= '0.12.2':
        branch_data = csr.summarize(csr.Skeleton(bin_c), separator='-')
    else:
        branch_data = csr.summarize(csr.Skeleton(bin_c))

    branch_data.head()


    edges = branch_data.loc[branch_data["branch-type"] == 1]
    branches = branch_data.loc[branch_data["branch-type"] == 2]

    points = []
    length = []

    edges = edges.reset_index()
    for i in range(edges.shape[0]):
        points.append([edges["image-coord-src-0"][i], edges["image-coord-src-1"][i]])
        points.append([edges["image-coord-dst-0"][i], edges["image-coord-dst-1"][i]])
        length.append(edges["branch-distance"][i])

    branching = []
    branches = branches.reset_index()

    for i in range(branches.shape[0]):
        branching.append([branches["image-coord-src-0"][i], branches["image-coord-src-1"][i]])
        branching.append([branches["image-coord-dst-0"][i], branches["image-coord-dst-1"][i]])

    ep = []
    bp = []
    length_edge = []

    i = 0
    while i <= len(points) - 1:
        if points[i] in branching:
            bp.append(points[i])
            ep.append(points[i + 1])
            if points[i + 1][1] > points[i][1]:
                pos = 1
            else:
                pos = 0
            length_edge.append([length[int(i / 2)], pos])


        else:
            ep.append(points[i])
            bp.append(points[i + 1])
            if points[i][1] > points[i + 1][1]:
                pos = 1
            else:
                pos = 0
            length_edge.append([length[int(i / 2)], pos])

        i = i + 2


    if len(ep) == 0:
        if branch_data["image-coord-src-0"][0] < branch_data["image-coord-dst-0"][0]:
            ep.append([branch_data["image-coord-src-0"][0], branch_data["image-coord-src-1"][0]])
            bp.append([branch_data["image-coord-dst-0"][0], branch_data["image-coord-dst-1"][0]])
        else:
            bp.append([branch_data["image-coord-src-0"][0], branch_data["image-coord-src-1"][0]])
            ep.append([branch_data["image-coord-dst-0"][0], branch_data["image-coord-dst-1"][0]])

    if len(ep) > 0:
        root = [ep[len(ep) - 1][0], ep[len(ep) - 1][1]]
    else:
        root = [bp[len(ep) - 1][0], bp[len(ep) - 1][1]]

    info = sorted(zip(bp, ep, length_edge))

    if len(ep) > 1:
        ep = ep[:-1]

    return info, ep, bp, length_edge, root


def read_real_plants():

    real_ep =[]
    real_bp =[]

    for day_real in range(2,28):
        if day_real < 10:
            image_name = plant_image_path + "/topo/Day_00" + str(
                day_real) + ".png"
        else:
            image_name = plant_image_path + "/topo/Day_0" + str(
                day_real) + ".png"

        image = io.imread(image_name)
        image_gray = color.rgb2gray(color.rgba2rgb(image))
        bin = image_gray > 0.1
        info_c, ep_c, bp_c, length_c, root_c = parse_dataframe(bin)
        real_bp.append(bp_c)
        real_ep.append(ep_c)

    return real_bp, real_ep



def calculate_cost(day_syn_bp, day_syn_ep,  real_bp, real_ep):
    cost_ = make_matrix(day_syn_ep, day_syn_bp, real_ep, real_bp)
    index = make_index(cost_)
    flag = 0

    distance_cost = []
    for i in range(0, len(index)):
        real_index = index[i][0]
        syn_index = index[i][1]

        if ((real_index < len(real_ep)) & (syn_index < len(day_syn_ep))):
            distance_cost.append(
                distance.euclidean(day_syn_ep[syn_index], real_ep[real_index]) + \
                distance.euclidean(day_syn_bp[syn_index], real_bp[real_index]))
        else:
            flag = flag + 1

    while (flag > 0):
        distance_cost.append(max(distance_cost))
        flag = flag - 1

    return sum(distance_cost)



def read_syn_plants(plants):
    f = open(file_path + plants + "/output.txt", "r")
    lines = f.readlines()
    day_temp = 0
    syn_bp = []
    syn_ep = []
    index = 0
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



def calculate_each_plant_cost(real_bp, real_ep):

    syn_plants = os.listdir(file_path)
    size = len(syn_plants)
    cost = np.zeros((size, 1), dtype=float)

    index_min_cost = np.zeros((size, 1), dtype=float)

    #print(real_bp)


    for j in range(0, len(syn_plants)):
        plant = syn_plants[j]
        syn_bp, syn_ep = read_syn_plants(plant)
        

        #start_l2 = timer()

        
        cost_plant = []
        for i in range(0, len(min([syn_bp, syn_ep, real_bp, real_ep]))):
            cost_day = calculate_cost(syn_bp[i], syn_ep[i], real_bp[i], real_ep[i])
            cost_plant.append(cost_day)

        cost[j] = sum(cost_plant)


        #end = timer()
        #print(f"Runtime of the program for cost function second loop is {end - start_l2}")

        #print(plant, cost[j], index_min_cost[j])


    if cost.size > 0:
        min_index = np.argmin(cost)
        f_p = open("./plants_cost_min.txt", "a")
        f_p.write(str(syn_plants[min_index]) + " " + str(cost[min_index]) + "\n")
        f_p.close()
        
    #print(syn_plants[min_index], cost[min_index])
    
    return syn_plants, cost



def read_parameters_from_files(plant_name):
    f = open("./parameter_values.txt", "r")
    lines = f.readlines()
    flag = 0
    i = 0
    parameter_value = []

    for line in lines:
        temp = line.split()
        if ((flag == 1) & (i < 12)):
            parameter_value.append(temp[1])
            i = i + 1
        if temp[0] == plant_name:
            flag = 1

        if ((flag == 1) & (i == 12)):
            f.close()
            return parameter_value


def read_parameters(syn_plants, sorted_index):
    parameter = np.zeros((size_x, parameter_number), dtype=float)

    for i in range(0, size_x):
        plant_name = syn_plants[sorted_index[i]]
        parameter_value = read_parameters_from_files(plant_name)

        for j in range(0, parameter_number):
            parameter[i, j] = parameter_value[j]



# --- From utils_nn.py ---

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

def setup_training_csv(model_name):
    """Setup CSV file for training logs with proper headers for aggregate stats"""
    csv_file = model_name + ".csv"
    
    # Ensure directory exists
    directory = os.path.dirname(csv_file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Check if header exists
    write_header = not os.path.exists(csv_file)
    if write_header:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            # Comprehensive header for ML analysis
            writer.writerow([
                "samples_processed", "epoch", "timestamp",
                "train_loss", "val_loss",
                "val_mae", "val_mse", "val_rmse",
                "val_rel_error_mean", "val_rel_error_median", 
                "val_accuracy_1pct"
            ])
    
    return csv_file

def log_training_stats(csv_file, samples_processed, epoch, train_loss, val_stats):
    """Log aggregate training statistics to CSV"""
    timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            samples_processed, epoch, timestamp,
            f"{train_loss:.6f}", 
            f"{val_stats['loss']:.6f}",
            f"{val_stats['mae']:.6f}",
            f"{val_stats['mse']:.6f}",
            f"{val_stats['rmse']:.6f}",
            f"{val_stats['rel_err_mean']:.6f}",
            f"{val_stats['rel_err_median']:.6f}",
            f"{val_stats['accuracy']:.2f}"
        ])

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
    Runs inside output_dir to prevent CWD file collisions (e.g. leafposition.dat).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Compile project if needed (in original CWD)
    if not os.path.exists("project"):
        ret = os.system("g++ -o project -Wall -Wextra lsystem/project.cpp -lm")
        if ret != 0:
            print("Error: Compilation of lsystem/project.cpp failed. Exiting.")
            sys.exit(1)

    # Get absolute paths to run safely from output_dir
    cwd = os.getcwd()
    abs_output_dir = os.path.abspath(output_dir)
    abs_param_file = os.path.abspath(param_file)
    project_exe = os.path.join(cwd, "project")
    
    # Function to get lsystem file paths
    def ls(f): return os.path.join(cwd, "lsystem", f)

    # Build lpfg command components (Using absolute paths)
    lpfg_args = [
        "lpfg", 
        "-w", "306", "256", 
        ls("lsystem.l"), 
        ls("view.v"), 
        ls("materials.mat"), 
        ls("contours.cset"), 
        ls("functions.fset"), 
        ls("functions.tset"), 
        abs_param_file
    ]

    # Run lpfg inside output_dir
    # This ensures leafposition.dat is created inside output_dir, not CWD
    log_file = os.path.join(abs_output_dir, "lpfg_log.txt")
    with open(log_file, "w") as f_log:
        process = subprocess.Popen(lpfg_args, cwd=abs_output_dir, stdout=f_log, stderr=subprocess.STDOUT)
        process.wait()

    # Run project executable to process leafposition.dat
    # Expected input: ./project Width Height InputFile
    # InputFile is now in abs_output_dir/leafposition.dat
    leafval_file = "leafposition.dat" 
    output_file = os.path.join(abs_output_dir, "output.txt")
    
    with open(output_file, "w") as f_out:
        # execute project binary using absolute path
        p_args = [project_exe, "2454", "2056", leafval_file]
        p_proc = subprocess.Popen(p_args, cwd=abs_output_dir, stdout=f_out)
        p_proc.wait()

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
        ret = os.system("g++ -o project -Wall -Wextra lsystem/project.cpp -lm")
        if ret != 0:
            print("Error: Compilation of lsystem/project.cpp failed. Exiting.")
            sys.exit(1)
    
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