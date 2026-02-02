import os
import csv
import torch
import shutil
import argparse
import numpy as np
import plant_comparison_nn
from utils_nn import build_random_parameter_file, generate_plant, read_syn_plant
from plant_comparison_nn import read_real_plants, calculate_cost

def generate_dataset_split(split_name, size, real_bp, real_ep, output_dir):
    """
    Generates a dataset split (Train, Val, or Test).
    """
    print(f"\nGeneraring {split_name} dataset ({size} samples)...")
    
    # Create directories
    split_dir = os.path.join(output_dir, split_name)
    structures_dir = os.path.join(split_dir, "structures")
    if not os.path.exists(structures_dir):
        os.makedirs(structures_dir)
    
    csv_file = os.path.join(output_dir, f"{split_name}.csv")
    
    # Initialize CSV with header
    # Header: ID, Total_Cost, Param_0 ... Param_12
    header = ["id", "cost"] + [f"param_{i}" for i in range(13)]
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
    temp_param_file = "temp_dataset_params.vset"
    temp_output_dir = "temp_dataset_output"
    
    for i in range(size):
        # 1. Generate Random Parameters
        params = build_random_parameter_file(temp_param_file)
        
        # 2. Generate Synthetic Plant (L-system)
        # This runs lpfg and the project converter
        try:
            generate_plant(temp_param_file, temp_output_dir)
            
            # 3. Read the Generated Structure (BP and EP)
            # generate_plant puts output in temp_output_dir/output.txt
            syn_bp, syn_ep = read_syn_plant(os.path.join(temp_output_dir, "output.txt"))
            
            # 4. Calculate True Cost vs Real Plant
            # We sum the cost over all overlapping days
            total_cost = 0.0
            # Ensure we strictly follow the availability of days in both
            num_days = min(len(syn_bp), len(real_bp))
            
            if num_days == 0:
                print(f"Warning: Sample {i} produced no valid structure overlapping with real plant.")
                total_cost = 1e6 # High penalty for failure
            else:
                for day in range(num_days):
                    # calculate_cost takes (syn_bp, syn_ep, real_bp, real_ep) for a SINGLE day
                    day_cost = calculate_cost(syn_bp[day], syn_ep[day], real_bp[day], real_ep[day])
                    total_cost += day_cost
            
            # 5. Save Data
            
            # Save to CSV
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                row = [i, total_cost] + params
                writer.writerow(row)
                
            # Save Structure separately (for the Structure Generation Network training)
            # We save the raw points as a dictionary of tensors or lists
            structure_data = {
                "bp": syn_bp,
                "ep": syn_ep
            }
            torch.save(structure_data, os.path.join(structures_dir, f"structure_{i}.pt"))
            
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{size} samples.")
                
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue

    # Cleanup
    if os.path.exists(temp_param_file):
        os.remove(temp_param_file)
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    print(f"Finished {split_name} dataset.")

def main():
    DEFAULT_PLANT = "Plant_063-32"
    DEFAULT_TRAIN = 100
    DEFAULT_VAL = 20
    DEFAULT_TEST = 20
    DEFAULT_OUT = "Datasets"

    parser = argparse.ArgumentParser(description="Generate datasets for PhytomorphicNN")
    parser.add_argument("--plant", type=str, default=DEFAULT_PLANT, help=f"Name of the real plant folder (default: {DEFAULT_PLANT})")
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN, help=f"Number of training samples (default: {DEFAULT_TRAIN})")
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL, help=f"Number of validation samples (default: {DEFAULT_VAL})")
    parser.add_argument("--test_size", type=int, default=DEFAULT_TEST, help=f"Number of test samples (default: {DEFAULT_TEST})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT, help=f"Output directory (default: {DEFAULT_OUT})")
    
    args = parser.parse_args()
    
    # Setup Real Plant Path logic in plant_comparison_nn
    # Note: plant_comparison_nn uses global variables, so we modify them here if possible
    # or ensure the path structure matches what it expects.
    # By default, it looks in ./Original_Images/{plant_name}
    
    print(f"Target Real Plant: {args.plant}")
    
    # Update the module-level variables in plant_comparison_nn to ensure the correct path is used
    plant_comparison_nn.real_plant_name = args.plant
    # CRITICAL: We must update plant_image_path because it was constructed at import time
    plant_comparison_nn.plant_image_path = os.path.join(plant_comparison_nn.plant_images_path, args.plant)
    
    print("Reading real plant data...")
    try:
        real_bp, real_ep = read_real_plants()
        print(f"Successfully loaded real plant data ({len(real_bp)} days).")
    except Exception as e:
        print(f"Failed to read real plant data: {e}")
        print("Make sure 'Original_Images/' directory exists and contains the plant folder.")
        return

    # Create main output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Generate Splits
    if args.train_size > 0:
        generate_dataset_split("Train", args.train_size, real_bp, real_ep, args.output_dir)
    
    if args.val_size > 0:
        generate_dataset_split("Validation", args.val_size, real_bp, real_ep, args.output_dir)
        
    if args.test_size > 0:
        generate_dataset_split("Test", args.test_size, real_bp, real_ep, args.output_dir)
        
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()
