"""
This script checks multiple surrogate models for optimizing plant parameters.
It reads real plant data, trains an optimizer network for each surrogate model,
and evaluates the optimized parameters against the real plant data.
It saves the best parameters and their corresponding cost.
Supports normal, boundary, and hierarchical surrogate models.
"""

import os
import sys
import torch
import torch.nn as nn
import shutil
import csv
import time as t
from utils_nn import build_parameter_file, generate_and_evaluate_in_dir, compute_normalization_stats
from plant_comparison_nn import calculate_cost, read_real_plants

# Import all surrogate model types
try:
    from surrogate_nn import HierarchicalPlantSurrogateNet
except ImportError:
    print("Warning: Could not import HierarchicalPlantSurrogateNet from surrogate_nn")
    HierarchicalPlantSurrogateNet = None

# Import PlantSurrogateNet for normal model support
try:
    from benchmark1_nn import PlantSurrogateNet
except ImportError:
    print("Warning: Could not import PlantSurrogateNet from benchmark_nn")
    PlantSurrogateNet = None

# Customizable variables:
param_min = torch.tensor([8.0, 2.8, -110.0, -4.0, 125.0, 3.0, 0.48, 0.8, 80.0, 170.0, 0.6, 0.88, 0.48])
param_max = torch.tensor([12.0, 3.2, 110.0, 4.0, 145.0, 7.0, 0.52, 1.2, 100.0, 190.0, 0.8, 0.92, 0.52])
weight_decay = 1e-5
num_restarts = 10
batch_size = 32
diversity_amount = 0.1
accuracy_threshold = 0.01
boundary_penalty_weight = 0.1   # New: weight for soft boundary penalty

directory = "data/Normal Data/Final/"
optimizer_directory = "data/Optimizer/"

# Create optimizer directory if it doesn't exist
if not os.path.exists(optimizer_directory):
    os.makedirs(optimizer_directory)

surrogate_models = [
    ["surrogate_model.pt", "hierarchical", HierarchicalPlantSurrogateNet],  # Mutant2 hierarchical model
    ["benchmark1_batch_surrogate_model.pt", "benchmark1-batch", PlantSurrogateNet],  # Batch-16 normal model
    ["benchmark2_online_surrogate_model.pt", "benchmark2-online", PlantSurrogateNet],  # Batch-16 normal model
]

for i in range(len(surrogate_models)):
    surrogate_models[i][0] = directory + surrogate_models[i][0]

def prepare_real_plant_batch(real_bp, real_ep, max_points=50):
    """Convert real plant data to fixed-size tensors for hierarchical model"""
    num_days = len(real_bp)
    
    # Initialize tensors
    bp_batch = torch.zeros(1, num_days, max_points, 2)
    ep_batch = torch.zeros(1, num_days, max_points, 2)
    
    for day in range(num_days):
        # Branch points
        bp_day = real_bp[day]
        if len(bp_day) > 0:
            bp_array = torch.tensor(bp_day[:max_points], dtype=torch.float32)
            bp_batch[0, day, :min(len(bp_day), max_points), :] = bp_array
        
        # End points
        ep_day = real_ep[day]
        if len(ep_day) > 0:
            ep_array = torch.tensor(ep_day[:max_points], dtype=torch.float32)
            ep_batch[0, day, :min(len(ep_day), max_points), :] = ep_array
    
    return bp_batch, ep_batch

def load_surrogate_model(model_path, model_type, model_class):
    """Load surrogate model based on its type"""
    if model_class is None:
        print(f"Model class not available for {model_type}")
        return None
        
    try:
        if model_type == "hierarchical":
            model = model_class()
        else:
            model = model_class()
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def evaluate_surrogate_model(surrogate, params_batch, model_type, real_bp_batch=None, real_ep_batch=None):
    """Evaluate surrogate model based on its type"""
    try:
        if model_type == "hierarchical":
            if real_bp_batch is not None and real_ep_batch is not None:
                # For hierarchical model, we need real plant data to get proper cost predictions
                # The model was trained to compare synthetic structures to real plants
                output = surrogate(params_batch, real_bp_batch, real_ep_batch)
                # If output is a tuple, extract the cost tensor
                if isinstance(output, tuple):
                    cost = output[0]
                else:
                    cost = output
                return cost
            else:
                # Fallback: structure complexity approximation (not ideal)
                with torch.no_grad():
                    bp_syn, bp_probs, ep_syn, ep_probs = surrogate(params_batch)
                    structure_complexity = (bp_probs.mean(dim=-1) + ep_probs.mean(dim=-1)).mean(dim=-1, keepdim=True)
                    return structure_complexity * 100  # Scale to reasonable cost range
        else:
            # For normal and boundary models
            return surrogate(params_batch)
    except Exception as e:
        print(f"Error evaluating surrogate model: {e}")
        return torch.zeros(params_batch.size(0), 1)

class OptimizerNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()  # constrain to [0,1]
        )
        self.param_min = param_min
        self.param_max = param_max

    def forward(self, x):
        out = self.net(x)
        return out * (self.param_max - self.param_min) + self.param_min

def clear_dir(output_dir):
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(output_dir)

print("Reading real plants...")
real_bp, real_ep = read_real_plants()
real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep)  # For hierarchical model

if len(sys.argv) > 1:
    try:
        num_restarts = int(sys.argv[1])
    except ValueError:
        print("Invalid argument for number of restarts, using default 5.")
        
num_runs = 1000

# Track the best results across all models
global_best_cost = float('inf')
global_best_params = None
global_best_model = None

# Loop over each surrogate model configuration
for model_path, model_type, model_class in surrogate_models:
    if os.path.exists(model_path):
        print(f"\nProcessing surrogate model: {model_path} (type: {model_type})")
        
        # Use optimizer directory for optimizer-related files
        model_basename = os.path.basename(model_path)
        optimizer_model_path = os.path.join(optimizer_directory, model_basename.replace(".pt", "_optimizer.pt"))
        csv_file = os.path.join(optimizer_directory, model_basename + ".csv")
        
        # Always overwrite CSV and write header
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true_cost", "pred_cost"] + [f"param_{i}" for i in range(13)])
    
        # Load the surrogate model and set it into eval mode
        surrogate = load_surrogate_model(model_path, model_type, model_class)
        if surrogate is None:
            print(f"Skipping {model_path} - could not load model")
            continue
    
        # Pre-expand real plant data for hierarchical models (needed for proper cost prediction)
        if model_type == "hierarchical":
            real_bp_batch_expanded = real_bp_batch.expand(batch_size, -1, -1, -1)
            real_ep_batch_expanded = real_ep_batch.expand(batch_size, -1, -1, -1)
        else:
            real_bp_batch_expanded = None
            real_ep_batch_expanded = None
    
        best_cost = float('inf')
        best_params = None
    
        for restart in range(num_restarts):
            optimizer_net = OptimizerNet()
            if os.path.exists(optimizer_model_path) and restart == 0:
                optimizer_net.load_state_dict(torch.load(optimizer_model_path))
                print(f"Loaded existing optimizer network from {optimizer_model_path}")
            else:
                print(f"Starting optimizer network from scratch (restart {restart+1}/{num_restarts}).")
    
            optimizer = torch.optim.Adam(optimizer_net.parameters(), lr=1e-2, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)  # More frequent decay
            param_noise_std = 0.02
            
            # Early stopping variables - more aggressive for hierarchical models
            best_loss = float('inf')
            patience = num_runs / 8  # Reduced patience for faster stopping
            no_improve_count = 0

            for step in range(num_runs):
                optimizer.zero_grad()
                noise = torch.rand(batch_size, 1)
                params = optimizer_net(noise)
                # Instead of hard clamping:
                # params_noisy = torch.max(torch.min(params_noisy, param_max), param_min)
                params_noisy = params + torch.randn_like(params) * param_noise_std

                # Calculate soft penalty for exceeding bounds
                excess_low = torch.clamp(param_min - params_noisy, min=0)
                excess_high = torch.clamp(params_noisy - param_max, min=0)
                boundary_penalty = boundary_penalty_weight * ((excess_low**2).mean() + (excess_high**2).mean())

                # Evaluate surrogate model based on its type
                if model_type == "hierarchical":
                    # Hierarchical models need real plant data for proper cost predictions
                    pred_cost = evaluate_surrogate_model(surrogate, params_noisy, model_type, 
                                                       real_bp_batch_expanded, real_ep_batch_expanded)
                else:
                    pred_cost = evaluate_surrogate_model(surrogate, params_noisy, model_type)
                
                loss = pred_cost.mean()
                diversity_loss = -params_noisy.var(dim=0).mean()
                total_loss_val = loss + diversity_amount * diversity_loss + boundary_penalty
                
                total_loss_val.backward()
                optimizer.step()
                scheduler.step()
                
                # Early stopping check
                current_loss = total_loss_val.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count >= patience:
                    print(f"\nEarly stopping at step {step+1} - no improvement for {patience} steps")
                    break
                
                if step % max(1, num_runs // 50) == 0 or step == num_runs - 1:  # More frequent progress updates (every ~20 steps)
                    percent = 100 * (step + 1) / num_runs
                    sys.stdout.write(f"\rRestart {restart+1}/{num_restarts} - Progress: {percent:.1f}% - surrogate cost={pred_cost.mean().item():.4f}")
                    sys.stdout.flush()
    
            noise = torch.rand(1, 1)
            optimized_params_tensor = optimizer_net(noise)
            optimized_params = optimized_params_tensor.detach().numpy().flatten()
            print(f"\nOptimized parameters (restart {restart+1}):", optimized_params)
            # Ensure parameter batch matches real plant batch size for hierarchical model
            if model_type == "hierarchical":
                param_batch = optimized_params_tensor.expand(batch_size, -1)
                pred_cost = evaluate_surrogate_model(
                    surrogate,
                    param_batch,
                    model_type,
                    real_bp_batch_expanded,
                    real_ep_batch_expanded
                )
            else:
                pred_cost = evaluate_surrogate_model(
                    surrogate,
                    torch.tensor(optimized_params, dtype=torch.float32).unsqueeze(0),
                    model_type
                )
            # Extract scalar predicted cost
            if isinstance(pred_cost, torch.Tensor):
                if pred_cost.numel() == 1:
                    pred_cost_val = float(pred_cost.item())
                else:
                    pred_cost_val = float(pred_cost.squeeze().mean().item())
            elif isinstance(pred_cost, (list, tuple)):
                # If tuple, try to extract first element and handle tensor inside
                pc = pred_cost[0]
                if isinstance(pc, torch.Tensor):
                    if pc.numel() == 1:
                        pred_cost_val = float(pc.item())
                    else:
                        pred_cost_val = float(pc.squeeze().mean().item())
                else:
                    pred_cost_val = float(pc)
            else:
                pred_cost_val = float(pred_cost)
            # Evaluate real cost
            param_file = os.path.join(optimizer_directory, f"temp_params.vset")
            build_parameter_file(param_file, optimized_params)
            clear_dir("temp_plant")
            real_cost = generate_and_evaluate_in_dir(param_file, real_bp, real_ep, "temp_plant", cost_fn=calculate_cost)
            print(f"Real cost for optimized parameters (restart {restart+1}): {real_cost:.4f}")
            clear_dir("temp_plant")
            # Log to CSV: real cost, predicted cost, parameters
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([real_cost, pred_cost_val] + list(optimized_params))

            if real_cost < best_cost:
                best_cost = real_cost
                best_params = optimized_params
                torch.save(optimizer_net.state_dict(), optimizer_model_path)
    
        print(f"\nBest real cost for {model_path}: {best_cost:.4f}")
        print("Best optimized parameters:", best_params)
        best_params_file = os.path.join(optimizer_directory, os.path.basename(model_path).replace(".pt", "_optimized_params_best.vset"))
        build_parameter_file(best_params_file, best_params)
        
        # Update global best if this model performed better
        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_params = best_params.copy()
            global_best_model = model_path
            
    else:
        print(f"Surrogate model not found: {model_path}")

# Print overall best results across all models
print(f"\n{'='*60}")
print("OVERALL BEST RESULTS ACROSS ALL MODELS:")
print(f"{'='*60}")
if global_best_model is not None:
    print(f"Best model: {global_best_model}")
    print(f"Best real cost: {global_best_cost:.4f}")
    print("Best optimized parameters:", global_best_params)
    
    # Save the overall best parameters
    overall_best_file = os.path.join(optimizer_directory, "overall_best_optimized_params.vset")
    build_parameter_file(overall_best_file, global_best_params)
    print(f"Overall best parameters saved to: {overall_best_file}")
else:
    print("No models were successfully processed.")
