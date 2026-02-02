"""
Hierarchical surrogate neural network for plant cost prediction.
Decomposes the problem into specialized modules:
1. Structure Generation Network (generates branch/end points from parameters)
2. Hungarian Assignment Network (learns optimal assignment patterns)
3. Cost Aggregation Network (combines assignments into final cost)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time as t
import csv
import numpy as np
from plant_comparison_nn import read_real_plants
from utils_nn import (build_random_parameter_file, compute_normalization_stats,
                      clear_surrogate_dir, setup_training_csv, 
                      log_training_step, print_training_progress,
                      generate_and_evaluate)

model_name = "data/surrogate_model.pt"
accuracy_threshold = 0.01

class StructureGenerationNet(nn.Module):
    """Generates plant structure points (branch points and end points) from L-system parameters"""
    def __init__(self, input_dim=13, max_points=50):
        super().__init__()
        self.max_points = max_points
        
        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Branch point generation (x, y coordinates + existence probability)
        self.bp_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_points * 3)  # x, y, existence_prob for each point
        )
        
        # End point generation (x, y coordinates + existence probability)
        self.ep_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_points * 3)  # x, y, existence_prob for each point
        )
        
    def forward(self, x):
        features = self.feature_net(x)
        
        # Generate branch points
        bp_raw = self.bp_net(features).reshape(-1, self.max_points, 3)
        bp_coords = bp_raw[:, :, :2] * 200  # Scale coordinates to reasonable range
        bp_probs = torch.sigmoid(bp_raw[:, :, 2])  # Existence probabilities
        
        # Generate end points
        ep_raw = self.ep_net(features).reshape(-1, self.max_points, 3)
        ep_coords = ep_raw[:, :, :2] * 200  # Scale coordinates to reasonable range
        ep_probs = torch.sigmoid(ep_raw[:, :, 2])  # Existence probabilities
        
        return bp_coords, bp_probs, ep_coords, ep_probs

class HungarianAssignmentNet(nn.Module):
    """Learns to predict optimal assignment patterns and costs"""
    def __init__(self, max_points=50):
        super().__init__()
        self.max_points = max_points
        
        # Process pairs of structures (synthetic vs real)
        self.structure_encoder = nn.Sequential(
            nn.Linear(max_points * 8, 256),  # bp_syn, ep_syn, bp_real, ep_real (flattened: 4 * max_points * 2)
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Predict assignment matrix (soft assignment weights)
        self.assignment_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, max_points * max_points),  # Assignment matrix
            nn.Softmax(dim=-1)
        )
        
        # Predict assignment costs
        self.cost_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Remove Softplus to allow lower predictions
        )
        
    def forward(self, bp_syn, ep_syn, bp_real, ep_real):
        batch_size = bp_syn.size(0)
        
        # Flatten and concatenate structures
        structure_features = torch.cat([
            bp_syn.reshape(batch_size, -1),
            ep_syn.reshape(batch_size, -1),
            bp_real.reshape(batch_size, -1),
            ep_real.reshape(batch_size, -1)
        ], dim=1)
        
        encoded = self.structure_encoder(structure_features)
        
        # Predict assignment matrix and total cost
        assignment_weights = self.assignment_net(encoded).reshape(batch_size, self.max_points, self.max_points)
        total_cost = self.cost_net(encoded)
        
        return assignment_weights, total_cost

class CostAggregationNet(nn.Module):
    """Aggregates costs across multiple days and assignment patterns"""
    def __init__(self, max_days=26):
        super().__init__()
        self.max_days = max_days
        
        # Process temporal sequence of costs
        self.temporal_net = nn.Sequential(
            nn.Linear(max_days, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Remove Softplus to allow lower predictions
        )
        
    def forward(self, daily_costs):
        # daily_costs: [batch_size, num_days]
        return self.temporal_net(daily_costs)

class HierarchicalPlantSurrogateNet(nn.Module):
    """Hierarchical network combining all modules"""
    def __init__(self, input_dim=13, max_points=50, max_days=26):
        super().__init__()
        self.structure_gen = StructureGenerationNet(input_dim, max_points)
        self.hungarian_net = HungarianAssignmentNet(max_points)
        self.cost_aggregator = CostAggregationNet(max_days)
        # Retrieve normalization stats for inputs and outputs
        input_mean, input_std, output_mean, output_std = compute_normalization_stats()
        self.input_mean = torch.tensor(input_mean, dtype=torch.float32)
        self.input_std = torch.tensor(input_std, dtype=torch.float32)
        self.output_mean = torch.tensor(output_mean, dtype=torch.float32)
        self.output_std = torch.tensor(output_std, dtype=torch.float32)
        
    def forward(self, x, real_bp_batch=None, real_ep_batch=None):
        batch_size = x.size(0)
        
        # Normalize inputs
        x_norm = (x - self.input_mean) / self.input_std
        
        # Generate synthetic plant structures
        bp_syn, bp_probs, ep_syn, ep_probs = self.structure_gen(x_norm)
        
        if real_bp_batch is None or real_ep_batch is None:
            # return structure predictions when no real plant is provided
            return bp_syn, bp_probs, ep_syn, ep_probs
        
        # Compute Hungarian assignment and costs for each day
        daily_costs = []
        
        for day in range(real_bp_batch.size(1)):  # Iterate over days
            bp_real_day = real_bp_batch[:, day, :, :]  # [batch_size, max_points, 2]
            ep_real_day = real_ep_batch[:, day, :, :]  # [batch_size, max_points, 2]
            
            # Get assignment and cost for this day
            assignment_weights, day_cost = self.hungarian_net(bp_syn, ep_syn, bp_real_day, ep_real_day)
            daily_costs.append(day_cost)
        
        # Stack daily costs and aggregate
        daily_costs_tensor = torch.stack(daily_costs, dim=1).squeeze(-1)  # [batch_size, num_days]
        
        # Pad or truncate to max_days
        if daily_costs_tensor.size(1) < 26:
            padding = torch.zeros(batch_size, 26 - daily_costs_tensor.size(1))
            daily_costs_tensor = torch.cat([daily_costs_tensor, padding], dim=1)
        else:
            daily_costs_tensor = daily_costs_tensor[:, :26]
        
        # Final cost aggregation
        final_cost = self.cost_aggregator(daily_costs_tensor)
        
        # Denormalize outputs with bias correction for low predictions
        denorm_cost = final_cost * self.output_std + self.output_mean
        
        # Apply a learnable bias correction to help with low-cost predictions
        # This helps the model overcome systematic bias toward higher values
        bias_correction = torch.where(denorm_cost < 60000, 
                                    -1000 * torch.sigmoid((60000 - denorm_cost) / 5000), 
                                    torch.zeros_like(denorm_cost))
        
        return denorm_cost + bias_correction

def prepare_real_plant_batch(real_bp, real_ep, max_points=50):
    """Convert real plant data to fixed-size tensors for batch processing"""
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

def hierarchical_loss_function(pred_cost, true_cost, bp_syn, bp_probs, ep_syn, ep_probs, real_bp, real_ep):
    """Multi-component loss function for hierarchical training"""
    
    # Primary cost prediction loss
    cost_loss = F.mse_loss(pred_cost, true_cost)
    
    # Structure generation loss (encourage reasonable point distributions)
    structure_loss = 0.0
    
    # Existence probability regularization (prevent too many/few points)
    bp_count_target = torch.tensor([min(len(day_bp), 50) for day_bp in real_bp]).float().mean()
    ep_count_target = torch.tensor([min(len(day_ep), 50) for day_ep in real_ep]).float().mean()
    
    bp_count_pred = bp_probs.sum()
    ep_count_pred = ep_probs.sum()
    
    count_loss = F.mse_loss(bp_count_pred, bp_count_target) + F.mse_loss(ep_count_pred, ep_count_target)
    
    # Reduced spatial distribution loss (to test lower positive bias)
    coord_regularization = 0.003 * (torch.var(bp_syn) + torch.var(ep_syn))
    
    total_loss = cost_loss + 0.1 * count_loss + 0.003 * coord_regularization
    
    return total_loss, cost_loss, count_loss, coord_regularization


if __name__ == "__main__":
    import time
    
    # Get number of runs from command line, default to 1000
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            print("Invalid argument for number of runs, using default 1000.")
            num_runs = 1000
    else:
        num_runs = 1000

    # Setup training CSV and get previous state
    start_run, prev_losses, csv_file = setup_training_csv(model_name)
            
    model = HierarchicalPlantSurrogateNet()
    initial_lr = 1e-4  # Lower learning rate for more complex model
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
        # Learning rate remains constant for all training

    # Always load if exists, but always train
    if os.path.exists(model_name):
        try:
            model.load_state_dict(torch.load(model_name))
            print(f"Loaded existing model from {model_name}")
        except Exception as e:
            print(f"Could not load existing model: {e}. Creating new model.")
    else:
        print(f"No existing model found at {model_name}, creating new model.")
    
    model.train()
    print("Reading real plants...")
    real_bp, real_ep = read_real_plants()
    real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep)
    print(f"Starting hierarchical training with {num_runs} plants...")
    
    total_loss = sum(prev_losses)
    total_samples = len(prev_losses)
    avg_loss_change_history = []
    rel_error_history = []

    start_time = time.time()

    for idx in range(num_runs):
        iter_start_time = time.time()
        clear_surrogate_dir()
        
        # 1. Generate random parameters and write to file
        params = build_random_parameter_file("surrogate_params.vset")
        params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
        params_tensor = params_tensor.float()  # Ensure float32 dtype
        # 2. Get true cost from L-system
        true_cost = generate_and_evaluate("surrogate_params.vset", real_bp, real_ep)
        true_cost_tensor = torch.tensor([[true_cost]], dtype=torch.float32)
        try:
            pred_cost = model(params_tensor, real_bp_batch, real_ep_batch)
        except RuntimeError as e:
            print(f"Error in forward pass: {e}")
            print(f"params_tensor shape: {params_tensor.shape}")
            print(f"real_bp_batch shape: {real_bp_batch.shape}")
            print(f"real_ep_batch shape: {real_ep_batch.shape}")
            # Check structure generation output
            with torch.no_grad():
                params_norm = (params_tensor - model.input_mean) / model.input_std
                bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen(params_norm)
                print(f"bp_syn shape: {bp_syn.shape}")
                print(f"ep_syn shape: {ep_syn.shape}")
            raise e
        # Get intermediate outputs for loss computation
        bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen(params_tensor)
        # 4. Compute hierarchical loss
        total_loss_val, cost_loss, count_loss, coord_reg = hierarchical_loss_function(
            pred_cost, true_cost_tensor, bp_syn, bp_probs, ep_syn, ep_probs, real_bp, real_ep
        )
        # 5. Backpropagation
        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()
        # 6. Update statistics
        total_loss += cost_loss.item()  # Track only cost loss for comparison
        total_samples += 1
        avg_loss = total_loss / total_samples
        timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
        
        if total_samples > 1:
            prev_avg_loss = (total_loss - cost_loss.item()) / (total_samples - 1)
            avg_loss_change = avg_loss - prev_avg_loss
        else:
            avg_loss_change = 0.0
            
        avg_loss_change_history.append(avg_loss_change)
        if len(avg_loss_change_history) > 1000:
            avg_loss_change_history.pop(0)
        avg_loss_change_1000 = sum(avg_loss_change_history) / len(avg_loss_change_history)
        
        # Compute relative error
        pred_cost_val = pred_cost.item()
        true_cost_val = true_cost_tensor.item()
        rel_error = abs(pred_cost_val - true_cost_val) / (abs(true_cost_val) + 1e-8)
        rel_error_history.append(rel_error)
        if len(rel_error_history) > 1000:
            rel_error_history.pop(0)

        # Accuracy: percent of rel_error < 0.1 in last 1000 samples
        accuracy_1000 = 100.0 * sum(e < accuracy_threshold for e in rel_error_history) / len(rel_error_history)

        # Learning rate remains constant; no adaptation
        current_lr = optimizer.param_groups[0]['lr']

        # Print progress with meaningful metrics
        print_training_progress(idx, num_runs, start_run, avg_loss, total_loss_val.item(), 
                               cost_loss.item(), accuracy_1000, current_lr, start_time,
                               rel_error=rel_error, pred_cost=pred_cost_val, true_cost=true_cost_val)

        clear_surrogate_dir()

        # Write to CSV with hierarchical loss components
        run_number = start_run + idx + 1
        log_training_step(csv_file, run_number, total_loss_val.item(), cost_loss.item(), 
                         coord_reg.item(), pred_cost.item(), true_cost_val, params, 
                         avg_loss, avg_loss_change)
    print()  # Newline after progress bar
    torch.save(model.state_dict(), model_name)
    print(f"Trained and saved hierarchical model to {model_name}")
    print(f"Total samples: {total_samples}, Total loss: {total_loss:.4f}, Average loss: {avg_loss:.4f}")
    
    # Print model architecture summary
    print("\nHierarchical Model Architecture:")
    print(f"- Structure Generation Net: {sum(p.numel() for p in model.structure_gen.parameters())} parameters")
    print(f"- Hungarian Assignment Net: {sum(p.numel() for p in model.hungarian_net.parameters())} parameters") 
    print(f"- Cost Aggregation Net: {sum(p.numel() for p in model.cost_aggregator.parameters())} parameters")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optional: Save intermediate outputs for analysis
    if total_samples > 0:
        model.eval()
        with torch.no_grad():
            # Generate sample structure predictions
            sample_params = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            # Normalize inputs for structure generation
            params_norm = (sample_params - model.input_mean) / (model.input_std + 1e-8)
            bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen(params_norm)
            
            print(f"\nSample structure generation:")
            print(f"- Predicted branch points: {bp_probs.sum().item():.1f}")
            print(f"- Predicted end points: {ep_probs.sum().item():.1f}")
            print(f"- Branch point coordinates range: [{bp_syn.min().item():.1f}, {bp_syn.max().item():.1f}]")
            print(f"- End point coordinates range: [{ep_syn.min().item():.1f}, {ep_syn.max().item():.1f}]")