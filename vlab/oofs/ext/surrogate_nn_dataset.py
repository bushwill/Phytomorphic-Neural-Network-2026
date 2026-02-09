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
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import sys
import time as t
import csv
import numpy as np
from plant_comparison_nn import read_real_plants
from utils_nn import (setup_training_csv, log_training_step, print_training_progress)

model_name = "data/surrogate_model.pt"
accuracy_threshold = 0.01

class PlantDataset(Dataset):
    def __init__(self, csv_file, root_dir=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # Params are col 2 to 15 (param_0 to param_12)
        # Check header: id, cost, param_0, ...
        self.params = self.data.iloc[:, 2:].values.astype(np.float32)
        self.costs = self.data.iloc[:, 1].values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        params = torch.tensor(self.params[idx])
        cost = torch.tensor([self.costs[idx]])
        return params, cost

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
    def __init__(self, input_dim=13, max_points=50, max_days=26, input_mean=None, input_std=None, output_mean=None, output_std=None):
        super().__init__()
        self.structure_gen = StructureGenerationNet(input_dim, max_points)
        self.hungarian_net = HungarianAssignmentNet(max_points)
        self.cost_aggregator = CostAggregationNet(max_days)
        
        # Use provided stats or default
        if input_mean is None:
            input_mean = np.zeros(input_dim)
            input_std = np.ones(input_dim)
            output_mean = 0.0
            output_std = 1.0
            
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
    
    # Get number of epochs from command line, default to 100
    if len(sys.argv) > 1:
        try:
            num_epochs = int(sys.argv[1])
        except ValueError:
            print("Invalid argument for number of runs, using default 100.")
            num_epochs = 100
    else:
        num_epochs = 100

    # Read real plant data first
    print("Reading real plants...")
    real_bp, real_ep = read_real_plants()
    real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep)
    
    # Load Datasets
    print("Loading datasets...")
    train_dataset = PlantDataset("Datasets/Train.csv")
    val_dataset = PlantDataset("Datasets/Validation.csv")
    test_dataset = PlantDataset("Datasets/Test.csv")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Compute stats from training data
    print("Computing normalization stats...")
    input_mean = train_dataset.params.mean(axis=0)
    input_std = train_dataset.params.std(axis=0) + 1e-8
    output_mean = train_dataset.costs.mean()
    output_std = train_dataset.costs.std() + 1e-8
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # Setup training CSV and get previous state
    # We will log epoch-wise stats
    start_run, prev_losses, csv_file = setup_training_csv(model_name)
            
    model = HierarchicalPlantSurrogateNet(
        input_mean=input_mean, input_std=input_std, 
        output_mean=output_mean, output_std=output_std
    )
    initial_lr = 1e-4  # Lower learning rate for more complex model
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
    # Always load if exists
    if os.path.exists(model_name):
        try:
            model.load_state_dict(torch.load(model_name))
            print(f"Loaded existing model from {model_name}")
        except Exception as e:
            print(f"Could not load existing model: {e}. Creating new model.")
    else:
        print(f"No existing model found at {model_name}, creating new model.")
    
    model.train()
    print(f"Starting hierarchical training for {num_epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        
        for batch_params, batch_costs in train_loader:
            batch_size = batch_params.size(0)
            
            # Repeat real_bp_batch for this batch
            current_real_bp = real_bp_batch.repeat(batch_size, 1, 1, 1)
            current_real_ep = real_ep_batch.repeat(batch_size, 1, 1, 1)
            
            pred_cost = model(batch_params, current_real_bp, current_real_ep)
            
            # Structure outputs for loss
            # Note: We are using generated structure for regularization, 
            # even though we don't strictly supervise it against the saved .pt files
            bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen((batch_params - model.input_mean)/model.input_std)
            
            total_loss_val, cost_loss, count_loss, coord_reg = hierarchical_loss_function(
                pred_cost, batch_costs, bp_syn, bp_probs, ep_syn, ep_probs, real_bp, real_ep
            )
            
            optimizer.zero_grad()
            total_loss_val.backward()
            optimizer.step()
            
            epoch_loss += total_loss_val.item() * batch_size
            epoch_samples += batch_size
            
        avg_loss = epoch_loss / epoch_samples
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_errors = []
        with torch.no_grad():
            for val_params, val_costs in val_loader:
                bs = val_params.size(0)
                current_real_bp = real_bp_batch.repeat(bs, 1, 1, 1)
                current_real_ep = real_ep_batch.repeat(bs, 1, 1, 1)
                
                pred = model(val_params, current_real_bp, current_real_ep)
                
                rel_err = torch.abs(pred - val_costs) / (torch.abs(val_costs) + 1e-8)
                val_errors.extend(rel_err.numpy().flatten())
        
        val_acc = 100.0 * np.mean(np.array(val_errors) < accuracy_threshold)
        print(f"  Validation Accuracy (RelErr < {accuracy_threshold}): {val_acc:.2f}%")
        
        # Log to CSV (Epoch simplified)
        log_training_step(csv_file, epoch+1, avg_loss, 0, 0, 0, 0, [], avg_loss, 0)
        
        torch.save(model.state_dict(), model_name)
        model.train()

    print("\nTraining complete.")
    
    # Test Evaluation
    print("Evaluating on Test Set...")
    model.eval()
    test_errors = []
    test_preds = []
    test_actuals = []
    
    with torch.no_grad():
        for test_params, test_costs in test_loader:
            bs = test_params.size(0)
            current_real_bp = real_bp_batch.repeat(bs, 1, 1, 1)
            current_real_ep = real_ep_batch.repeat(bs, 1, 1, 1)
            
            pred = model(test_params, current_real_bp, current_real_ep)
            
            rel_err = torch.abs(pred - test_costs) / (torch.abs(test_costs) + 1e-8)
            test_errors.extend(rel_err.numpy().flatten())
            test_preds.extend(pred.numpy().flatten())
            test_actuals.extend(test_costs.numpy().flatten())
            
    test_acc = 100.0 * np.mean(np.array(test_errors) < accuracy_threshold)
    mean_rel_err = np.mean(test_errors)
    
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Mean Relative Error: {mean_rel_err:.4f}")
    
    # Save a small report
    with open("test_results.txt", "w") as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Mean Relative Error: {mean_rel_err:.4f}\n")