"""
Hierarchical surrogate neural network for plant cost prediction with Sinkhorn-based Assignment.
Decomposes the problem into specialized modules:
1. Structure Generation Network (generates branch/end points from parameters)
2. Sinkhorn Assignment Network (differentiable, permutation-invariant assignment)
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
import math
import numpy as np
from plant_comparison_nn import read_real_plants
from utils_nn import (setup_training_csv, log_training_stats)

model_name = "data/surrogate_model_sinkhorn.pt"
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

def log_sinkhorn_iterations(log_alpha, n_iters=5):
    """ 
    Perform Sinkhorn normalization in log-space for stability.
    log_alpha: [batch_size, N, N] - Log of the score matrix
    """
    for _ in range(n_iters):
        # Row normalization
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # Column normalization 
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()

class PointSetEncoder(nn.Module):
    """
    Encodes a set of points (branches) in a permutation-invariant way using Attention.
    """
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        # Embed each point independently
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Self-attention to understand context (how this branch relates to others in the same plant)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [batch, num_points, input_dim]
        emb = self.embedding(x)
        
        # Self-attention
        attn_out, _ = self.self_attention(emb, emb, emb)
        return self.norm(emb + attn_out)

class SinkhornAssignmentNet(nn.Module):
    """
    Learns to predict optimal assignment patterns using Sinkhorn iterations.
    Permutation Invariant w.r.t input order of branches.
    Handles 'ghost points' by weighting costs with existence probabilities.
    """
    def __init__(self, max_points=50, feature_dim=64):
        super().__init__()
        self.max_points = max_points
        
        # Encoder for each branch (BPx, BPy, EPx, EPy)
        self.encoder = PointSetEncoder(input_dim=4, hidden_dim=feature_dim)
        
        # Learnable temperature for Sinkhorn (log-space for stability)
        self.log_temperature = nn.Parameter(torch.tensor(0.0)) # exp(0) = 1.0

    def forward(self, bp_syn, ep_syn, bp_real, ep_real, bp_probs_syn=None, ep_probs_syn=None):
        batch_size = bp_syn.size(0)
        
        # 1. Construct Set Features
        # Combine BP and EP into a single feature vector for each branch: [batch, max_points, 4]
        syn_features = torch.cat([bp_syn, ep_syn], dim=-1)
        real_features = torch.cat([bp_real, ep_real], dim=-1)
        
        # 2. Encode Sets (Permutation Invariant / Equivariant)
        # syn_emb: [batch, max_points, feature_dim]
        syn_emb = self.encoder(syn_features)
        real_emb = self.encoder(real_features)
        
        # 3. Compute Similarity Matrix (Score Matrix)
        # Ensure temperature is positive using exp(log_temp)
        temperature = torch.exp(self.log_temperature)
        
        # [batch, N, D] @ [batch, D, N] -> [batch, N, N]
        scores = torch.bmm(syn_emb, real_emb.transpose(1, 2)) / temperature
        
        # 4. Sinkhorn Normalization (Differentiable Assignment)
        # Returns Doubly Stochastic Matrix P [batch, N, N]
        assignment_matrix = log_sinkhorn_iterations(scores, n_iters=5)
        
        # 5. Compute "Physical" Cost Matrix with Ghost Point Handling
        # The actual objective cost we want to minimize: Geometric Distance
        
        # Expand for broadcasting
        syn_bp_exp = bp_syn.unsqueeze(2)
        real_bp_exp = bp_real.unsqueeze(1)
        syn_ep_exp = ep_syn.unsqueeze(2)
        real_ep_exp = ep_real.unsqueeze(1)
        
        bp_dist = torch.norm(syn_bp_exp - real_bp_exp, dim=-1)
        ep_dist = torch.norm(syn_ep_exp - real_ep_exp, dim=-1)
        
        physical_cost_matrix = bp_dist + ep_dist # [batch, N, N]
        
        # Handle Ghost Points: Multiply cost by existence probability
        
        if bp_probs_syn is not None and ep_probs_syn is not None:
             # Combine probabilities (branch exists if both BP and EP exist/are valid)
             # [batch, N, 1]
             point_existence = (bp_probs_syn * ep_probs_syn).unsqueeze(-1)
             
             # Weight the physical cost: 
             # If point exists (1.0), full cost. If not (0.0), cost is 0.
             physical_cost_matrix = physical_cost_matrix * point_existence

        # 6. Expected Cost
        # Sum(P_ij * C_ij)
        total_cost = torch.sum(assignment_matrix * physical_cost_matrix, dim=(-1, -2))
        
        # Reshape to [batch, 1] for compatibility
        return assignment_matrix, total_cost.unsqueeze(-1)

class CostAggregationNet(nn.Module):
    """Aggregates costs across multiple days"""
    def __init__(self, max_days=26):
        super().__init__()
        self.max_days = max_days
        
        # Process temporal sequence of costs
        # Simple weighted sum or MLP
        # The input costs are "Physical Costs" (distances ~1,000-100,000 range), but weights are initialized for normalized inputs (~1)
        # We need BatchNorm to normalize inputs at runtime, or we can just divide by a constant
        # BatchNorm is safer as it learns the true distribution
        self.norm = nn.BatchNorm1d(max_days)
        
        self.temporal_net = nn.Sequential(
            nn.Linear(max_days, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, daily_costs):
        # daily_costs: [batch_size, num_days]
        # Normalize the costs before entering the MLP
        norm_costs = self.norm(daily_costs)
        return self.temporal_net(norm_costs)

class HierarchicalPlantSurrogateNet(nn.Module):
    """Hierarchical network combining all modules with Sinkhorn"""
    def __init__(self, input_dim=13, max_points=50, max_days=26, input_mean=None, input_std=None, output_mean=None, output_std=None):
        super().__init__()
        self.structure_gen = StructureGenerationNet(input_dim, max_points)
        self.sinkhorn_net = SinkhornAssignmentNet(max_points) # Renamed from hungarian_net
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
            return bp_syn, bp_probs, ep_syn, ep_probs
        
        # Batch-process all days simultaneously
        num_days = real_bp_batch.size(1)
        max_points = bp_syn.size(1)
        
        # Expand synthetic structures for each day
        # [batch, points, 2] -> [batch, 1, points, 2] -> [batch, num_days, points, 2] -> [batch*num_days, points, 2]
        bp_syn_expanded = bp_syn.unsqueeze(1).expand(-1, num_days, -1, -1).reshape(-1, max_points, 2)
        ep_syn_expanded = ep_syn.unsqueeze(1).expand(-1, num_days, -1, -1).reshape(-1, max_points, 2)
        bp_probs_expanded = bp_probs.unsqueeze(1).expand(-1, num_days, -1).reshape(-1, max_points)
        ep_probs_expanded = ep_probs.unsqueeze(1).expand(-1, num_days, -1).reshape(-1, max_points)
        
        # Flatten real data
        # [batch, num_days, points, 2] -> [batch*num_days, points, 2]
        real_bp_flat = real_bp_batch.reshape(-1, max_points, 2)
        real_ep_flat = real_ep_batch.reshape(-1, max_points, 2)
        
        # Compute Sinkhorn assignment for all batch-days at once
        _, total_costs_flat = self.sinkhorn_net(
            bp_syn_expanded, ep_syn_expanded, 
            real_bp_flat, real_ep_flat, 
            bp_probs_syn=bp_probs_expanded, ep_probs_syn=ep_probs_expanded
        )
        
        # Reshape costs back to [batch, num_days]
        daily_costs_tensor = total_costs_flat.view(batch_size, num_days)
        
        # Pad or truncate to max_days
        if daily_costs_tensor.size(1) < 26:
            padding = torch.zeros(batch_size, 26 - daily_costs_tensor.size(1))
            daily_costs_tensor = torch.cat([daily_costs_tensor, padding], dim=1)
        else:
            daily_costs_tensor = daily_costs_tensor[:, :26]
        
        # Final cost aggregation
        final_cost = self.cost_aggregator(daily_costs_tensor)
        
        # Denormalize outputs with bias correction
        denorm_cost = final_cost * self.output_std + self.output_mean
        
        # Enforce physical positivity constraint using Softplus
        return F.softplus(denorm_cost)

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
    bp_count_target = torch.tensor([min(len(day_bp), 50) for day_bp in real_bp]).float().mean()
    ep_count_target = torch.tensor([min(len(day_ep), 50) for day_ep in real_ep]).float().mean()
    
    bp_count_pred = bp_probs.sum(dim=1).mean()
    ep_count_pred = ep_probs.sum(dim=1).mean()
    
    count_loss = F.mse_loss(bp_count_pred, bp_count_target) + F.mse_loss(ep_count_pred, ep_count_target)
    
    # Reduced spatial distribution loss (to test lower positive bias)
    # Variance of coordinates (0-200) is high (~3000). 
    # We scale down significantly to match normalized cost loss (~0.01-0.1 range)
    coord_regularization = 1e-5 * (torch.var(bp_syn) + torch.var(ep_syn))
    
    # Count MSE is typically ~10-25. Multiplier 0.005 brings it to ~0.05-0.1 range.
    total_loss = cost_loss + 0.005 * count_loss + coord_regularization
    
    return total_loss, cost_loss, count_loss, coord_regularization


def validate_model(model, val_loader, real_bp_batch, real_ep_batch):
    """
    Runs validation on the validation set and returns metrics.
    """
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_rel_err = 0.0
    num_batches = 0
    total_samples = 0
    
    with torch.no_grad():
        for val_params, val_costs in val_loader:
            batch_size = val_params.size(0)
            
            # Prepare real plant data batches
            current_real_bp = real_bp_batch.repeat(batch_size, 1, 1, 1)
            current_real_ep = real_ep_batch.repeat(batch_size, 1, 1, 1)
            
            # Forward pass
            pred_cost = model(val_params, current_real_bp, current_real_ep)
            
            # Calculate metrics
            batch_loss = F.mse_loss(pred_cost, val_costs)
            batch_mae = F.l1_loss(pred_cost, val_costs)
            batch_rel_err = torch.abs(pred_cost - val_costs) / (torch.abs(val_costs) + 1e-8)
            
            val_loss += batch_loss.item() * batch_size
            val_mae += batch_mae.item() * batch_size
            val_rel_err += batch_rel_err.sum().item()
            
            total_samples += batch_size
            num_batches += 1
            
    avg_loss = val_loss / total_samples
    avg_mae = val_mae / total_samples
    avg_rel_err = val_rel_err / total_samples
    
    model.train()
    
    return {
        'loss': avg_loss,
        'mae': avg_mae,
        'rel_err_mean': avg_rel_err
    }


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

    # Setup training CSV 
    csv_file = setup_training_csv(model_name)
    
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
    
    # Enable anomaly detection for Sinkhorn debugging
    # torch.autograd.set_detect_anomaly(True)
    
    model.train()
    print(f"Starting hierarchical training for {num_epochs} epochs with Sinkhorn Assignment...")
    print("Validating every 100 samples processed.")
    
    start_time = time.time()
    total_samples_processed = 0
    samples_since_validation = 0
    validation_interval = 100
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        for batch_params, batch_costs in train_loader:
            batch_size = batch_params.size(0)
            
            # Repeat real_bp_batch for this batch
            current_real_bp = real_bp_batch.repeat(batch_size, 1, 1, 1)
            current_real_ep = real_ep_batch.repeat(batch_size, 1, 1, 1)
            
            pred_cost = model(batch_params, current_real_bp, current_real_ep)
            
            # Normalize for loss calculation
            pred_cost_norm = (pred_cost - model.output_mean) / model.output_std
            batch_costs_norm = (batch_costs - model.output_mean) / model.output_std
            
            # Structure outputs for loss
            bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen((batch_params - model.input_mean)/model.input_std)
            
            total_loss_val, cost_loss, count_loss, coord_reg = hierarchical_loss_function(
                pred_cost_norm, batch_costs_norm, bp_syn, bp_probs, ep_syn, ep_probs, real_bp, real_ep
            )
            
            optimizer.zero_grad()
            total_loss_val.backward()
            optimizer.step()
            
            # Clamp temperature to avoid numerical instability
            # With log_temperature, we can clamp the log value to e.g. -5 (exp(-5) ~ 0.006)
            model.sinkhorn_net.log_temperature.data.clamp_(min=-5.0)
            
            epoch_loss += total_loss_val.item()
            epoch_batches += 1
            
            total_samples_processed += batch_size
            samples_since_validation += batch_size
            
            # Check if it's time to validate
            if samples_since_validation >= validation_interval:
                current_epoch_loss = epoch_loss / epoch_batches
                
                # Run full validation
                val_metrics = validate_model(model, val_loader, real_bp_batch, real_ep_batch)
                
                print(f"Sample {total_samples_processed} | Train Loss: {current_epoch_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | Val MAE: {val_metrics['mae']:.2f} | "
                      f"RelErr: {val_metrics['rel_err_mean']:.4f}")
                
                # Log detailed stats
                log_training_stats(csv_file, total_samples_processed, epoch+1, current_epoch_loss, val_metrics)
                
                # Reset counter
                samples_since_validation = 0
                
                # Save model
                torch.save(model.state_dict(), model_name)
        
    print("\nTraining complete.")
    
    # Test Evaluation
    print("Evaluating on Test Set...")
    model.eval()
    test_errors = []
    
    with torch.no_grad():
        for test_params, test_costs in test_loader:
            bs = test_params.size(0)
            current_real_bp = real_bp_batch.repeat(bs, 1, 1, 1)
            current_real_ep = real_ep_batch.repeat(bs, 1, 1, 1)
            
            pred = model(test_params, current_real_bp, current_real_ep)
            
            rel_err = torch.abs(pred - test_costs) / (torch.abs(test_costs) + 1e-8)
            test_errors.extend(rel_err.numpy().flatten())
            
    test_acc = 100.0 * np.mean(np.array(test_errors) < accuracy_threshold)
    mean_rel_err = np.mean(test_errors)
    median_rel_err = np.median(test_errors)
    
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Mean Relative Error: {mean_rel_err:.4f}")
    print(f"Median Relative Error: {median_rel_err:.4f}")
