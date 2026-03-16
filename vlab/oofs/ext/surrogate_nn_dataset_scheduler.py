"""
Hierarchical Surrogate Neural Network (Baseline + Scheduler).
Decomposes the problem into:
1. Structure Generation: Generates synthetic plant structure from L-system params.
2. Structure Comparison: Compares synthetic structure with real daily snapshots.
3. Cost Aggregation: Aggregates daily comparison costs into a final fitness value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# --- Configuration ---
MODEL_NAME = "surrogate_model_scheduler.pt"

class PlantDataset(Dataset):
    """
    Standard PyTorch Dataset for plant parameters and costs.
    Expects CSV with columns: [ID, Cost, Param1, Param2, ..., Param13]
    """
    def __init__(self, csv_file, root_dir=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # Assuming params start at column 2 (index 2)
        self.params = self.data.iloc[:, 2:].values.astype(np.float32)
        self.costs = self.data.iloc[:, 1].values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        params = torch.tensor(self.params[idx])
        cost = torch.tensor([self.costs[idx]])
        return params, cost

class StructureGenerationNet(nn.Module):
    """
    Generates point clouds (Branch Points & End Points) from parameters.
    Also predicts 'existence probability' for each point to handle variable numbers of points.
    """
    def __init__(self, input_dim=13, max_points=50):
        super().__init__()
        self.max_points = max_points
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Branch Point Head
        self.bp_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_points * 3) # x, y, prob
        )
        
        # End Point Head
        self.ep_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_points * 3) # x, y, prob
        )
        
    def forward(self, x):
        features = self.feature_net(x)
        
        # Process Branch Points
        bp_raw = self.bp_net(features).reshape(-1, self.max_points, 3)
        bp_coords = bp_raw[:, :, :2] * 200.0 # Scale to ~image size
        bp_probs = torch.sigmoid(bp_raw[:, :, 2])
        
        # Process End Points
        ep_raw = self.ep_net(features).reshape(-1, self.max_points, 3)
        ep_coords = ep_raw[:, :, :2] * 200.0
        ep_probs = torch.sigmoid(ep_raw[:, :, 2])
        
        return bp_coords, bp_probs, ep_coords, ep_probs

class StructureComparisonNet(nn.Module):
    """
    Compares a generated structure (synthetic) with a real structure (target).
    Originally named 'HungarianAssignmentNet' but functions as a learnable distance metric.
    """
    def __init__(self, max_points=50):
        super().__init__()
        self.max_points = max_points
        
        # Inputs: (BP_Syn, EP_Syn, BP_Real, EP_Real) flattened
        # Each is max_points * 2 coords.
        input_size = (max_points * 2) * 4 
        
        self.structure_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Predicts a 'cost' or 'distance' directly
        self.cost_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, bp_syn, ep_syn, bp_real, ep_real):
        batch_size = bp_syn.size(0)
        scale = 200.0
        
        # Normalize coordinates
        bp_syn_norm = bp_syn / scale
        ep_syn_norm = ep_syn / scale
        bp_real_norm = bp_real / scale
        ep_real_norm = ep_real / scale
        
        # Flatten and Concatenate
        structure_features = torch.cat([
            bp_syn_norm.reshape(batch_size, -1),
            ep_syn_norm.reshape(batch_size, -1),
            bp_real_norm.reshape(batch_size, -1),
            ep_real_norm.reshape(batch_size, -1)
        ], dim=1)
        
        encoded = self.structure_encoder(structure_features)
        
        # Output is a scalar representing the 'cost' of the mismatch for this pair
        # We scale it back up to match the magnitude of real costs
        raw_cost = self.cost_net(encoded)
        return raw_cost * 10000.0

class CostAggregationNet(nn.Module):
    """Aggregates daily costs into a final total cost."""
    def __init__(self, max_days=26):
        super().__init__()
        self.max_days = max_days
        self.input_norm_scale = 10000.0
        
        self.temporal_net = nn.Sequential(
            nn.Linear(max_days, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, daily_costs):
        norm_input = daily_costs / self.input_norm_scale
        return self.temporal_net(norm_input)

class HierarchicalPlantSurrogateNet(nn.Module):
    """
    Main Surrogate Model class.
    """
    def __init__(self, input_dim=13, max_points=50, max_days=26, 
                 input_mean=None, input_std=None, output_mean=None, output_std=None):
        super().__init__()
        self.structure_gen = StructureGenerationNet(input_dim, max_points)
        self.comparison_net = StructureComparisonNet(max_points)
        self.cost_aggregator = CostAggregationNet(max_days)
        
        # Register normalization stats as buffers (not learnable parameters)
        self.register_buffer('input_mean', torch.tensor(input_mean if input_mean is not None else np.zeros(input_dim), dtype=torch.float32))
        self.register_buffer('input_std', torch.tensor(input_std if input_std is not None else np.ones(input_dim), dtype=torch.float32))
        self.register_buffer('output_mean', torch.tensor(output_mean if output_mean is not None else 0.0, dtype=torch.float32))
        self.register_buffer('output_std', torch.tensor(output_std if output_std is not None else 1.0, dtype=torch.float32))
        
    def forward(self, x, real_bp_batch=None, real_ep_batch=None):
        # 1. Generate Structure from Normalized Params
        x_norm = (x - self.input_mean) / self.input_std
        bp_syn, bp_probs, ep_syn, ep_probs = self.structure_gen(x_norm)
        
        # If no real data provided (e.g., just visualizing structure), return structure
        if real_bp_batch is None or real_ep_batch is None:
            return bp_syn, bp_probs, ep_syn, ep_probs
        
        # 2. Compare Synthetic Structure with Real Structure for each day
        # Note: This loop can be slow. Could be vectorized if StructureComparisonNet supported time dimension.
        daily_costs = []
        num_days = real_bp_batch.size(1)
        
        for day in range(num_days):
            bp_real_day = real_bp_batch[:, day, :, :]
            ep_real_day = real_ep_batch[:, day, :, :]
            
            day_cost = self.comparison_net(bp_syn, ep_syn, bp_real_day, ep_real_day)
            daily_costs.append(day_cost)
        
        # Stack into (Batch, Days)
        daily_costs_tensor = torch.stack(daily_costs, dim=1).squeeze(-1)
        
        # Pad or Truncate to fixed number of days for Aggregator
        current_days = daily_costs_tensor.size(1)
        target_days = 26
        if current_days < target_days:
            padding = torch.zeros(daily_costs_tensor.size(0), target_days - current_days, device=x.device)
            daily_costs_tensor = torch.cat([daily_costs_tensor, padding], dim=1)
        else:
            daily_costs_tensor = daily_costs_tensor[:, :target_days]
        
        # 3. Aggregate Costs
        final_cost_norm = self.cost_aggregator(daily_costs_tensor)
        
        # 4. De-normalize to Real Scale
        denorm_cost = final_cost_norm * self.output_std + self.output_mean
        
        # Optional: Bias correction for low cost values (legacy logic preserved)
        bias_correction = torch.where(denorm_cost < 60000, 
                                    -1000 * torch.sigmoid((60000 - denorm_cost) / 5000), 
                                    torch.zeros_like(denorm_cost))
                                    
        return F.softplus(denorm_cost + bias_correction)

def hierarchical_loss_function(pred_cost, true_cost, bp_syn, bp_probs, ep_syn, ep_probs, real_bp, real_ep):
    """
    Computes composite loss:
    1. MSE on Final Cost (Regression)
    2. Auxiliary loss on predicted Point Counts (approximate structure correctness)
    3. Regularization on Coordinate Variance
    """
    # Main Task Loss
    cost_loss = F.mse_loss(pred_cost, true_cost)
    
    # Auxiliary: Point Count Matching
    # (Since we have probs, sum(probs) = expected count)
    bp_count_target = torch.tensor([min(len(day_bp), 50) for day_bp in real_bp], device=pred_cost.device).float().mean()
    ep_count_target = torch.tensor([min(len(day_ep), 50) for day_ep in real_ep], device=pred_cost.device).float().mean()
    
    bp_count_pred = bp_probs.mean(dim=0).sum() # Average across batch, then sum probs
    ep_count_pred = ep_probs.mean(dim=0).sum()
    # Actually, simpler: sum probs per sample, then mean across batch
    bp_count_pred = bp_probs.sum(dim=1).mean()
    ep_count_pred = ep_probs.sum(dim=1).mean()

    count_loss = F.mse_loss(bp_count_pred, bp_count_target) + F.mse_loss(ep_count_pred, ep_count_target)
    
    # Regularization: keep coordinates from exploding
    scale = 200.0
    coord_regularization = 1e-3 * (torch.var(bp_syn/scale) + torch.var(ep_syn/scale))
    
    total_loss = cost_loss + 0.005 * count_loss + coord_regularization
    return total_loss, cost_loss, count_loss, coord_regularization

def get_scheduler(optimizer):
    """Returns the scheduler to use for this model."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

