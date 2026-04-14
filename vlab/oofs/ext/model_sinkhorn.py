"""
Hierarchical Surrogate Neural Network (Sinkhorn + Scheduler).

Input: Plant Growth Parameters (13 params)
Output: Predicted Structural Difference (Cost) vs Real Plant

Architecture Decomposed:
1. Structure Generation: Generates synthetic point clouds from parameters.
2. Sinkhorn Assignment: Differentiable comparison between Synthetic and Real point clouds.
   - Computes soft assignment matrix (Probabilities)
   - Computes physical distance matrix (Geometry)
   - Combines them for a differentiable loss.
3. Cost Aggregation: Aggregates daily costs into final metric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math

# --- Configuration ---
MODEL_NAME = "surrogate_model_sinkhorn_scheduler.pt"
INPUT_DIM = 13
MAX_POINTS = 50
MAX_DAYS = 26

class PlantDataset(Dataset):
    """
    Standard PyTorch Dataset for plant parameters and costs.
    Expects CSV with columns: [ID, Cost, Param1, Param2, ..., Param13]
    """
    def __init__(self, csv_file, root_dir=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # Params start at column 2 (Index 2)
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
    Step 1: Structure Generation
    ----------------------------
    Takes the 13 L-system parameters and generates a "Synthetic Point Cloud".
    It predicts two sets of points:
    - Branch Points (bp): Where branches split.
    - End Points (ep): Tips of leaves/branches.
    
    It also predicts an 'existence probability' for each point, allowing the
    network to output variable numbers of points (by setting prob ~ 0).
    """
    def __init__(self, input_dim=INPUT_DIM, max_points=MAX_POINTS):
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
        self.bp_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_points * 3) # Output: x, y, probability
        )
        self.ep_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_points * 3) # Output: x, y, probability
        )
        
    def forward(self, x):
        features = self.feature_net(x)
        
        # Branch Points
        bp_raw = self.bp_net(features).reshape(-1, self.max_points, 3)
        bp_coords = bp_raw[:, :, :2] * 200.0 # Scale to image coordinates
        bp_probs = torch.sigmoid(bp_raw[:, :, 2])
        
        # End Points
        ep_raw = self.ep_net(features).reshape(-1, self.max_points, 3)
        ep_coords = ep_raw[:, :, :2] * 200.0
        ep_probs = torch.sigmoid(ep_raw[:, :, 2])
        
        return bp_coords, bp_probs, ep_coords, ep_probs

# --- Sinkhorn Layers ---

def log_sinkhorn_iterations(log_alpha, n_iters=5):
    """
    Performs Sinkhorn normalization in log-space for numerical stability.
    Iteratively normalizes rows and columns to sum to 1.
    Result is a 'Doubly Stochastic Matrix' (Soft Permutation).
    """
    for _ in range(n_iters):
        # Row normalization
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # Column normalization
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()

class PointSetEncoder(nn.Module):
    """
    Encodes a set of 2D points into a feature-rich representation.
    Uses Self-Attention so the network understands global shape/structure
    before trying to match points.
    """
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Multi-head attention captures relationships between points
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: [Batch, Points, 4] (x, y, type, prob)
        emb = self.embedding(x)
        attn_out, _ = self.self_attention(emb, emb, emb)
        return self.norm(emb + attn_out)

class SinkhornAssignmentNet(nn.Module):
    """
    Step 2: Differentiable Set Matching (Sinkhorn)
    ----------------------------------------------
    Calculates the 'Earth Mover's Distance' between Synthetic and Real points.
    
    Process:
    1. Feature Extraction: Encode both point sets (Syn & Real).
    2. Score Matrix: Compute similarity (dot product) between all pairs.
    3. Sinkhorn: Normalize scores into a soft Assignment Matrix (Probability they match).
    4. Physical Cost: Compute actual Euclidean distances between all pairs.
    5. Weighted Sum: Multiply Probabilities * Distances to get Expected Cost.
    """
    def __init__(self, max_points=MAX_POINTS, feature_dim=64):
        super().__init__()
        self.max_points = max_points
        self.encoder = PointSetEncoder(input_dim=4, hidden_dim=feature_dim)
        # Learnable temperature controls how "sharp" the matching is
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, bp_syn, ep_syn, bp_real, ep_real, bp_probs_syn=None, ep_probs_syn=None):
        # inputs are [Batch, Points, 2]
        
        # Combine Branch and End points into single set for feature extraction
        # Each point gets 4 dims: (x, y, x, y) - simplified feature vector
        syn_features = torch.cat([bp_syn, ep_syn], dim=-1)   
        real_features = torch.cat([bp_real, ep_real], dim=-1) 
        
        # 1. Encode Features (Geometric Context)
        syn_emb = self.encoder(syn_features)
        real_emb = self.encoder(real_features)
        
        # 2. Compute Score Matrix (Similarity)
        temperature = torch.exp(self.log_temperature)
        scores = torch.bmm(syn_emb, real_emb.transpose(1, 2)) / temperature
        
        # 3. Sinkhorn Iterations (Normalization)
        assignment_matrix = log_sinkhorn_iterations(scores, n_iters=5)
        
        # 4. Compute Physical Cost Matrix (Euclidean Distances)
        # Broadcast to create [Batch, N_Syn, N_Real] matrix
        syn_bp_exp = bp_syn.unsqueeze(2)
        real_bp_exp = bp_real.unsqueeze(1)
        syn_ep_exp = ep_syn.unsqueeze(2)
        real_ep_exp = ep_real.unsqueeze(1)
        
        # Distance = L2(BP_Syn - BP_Real) + L2(EP_Syn - EP_Real)
        bp_dist = torch.norm(syn_bp_exp - real_bp_exp, dim=-1)
        ep_dist = torch.norm(syn_ep_exp - real_ep_exp, dim=-1)
        physical_cost_matrix = bp_dist + ep_dist 
        
        # Weight by existence probability
        if bp_probs_syn is not None and ep_probs_syn is not None:
             point_existence = (bp_probs_syn * ep_probs_syn).unsqueeze(-1)
             physical_cost_matrix = physical_cost_matrix * point_existence

        # 5. Calculate Total Expected Cost
        # Sum(Assignment_Prob * Physical_Distance)
        total_cost = torch.sum(assignment_matrix * physical_cost_matrix, dim=(-1, -2))
        
        return assignment_matrix, total_cost.unsqueeze(-1)

class CostAggregationNet(nn.Module):
    """
    Step 3: Temporal Aggregation
    ----------------------------
    Takes the costs from all 26 days and aggregates them into a single
    scalar metric representing overall "fitness".
    """
    def __init__(self, max_days=MAX_DAYS):
        super().__init__()
        self.max_days = max_days
        self.norm = nn.BatchNorm1d(max_days)
        self.temporal_net = nn.Sequential(
            nn.Linear(max_days, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, daily_costs):
        norm_costs = self.norm(daily_costs)
        return self.temporal_net(norm_costs)

class HierarchicalPlantSurrogateNet(nn.Module):
    """
    Main Model Wrapper.
    Orchestrates the flow: Params -> Structure -> Sinkhorn Match -> Aggregation -> Cost.
    """
    def __init__(self, input_dim=INPUT_DIM, max_points=MAX_POINTS, max_days=MAX_DAYS, 
                 input_mean=None, input_std=None, output_mean=None, output_std=None):
        super().__init__()
        self.structure_gen = StructureGenerationNet(input_dim, max_points)
        self.sinkhorn_net = SinkhornAssignmentNet(max_points)
        self.cost_aggregator = CostAggregationNet(max_days)
        
        # Normalization Stats (Registered buffers, not learnable)
        self.register_buffer('input_mean', torch.tensor(input_mean if input_mean is not None else np.zeros(input_dim), dtype=torch.float32))
        self.register_buffer('input_std', torch.tensor(input_std if input_std is not None else np.ones(input_dim), dtype=torch.float32))
        self.register_buffer('output_mean', torch.tensor(output_mean if output_mean is not None else 0.0, dtype=torch.float32))
        self.register_buffer('output_std', torch.tensor(output_std if output_std is not None else 1.0, dtype=torch.float32))
        
    def forward(self, x, real_bp_batch=None, real_ep_batch=None):
        # 1. Input Normalization & Structure Generation
        x_norm = (x - self.input_mean) / self.input_std
        bp_syn, bp_probs, ep_syn, ep_probs = self.structure_gen(x_norm)
        
        # If inferencing without real targets (just generation), return structure
        if real_bp_batch is None or real_ep_batch is None:
            return bp_syn, bp_probs, ep_syn, ep_probs
        
        # 2. Sinkhorn Assignment (Vectorized Over Days)
        # Reshape to treat (Batch * Days) as a large batch for parallel processing
        batch_size = x.size(0)
        num_days = real_bp_batch.size(1)
        max_points = bp_syn.size(1)
        
        # Expand Synthetic structure to match number of days (Syn structure is constant for params, but compared against changing daily targets)
        bp_syn_expanded = bp_syn.unsqueeze(1).expand(-1, num_days, -1, -1).reshape(-1, max_points, 2)
        ep_syn_expanded = ep_syn.unsqueeze(1).expand(-1, num_days, -1, -1).reshape(-1, max_points, 2)
        bp_probs_expanded = bp_probs.unsqueeze(1).expand(-1, num_days, -1).reshape(-1, max_points)
        ep_probs_expanded = ep_probs.unsqueeze(1).expand(-1, num_days, -1).reshape(-1, max_points)
        
        # Flatten Real structure
        real_bp_flat = real_bp_batch.reshape(-1, max_points, 2)
        real_ep_flat = real_ep_batch.reshape(-1, max_points, 2)
        
        # Compute Costs
        _, total_costs_flat = self.sinkhorn_net(
            bp_syn_expanded, ep_syn_expanded, 
            real_bp_flat, real_ep_flat, 
            bp_probs_syn=bp_probs_expanded, ep_probs_syn=ep_probs_expanded
        )
        
        # Reshape back to (Batch, Days)
        daily_costs_tensor = total_costs_flat.view(batch_size, num_days)
        
        # Handle day count mismatches (Pad/Truncate)
        current_days = daily_costs_tensor.size(1)
        target_days = 26
        if current_days < target_days:
             padding = torch.zeros(batch_size, target_days - current_days, device=x.device)
             daily_costs_tensor = torch.cat([daily_costs_tensor, padding], dim=1)
        else:
             daily_costs_tensor = daily_costs_tensor[:, :target_days]
        
        # 3. Aggregation & De-normalization
        final_cost = self.cost_aggregator(daily_costs_tensor)
        
        # Recover real scale
        denorm_cost = final_cost * self.output_std + self.output_mean
        return F.softplus(denorm_cost)



# Helper for compatibility
def prepare_real_plant_batch(real_bp, real_ep, max_points=50):
    """(Deprecated) Local version, main logic should use train_models.py version."""
    num_days = len(real_bp)
    bp_batch = torch.zeros(1, num_days, max_points, 2)
    ep_batch = torch.zeros(1, num_days, max_points, 2)
    for day in range(num_days):
        bp_day = real_bp[day]
        if len(bp_day) > 0:
            bp_array = torch.tensor(bp_day[:max_points], dtype=torch.float32)
            bp_batch[0, day, :min(len(bp_day), max_points), :] = bp_array
        ep_day = real_ep[day]
        if len(ep_day) > 0:
            ep_array = torch.tensor(ep_day[:max_points], dtype=torch.float32)
            ep_batch[0, day, :min(len(ep_day), max_points), :] = ep_array
    return bp_batch, ep_batch

def hierarchical_loss_function(pred_cost, true_cost, bp_syn, bp_probs, ep_syn, ep_probs, inputs, real_bp, real_ep):
    """Loss function matching baseline model structure."""
    cost_loss = F.mse_loss(pred_cost, true_cost)
    
    # Point Count Loss
    bp_count_pred = bp_probs.sum(dim=1).mean()
    ep_count_pred = ep_probs.sum(dim=1).mean()
    
    # Use Baseline's robust logic: Target the Reference Plant's average structure density
    # This provides a stable regularization target consistent with the cost comparison domain
    bp_count_target = torch.tensor([min(len(day_bp), 50) for day_bp in real_bp], device=pred_cost.device).float().mean()
    ep_count_target = torch.tensor([min(len(day_ep), 50) for day_ep in real_ep], device=pred_cost.device).float().mean()
    
    count_loss = F.mse_loss(bp_count_pred, bp_count_target) + F.mse_loss(ep_count_pred, ep_count_target)
    
    scale = 200.0
    coord_regularization = 1e-3 * (torch.var(bp_syn/scale) + torch.var(ep_syn/scale))
    
    return cost_loss + 0.005 * count_loss + coord_regularization, cost_loss, count_loss, coord_regularization

def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

