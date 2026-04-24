import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np

# --- Configuration ---
INPUT_DIM = 13
MAX_POINTS = 50
MAX_DAYS = 26

class PlantDataset(Dataset):
    """
    Standard PyTorch Dataset for plant parameters and costs.
    
    Expects a CSV file with the following columns:
    - ID: The unique identifier for the sample (Column 0).
    - Cost: The simulated cost value (Column 1).
    - Param1, Param2, ..., Param13: The L-system parameters (Columns 2+).
    
    Args:
        csv_file (str): Path to the dataset CSV file.
        root_dir (str, optional): Directory containing dataset dependencies.
    """
    def __init__(self, csv_file, root_dir=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # Params start at column 2 (Index 2)
        self.ids = self.data.iloc[:, 0].values.astype(np.int64)
        self.params = self.data.iloc[:, 2:].values.astype(np.float32)
        self.costs = self.data.iloc[:, 1].values.astype(np.float32)
        csv_dir = os.path.dirname(os.path.abspath(csv_file))
        split_name = os.path.splitext(os.path.basename(csv_file))[0]
        self.structures_dir = os.path.join(csv_dir, split_name, "structures")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        params = torch.tensor(self.params[idx])
        cost = torch.tensor([self.costs[idx]])
        sample_id = torch.tensor(self.ids[idx], dtype=torch.long)
        return params, cost, sample_id

class StructureGenerationNet(nn.Module):
    """
    Step 1: Structure Generation Module.
    Takes the 13 L-system parameters and generates a 'Synthetic Point Cloud'.
    
    It predicts two separated sets of topological points:
        - Branch Points (bp): Coordinates where branches bifurcate.
        - End Points (ep): Coordinates at the literal tip of the active branch.
        
    It also predicts an 'existence probability' for each theoretical point to handle
    dynamically growing structures where the number of instantiated points varies.
    
    Args:
        input_dim (int): Number of biological parameters input into the network. Defaults to INPUT_DIM.
        max_points (int): Maximum possible theoretical points the plant can have. Defaults to MAX_POINTS.
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
            nn.Linear(128, max_points * 3) # x, y, prob
        )
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

def log_sinkhorn_iterations(log_alpha, n_iters=5):
    """
    Project pairwise scores to a doubly stochastic matrix in log-space.
    
    This function iteratively applies the Sinkhorn-Knopp algorithm to
    normalize the row and column sums of log-probabilities for differentiable
    soft mappings.
    
    Args:
        log_alpha (torch.Tensor): Unnormalized log-score matrix.
        n_iters (int): The number of projective Sinkhorn iterations.
        
    Returns:
        torch.Tensor: A stochastic distance mapping.
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()

class PointSetEncoder(nn.Module):
    """
    Permutation-equivariant point encoder.
    
    This module uses self-attention inside the point cloud structure prior to assignment 
    and aggregation. This ensures the embedded feature points represent not only distinct
    individual locations, but also their relationship to the holistic plant structure.
    
    Args:
        input_dim (int): Number of feature dimensions (normally x, y, and probability metadata).
        hidden_dim (int): Size of the hidden representation.
    """
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        emb = self.embedding(x)
        attn_out, _ = self.self_attention(emb, emb, emb)
        return self.norm(emb + attn_out)

class DynamicCoordinateScaler(nn.Module):
    """
    Predicts per-sample bounding-box coordinate scales based on real-plant geometry points.
    
    This layer helps correct differences in pixel scaling versus intrinsic measurement sizes
    by calculating the centered standard deviation and absolute mean across all points
    in an active plant.
    
    Args:
        hidden_dim (int): Number of nodes inside the scaling network.
        init_scale (float): Base initial scale multiplier to start the bounding logic.
    """
    def __init__(self, hidden_dim=32, init_scale=200.0):
        super().__init__()
        self.base_log_scale = nn.Parameter(torch.log(torch.tensor([init_scale, init_scale], dtype=torch.float32)))
        self.scale_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, bp_real, ep_real, bp_mask=None, ep_mask=None):
        if bp_mask is None:
            bp_mask = (bp_real.abs().sum(dim=-1) > 0)
        if ep_mask is None:
            ep_mask = (ep_real.abs().sum(dim=-1) > 0)

        bp_mask_f = bp_mask.to(dtype=bp_real.dtype).unsqueeze(-1)
        ep_mask_f = ep_mask.to(dtype=ep_real.dtype).unsqueeze(-1)

        bp_count_points = bp_mask.to(dtype=bp_real.dtype).sum(dim=1).clamp_min(1.0)
        ep_count_points = ep_mask.to(dtype=ep_real.dtype).sum(dim=1).clamp_min(1.0)
        bp_count_coords = (bp_count_points * bp_real.size(-1)).clamp_min(1.0)
        ep_count_coords = (ep_count_points * ep_real.size(-1)).clamp_min(1.0)

        bp_abs_mean = (bp_real.abs() * bp_mask_f).sum(dim=(1, 2)) / bp_count_coords
        bp_mask_sum = bp_mask_f.sum(dim=1).clamp_min(1.0).view(-1, 1, 1)
        bp_center = (bp_real * bp_mask_f).sum(dim=1, keepdim=True) / bp_mask_sum
        bp_centered = (bp_real - bp_center) * bp_mask_f
        bp_std = torch.sqrt(bp_centered.pow(2).sum(dim=(1, 2)) / bp_count_coords + 1e-6)

        ep_abs_mean = (ep_real.abs() * ep_mask_f).sum(dim=(1, 2)) / ep_count_coords
        ep_mask_sum = ep_mask_f.sum(dim=1).clamp_min(1.0).view(-1, 1, 1)
        ep_center = (ep_real * ep_mask_f).sum(dim=1, keepdim=True) / ep_mask_sum
        ep_centered = (ep_real - ep_center) * ep_mask_f
        ep_std = torch.sqrt(ep_centered.pow(2).sum(dim=(1, 2)) / ep_count_coords + 1e-6)

        stats = torch.stack([bp_abs_mean, bp_std, ep_abs_mean, ep_std], dim=-1)

        delta = self.scale_net(stats)
        base_scale = torch.exp(self.base_log_scale).unsqueeze(0)
        scales = base_scale * torch.exp(0.1 * torch.tanh(delta))
        scales = scales.clamp_min(1e-3)
        bp_scale = scales[:, 0].reshape(-1, 1, 1)
        ep_scale = scales[:, 1].reshape(-1, 1, 1)
        return bp_scale, ep_scale

class CostAggregationNet(nn.Module):
    """
    Step 3: Aggregation Network.
    
    Aggregates daily costs arrays representing longitudinal growth snapshots into a final total 
    cost via a lightweight temporal network block. This allows the model to differentiate
    early growth vs late growth importance dynamically instead of uniformly summing the errors.
    
    Args:
        max_days (int): Configured maximum number of days to analyze longitudinally.
        use_aggregator (bool): Ablation flag. If False, bypasses the network and uses a mean.
    """
    def __init__(self, max_days=MAX_DAYS, use_aggregator=True):
        super().__init__()
        self.max_days = max_days
        self.use_aggregator = use_aggregator
        self.input_norm_scale = 1.0
        if self.use_aggregator:
            self.temporal_net = nn.Sequential(
                nn.Linear(max_days, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
    def forward(self, daily_costs):
        norm_input = daily_costs / self.input_norm_scale
        if self.use_aggregator:
            return self.temporal_net(norm_input)
        else:
            return norm_input.mean(dim=-1, keepdim=True)

class BaseHierarchicalPlantSurrogateNet(nn.Module):
    """Base class for Hierarchical Surrogate Models."""
    def __init__(self, input_dim=INPUT_DIM, max_points=MAX_POINTS, max_days=MAX_DAYS, 
                 input_mean=None, input_std=None, output_mean=None, output_std=None,
                 use_aggregator=True):
        super().__init__()
        self.structure_gen = StructureGenerationNet(input_dim, max_points)
        self.cost_aggregator = CostAggregationNet(max_days, use_aggregator=use_aggregator)
        
        self.register_buffer('input_mean', torch.tensor(input_mean if input_mean is not None else np.zeros(input_dim), dtype=torch.float32))
        self.register_buffer('input_std', torch.tensor(input_std if input_std is not None else np.ones(input_dim), dtype=torch.float32))
        self.register_buffer('output_mean', torch.tensor(output_mean if output_mean is not None else 0.0, dtype=torch.float32))
        self.register_buffer('output_std', torch.tensor(output_std if output_std is not None else 1.0, dtype=torch.float32))
        
        # Subclasses should define self.assignment_net (or comparison_net)
        
    def forward(self, x, real_bp_batch=None, real_ep_batch=None):
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        bp_syn, bp_probs, ep_syn, ep_probs = self.structure_gen(x_norm)
        
        if real_bp_batch is None or real_ep_batch is None:
            return bp_syn, bp_probs, ep_syn, ep_probs
        
        batch_size = x.size(0)
        num_days = real_bp_batch.size(1)
        max_points = bp_syn.size(1)
        
        bp_syn_expanded = bp_syn.unsqueeze(1).expand(-1, num_days, -1, -1).reshape(-1, max_points, 2)
        ep_syn_expanded = ep_syn.unsqueeze(1).expand(-1, num_days, -1, -1).reshape(-1, max_points, 2)
        bp_probs_expanded = bp_probs.unsqueeze(1).expand(-1, num_days, -1).reshape(-1, max_points)
        ep_probs_expanded = ep_probs.unsqueeze(1).expand(-1, num_days, -1).reshape(-1, max_points)
        
        real_bp_flat = real_bp_batch.reshape(-1, max_points, 2)
        real_ep_flat = real_ep_batch.reshape(-1, max_points, 2)
        
        assignment_net = getattr(self, 'assignment_net', getattr(self, 'comparison_net', getattr(self, 'hungarian_net', None)))
        _, total_costs_flat = assignment_net(
            bp_syn_expanded, ep_syn_expanded, 
            real_bp_flat, real_ep_flat, 
            bp_probs_syn=bp_probs_expanded, ep_probs_syn=ep_probs_expanded
        )
        
        daily_costs_tensor = total_costs_flat.view(batch_size, num_days)
        
        current_days = daily_costs_tensor.size(1)
        target_days = 26
        if current_days < target_days:
            padding = torch.zeros(batch_size, target_days - current_days, device=x.device)
            daily_costs_tensor = torch.cat([daily_costs_tensor, padding], dim=1)
        else:
            daily_costs_tensor = daily_costs_tensor[:, :target_days]
        
        final_cost_norm = self.cost_aggregator(daily_costs_tensor)
        denorm_cost = final_cost_norm * self.output_std + self.output_mean
        
        return F.softplus(denorm_cost)

def hierarchical_loss_function(
    pred_cost, true_cost, bp_syn, bp_probs, ep_syn, ep_probs, inputs,
    real_bp, real_ep, guidance_bp=None, guidance_ep=None,
    guidance_bp_mask=None, guidance_ep_mask=None,
):
    """Regression loss plus optional L-system structure guidance."""
    cost_loss = F.mse_loss(pred_cost, true_cost)
    guidance_loss = pred_cost.new_tensor(0.0)

    if guidance_bp is not None and guidance_ep is not None and guidance_bp_mask is not None and guidance_ep_mask is not None:
        bp_mask = guidance_bp_mask.to(dtype=torch.bool)
        ep_mask = guidance_ep_mask.to(dtype=torch.bool)

        if bp_mask.sum() > 0:
            bp_pairwise_dist = torch.cdist(bp_syn, guidance_bp, p=2)
            bp_pairwise_dist = bp_pairwise_dist.masked_fill(~bp_mask.unsqueeze(1), 1e6)
            bp_pred_to_target = bp_pairwise_dist.min(dim=-1).values
            bp_target_to_pred = bp_pairwise_dist.min(dim=-2).values
            bp_pred_weights = bp_probs / (bp_probs.sum(dim=1, keepdim=True) + 1e-6)
            bp_target_weights = bp_mask.float() / (bp_mask.float().sum(dim=1, keepdim=True) + 1e-6)
            bp_coord_loss = ((bp_pred_weights * bp_pred_to_target).sum(dim=1) + (bp_target_weights * bp_target_to_pred).sum(dim=1)).mean()
        else:
            bp_coord_loss = pred_cost.new_tensor(0.0)

        if ep_mask.sum() > 0:
            ep_pairwise_dist = torch.cdist(ep_syn, guidance_ep, p=2)
            ep_pairwise_dist = ep_pairwise_dist.masked_fill(~ep_mask.unsqueeze(1), 1e6)
            ep_pred_to_target = ep_pairwise_dist.min(dim=-1).values
            ep_target_to_pred = ep_pairwise_dist.min(dim=-2).values
            ep_pred_weights = ep_probs / (ep_probs.sum(dim=1, keepdim=True) + 1e-6)
            ep_target_weights = ep_mask.float() / (ep_mask.float().sum(dim=1, keepdim=True) + 1e-6)
            ep_coord_loss = ((ep_pred_weights * ep_pred_to_target).sum(dim=1) + (ep_target_weights * ep_target_to_pred).sum(dim=1)).mean()
        else:
            ep_coord_loss = pred_cost.new_tensor(0.0)

        bp_count_target = bp_mask.sum(dim=1).float()
        ep_count_target = ep_mask.sum(dim=1).float()
        bp_count_pred = bp_probs.sum(dim=1)
        ep_count_pred = ep_probs.sum(dim=1)
        count_loss = F.mse_loss(bp_count_pred, bp_count_target) + F.mse_loss(ep_count_pred, ep_count_target)

        guidance_loss = bp_coord_loss + ep_coord_loss + 0.1 * count_loss

    total_loss = cost_loss + 0.01 * guidance_loss
    return total_loss, cost_loss, guidance_loss, pred_cost.new_tensor(0.0)

def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
