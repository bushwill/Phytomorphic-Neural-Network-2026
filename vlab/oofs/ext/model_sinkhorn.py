"""
Continuous Sinkhorn-Knopp Surrogate Architecture.

Objective:
    Applies Optimal Transport with entropy regularization to establish a fully continuous
    and natively differentiable bipartite assignment matrix between synthesized hierarchical 
    L-System geometries and target biological point configurations.

Pipeline Decomposition:
    1. Structure Generation Module: Learns the spatial instantiation of topology from latent parameters.
    2. Entropic Regularization (Sinkhorn): Projects hard combinatorial assignments into continuous probability spaces.
    3. Temporal Aggregation: Integrates daily developmental frames into an end-to-end longitudinal cost metric.
"""

import torch
import torch.nn as nn

from model_base import (
    INPUT_DIM, MAX_POINTS, MAX_DAYS,
    PointSetEncoder, DynamicCoordinateScaler,
    BaseHierarchicalPlantSurrogateNet, log_sinkhorn_iterations, get_scheduler
)

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
    def __init__(self, max_points=MAX_POINTS, n_iters=7, feature_dim=64, use_encoder=True, use_scaler=True):
        super().__init__()
        self.max_points = max_points
        self.n_iters = n_iters
        self.use_encoder = use_encoder
        self.use_scaler = use_scaler
        
        if self.use_encoder:
            self.bp_encoder = PointSetEncoder(input_dim=3, hidden_dim=feature_dim)
            self.ep_encoder = PointSetEncoder(input_dim=3, hidden_dim=feature_dim)
        else:
            self.bp_encoder = nn.Linear(3, feature_dim)
            self.ep_encoder = nn.Linear(3, feature_dim)
            
        if self.use_scaler:
            self.coord_scaler = DynamicCoordinateScaler()
            
        # Learnable temperature controls how "sharp" the matching is
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, bp_syn, ep_syn, bp_real, ep_real, bp_probs_syn=None, ep_probs_syn=None):
        """
        Executes the differentiable set matching pipeline using Sinkhorn-Knopp normalizations.
        
        This algorithm computes the Earth Mover's Distance between the generated topology 
        and the biological target. It embeds both sets, calculates their euclidean and feature 
        distances, then normalizes that distance matrix into a soft probability assignment
        where gradients can flow freely back to the initial parameter generator.
        
        Args:
            bp_syn (torch.Tensor): Synthesized branch points [Batch, Points, 3].
            ep_syn (torch.Tensor): Synthesized end points [Batch, Points, 3].
            bp_real (torch.Tensor): Active ground target branch points [Batch, Points, 3].
            ep_real (torch.Tensor): Active ground target end points [Batch, Points, 3].
            bp_probs_syn (torch.Tensor, optional): Probabilities for synthesized branches.
            ep_probs_syn (torch.Tensor, optional): Probabilities for synthesized endings.
            
        Returns:
            tuple: A (assignments, total_cost) tuple where assignments hold the match mappings
            and total_cost sums expected distances under the continuous Sinkhorn assignment constraint.
        """
        if bp_probs_syn is None:
            bp_probs_syn = torch.ones(bp_syn.size(0), bp_syn.size(1), device=bp_syn.device, dtype=bp_syn.dtype)
        if ep_probs_syn is None:
            ep_probs_syn = torch.ones(ep_syn.size(0), ep_syn.size(1), device=ep_syn.device, dtype=ep_syn.dtype)

        # Real-point presence mask: padded zeros get lower influence.
        bp_valid_cols = (bp_real.abs().sum(dim=-1) > 0)
        ep_valid_cols = (ep_real.abs().sum(dim=-1) > 0)
        bp_probs_real = bp_valid_cols.to(bp_real.dtype)
        ep_probs_real = ep_valid_cols.to(ep_real.dtype)

        if self.use_scaler:
            bp_scale, ep_scale = self.coord_scaler(bp_real, ep_real, bp_probs_real > 0, ep_probs_real > 0)
            bp_syn_scaled = bp_syn / bp_scale
            bp_real_scaled = bp_real / bp_scale
            ep_syn_scaled = ep_syn / ep_scale
            ep_real_scaled = ep_real / ep_scale
        else:
            bp_syn_scaled, bp_real_scaled = bp_syn, bp_real
            ep_syn_scaled, ep_real_scaled = ep_syn, ep_real

        # Encode BP and EP sets separately (permutation-equivariant), including probabilities.
        bp_syn_features = torch.cat([bp_syn_scaled, bp_probs_syn.unsqueeze(-1)], dim=-1)
        bp_real_features = torch.cat([bp_real_scaled, bp_probs_real.unsqueeze(-1)], dim=-1)
        ep_syn_features = torch.cat([ep_syn_scaled, ep_probs_syn.unsqueeze(-1)], dim=-1)
        ep_real_features = torch.cat([ep_real_scaled, ep_probs_real.unsqueeze(-1)], dim=-1)

        bp_syn_emb = self.bp_encoder(bp_syn_features)
        bp_real_emb = self.bp_encoder(bp_real_features)
        ep_syn_emb = self.ep_encoder(ep_syn_features)
        ep_real_emb = self.ep_encoder(ep_real_features)

        bp_dist = torch.cdist(bp_syn_scaled, bp_real_scaled, p=2)
        ep_dist = torch.cdist(ep_syn_scaled, ep_real_scaled, p=2)

        temperature = torch.exp(self.log_temperature).clamp_min(1e-4)
        bp_scores = torch.bmm(bp_syn_emb, bp_real_emb.transpose(1, 2)) / temperature
        ep_scores = torch.bmm(ep_syn_emb, ep_real_emb.transpose(1, 2)) / temperature

        # Ensure at least one valid real-point column per sample for stable Sinkhorn normalization.
        bp_no_valid = ~bp_valid_cols.any(dim=1)
        ep_no_valid = ~ep_valid_cols.any(dim=1)
        if bp_no_valid.any():
            bp_valid_cols = bp_valid_cols.clone()
            bp_valid_cols[bp_no_valid, 0] = True
        if ep_no_valid.any():
            ep_valid_cols = ep_valid_cols.clone()
            ep_valid_cols[ep_no_valid, 0] = True

        bp_scores = bp_scores.masked_fill(~bp_valid_cols.unsqueeze(1), -1e9)
        ep_scores = ep_scores.masked_fill(~ep_valid_cols.unsqueeze(1), -1e9)

        bp_assignment = log_sinkhorn_iterations(bp_scores, n_iters=self.n_iters)
        ep_assignment = log_sinkhorn_iterations(ep_scores, n_iters=self.n_iters)

        bp_weight = bp_probs_syn.unsqueeze(-1) * bp_probs_real.unsqueeze(1)
        ep_weight = ep_probs_syn.unsqueeze(-1) * ep_probs_real.unsqueeze(1)

        bp_cost = torch.sum(bp_assignment * bp_dist * bp_weight, dim=(-1, -2))
        ep_cost = torch.sum(ep_assignment * ep_dist * ep_weight, dim=(-1, -2))
        total_cost = (bp_cost + ep_cost).unsqueeze(-1)

        return (bp_assignment, ep_assignment), total_cost

class HierarchicalPlantSurrogateNet(BaseHierarchicalPlantSurrogateNet):
    """
    Main Continuous Sinkhorn Surrogate Model Wrapper.
    
    Orchestrates the entire combinatorial pipeline:
      1. Generates the structure (Network mapped from L-System Params -> Coordinates).
      2. Learns the distance using entropic regularized Sinkhorn assignments 
         (projects Generated -> Target matching directly into continuous probability space).
      3. Aggregates all structural and temporal stages using continuous evaluation.
      
    Args:
        input_dim (int): Number of L-System driving parameters.
        max_points (int): Maximum possible morphological points expected.
        max_days (int): Maximum number of temporal developmental stages observed.
        input_mean (torch.Tensor, optional): Precomputed Z-score mean for parameters.
        input_std (torch.Tensor, optional): Precomputed Z-score variance for parameters.
        output_mean (torch.Tensor, optional): Precomputed Z-score normalization for final cost.
        output_std (torch.Tensor, optional): Precomputed Z-score normalization for final cost.
        use_encoder (bool): Whether to activate the structured PointSetEncoder.
        use_scaler (bool): Whether to activate the DynamicCoordinateScaler normalization.
        use_aggregator (bool): Whether to learn temporal loss aggregation.
    """
    def __init__(self, input_dim=INPUT_DIM, max_points=MAX_POINTS, max_days=MAX_DAYS, 
                 input_mean=None, input_std=None, output_mean=None, output_std=None,
                 use_encoder=True, use_scaler=True, use_aggregator=True):
        super().__init__(input_dim, max_points, max_days, input_mean, input_std, output_mean, output_std, use_aggregator=use_aggregator)
        self.assignment_net = SinkhornAssignmentNet(
            max_points=max_points,
            use_encoder=use_encoder, 
            use_scaler=use_scaler
        )

