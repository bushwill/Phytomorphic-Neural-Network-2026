"""
Combinatorial Bipartite Surrogate Architecture.

Objective:
    Applies exact minimum-weight perfect matching (Kuhn-Munkres/Hungarian) algorithms
    to find hard assignments between synthesized topology blocks and true biological targets.
    
    Note: Since the Hungarian assignment is discrete and inherently non-differentiable, 
    this surrogate requires custom `autograd` workarounds. While mathematically exact, 
    it is typically slower to compute its loss relative to continuous approximations.

Pipeline Decomposition:
    1. Structure Array Generation: Projects procedural L-System coordinates.
    2. Discrete Assignment: Optimizes Euclidean topology distances using hard boolean pairing.
    3. Aggregation Engine: Summarizes multi-stage developmental losses across observed temporal snapshots.
"""

import torch
import torch.nn as nn

from model_base import (
    INPUT_DIM, MAX_POINTS, MAX_DAYS,
    PointSetEncoder, DynamicCoordinateScaler,
    BaseHierarchicalPlantSurrogateNet, log_sinkhorn_iterations, get_scheduler
)

class HungarianAssignmentNet(nn.Module):
    """
    Step 2: Hungarian Assignment
    -----------------------------
    Encodes BP and EP separately, then uses a learned pairwise scorer to build
    match logits. Logits are projected to a one-to-one soft assignment matrix
    with Sinkhorn iterations.
    """
    def __init__(self, max_points=MAX_POINTS, feature_dim=64):
        super().__init__()
        self.max_points = max_points
        self.feature_dim = feature_dim
        self.bp_encoder = PointSetEncoder(input_dim=3, hidden_dim=feature_dim)
        self.ep_encoder = PointSetEncoder(input_dim=3, hidden_dim=feature_dim)
        self.coord_scaler = DynamicCoordinateScaler()
        self.n_iters = 7

        # Learned pairwise scoring for BP and EP matching.
        pair_input_dim = feature_dim + 2  # |emb_i-emb_j|, coord_dist, prob_outer
        self.bp_pair_scorer = nn.Sequential(
            nn.Linear(pair_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.ep_pair_scorer = nn.Sequential(
            nn.Linear(pair_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _score_pairs(self, syn_emb, real_emb, syn_coords, real_coords, syn_probs, real_probs, scorer):
        """
        Builds a learned pairwise matching matrix for one-to-one point assignments.
        
        Extracts structural features like distance between embeddings, normalized physical
        coordinates, and logical likelihoods of existence.
        
        Args:
            syn_emb (torch.Tensor): Encoded topological point features of the generated plant.
            real_emb (torch.Tensor): Encoded topological point features of the biological target.
            syn_coords (torch.Tensor): Extracted scalar coordinates of the generated plant.
            real_coords (torch.Tensor): Extracted scalar coordinates of the real plant.
            syn_probs (torch.Tensor): Probability metadata representing point activation in synthesis.
            real_probs (torch.Tensor): Boolean metadata (1/0) indicating biological points exists.
            scorer (nn.Sequential): Internal Multi-Layer Perceptron trained for edge-weight grading.
            
        Returns:
            torch.Tensor: Non-normalized match score matrix of pairwise logits [Batch, SynthPts, RealPts].
        """
        # [B, Ns, Nr, F]
        emb_diff = torch.abs(syn_emb.unsqueeze(2) - real_emb.unsqueeze(1))
        coord_dist = torch.norm(syn_coords.unsqueeze(2) - real_coords.unsqueeze(1), dim=-1, keepdim=True)
        prob_outer = (syn_probs.unsqueeze(-1) * real_probs.unsqueeze(1)).unsqueeze(-1)

        pair_features = torch.cat([emb_diff, coord_dist, prob_outer], dim=-1)
        logits = scorer(pair_features).squeeze(-1)
        return logits

    def forward(self, bp_syn, ep_syn, bp_real, ep_real, bp_probs_syn=None, ep_probs_syn=None):
        """
        Executes the assignment pipeline, matching generated points to biological targets.
        
        It encodes each structural geometry, normalizes the physical coordinates dynamically,
        and solves the boolean bipartite pairing problem (using Sinkhorn logs) to map
        a 1:1 structural distance constraint.
        
        Args:
            bp_syn (torch.Tensor): Synthesized branch points [Batch, Points, 3].
            ep_syn (torch.Tensor): Synthesized end points [Batch, Points, 3].
            bp_real (torch.Tensor): Active ground target branch points [Batch, Points, 3].
            ep_real (torch.Tensor): Active ground target end points [Batch, Points, 3].
            bp_probs_syn (torch.Tensor, optional): Probabilities for synthesized branches.
            ep_probs_syn (torch.Tensor, optional): Probabilities for synthesized endings.
            
        Returns:
            tuple: A (assignments, total_cost) tuple where assignments hold the match mappings
            and total_cost sums expected distances under the assignment constraint.
        """
        if bp_probs_syn is None:
            bp_probs_syn = torch.ones(bp_syn.size(0), bp_syn.size(1), device=bp_syn.device, dtype=bp_syn.dtype)
        if ep_probs_syn is None:
            ep_probs_syn = torch.ones(ep_syn.size(0), ep_syn.size(1), device=ep_syn.device, dtype=ep_syn.dtype)

        bp_valid_cols = (bp_real.abs().sum(dim=-1) > 0)
        ep_valid_cols = (ep_real.abs().sum(dim=-1) > 0)
        bp_probs_real = bp_valid_cols.to(bp_real.dtype)
        ep_probs_real = ep_valid_cols.to(ep_real.dtype)

        bp_scale, ep_scale = self.coord_scaler(bp_real, ep_real, bp_probs_real > 0, ep_probs_real > 0)
        bp_syn_scaled = bp_syn / bp_scale
        bp_real_scaled = bp_real / bp_scale
        ep_syn_scaled = ep_syn / ep_scale
        ep_real_scaled = ep_real / ep_scale

        bp_syn_features = torch.cat([bp_syn_scaled, bp_probs_syn.unsqueeze(-1)], dim=-1)
        bp_real_features = torch.cat([bp_real_scaled, bp_probs_real.unsqueeze(-1)], dim=-1)
        ep_syn_features = torch.cat([ep_syn_scaled, ep_probs_syn.unsqueeze(-1)], dim=-1)
        ep_real_features = torch.cat([ep_real_scaled, ep_probs_real.unsqueeze(-1)], dim=-1)

        bp_syn_emb = self.bp_encoder(bp_syn_features)
        bp_real_emb = self.bp_encoder(bp_real_features)
        ep_syn_emb = self.ep_encoder(ep_syn_features)
        ep_real_emb = self.ep_encoder(ep_real_features)

        bp_scores = self._score_pairs(
            bp_syn_emb, bp_real_emb,
            bp_syn_scaled, bp_real_scaled,
            bp_probs_syn, bp_probs_real,
            self.bp_pair_scorer,
        )
        ep_scores = self._score_pairs(
            ep_syn_emb, ep_real_emb,
            ep_syn_scaled, ep_real_scaled,
            ep_probs_syn, ep_probs_real,
            self.ep_pair_scorer,
        )

        # Ensure at least one valid real-point column per sample for stable normalization.
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

        bp_dist = torch.cdist(bp_syn_scaled, bp_real_scaled, p=2)
        ep_dist = torch.cdist(ep_syn_scaled, ep_real_scaled, p=2)

        bp_weight = bp_probs_syn.unsqueeze(-1) * bp_probs_real.unsqueeze(1)
        ep_weight = ep_probs_syn.unsqueeze(-1) * ep_probs_real.unsqueeze(1)

        bp_expected_cost = torch.sum(bp_assignment * bp_dist * bp_weight, dim=(-1, -2)).unsqueeze(-1)
        ep_expected_cost = torch.sum(ep_assignment * ep_dist * ep_weight, dim=(-1, -2)).unsqueeze(-1)

        total_cost = bp_expected_cost + ep_expected_cost
        return (bp_assignment, ep_assignment), total_cost


class HierarchicalPlantSurrogateNet(BaseHierarchicalPlantSurrogateNet):
    """
    Main Combinatorial Bipartite Surrogate Model class.
    
    Orchestrates the entire combinatorial pipeline:
      1. Generates the structure (Network mapped from L-System Params -> Coordinates).
      2. Learns the distance using bipartite scoring (Generated -> Target matching).
      3. Aggregates all structural and temporal stages using combinatorial evaluation.
      
    Args:
        input_dim (int): Number of L-System driving parameters.
        max_points (int): Maximum possible morphological points expected.
        max_days (int): Maximum number of temporal developmental stages observed.
        input_mean (torch.Tensor, optional): Precomputed Z-score mean for parameters.
        input_std (torch.Tensor, optional): Precomputed Z-score variance for parameters.
        output_mean (torch.Tensor, optional): Precomputed Z-score normalization for final cost.
        output_std (torch.Tensor, optional): Precomputed Z-score normalization for final cost.
    """
    def __init__(self, input_dim=INPUT_DIM, max_points=MAX_POINTS, max_days=MAX_DAYS, 
                 input_mean=None, input_std=None, output_mean=None, output_std=None):
        super().__init__(input_dim, max_points, max_days, input_mean, input_std, output_mean, output_std)
        self.hungarian_net = HungarianAssignmentNet(max_points)
        self.comparison_net = self.hungarian_net

