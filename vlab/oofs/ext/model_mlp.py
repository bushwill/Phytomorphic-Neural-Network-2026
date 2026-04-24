"""
Benchmark Standard MLP Surrogate Model.

Input: Plant Growth Parameters (13 params)
Output: Predicted Cost (Scalar)

Architecture:
- Simple Feed-Forward Neural Network (MLP).
- Does NOT generate structure.
- Does NOT use specialized distance metrics.
- Serves as a baseline to prove that the structural/hierarchical approach adds value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# --- Configuration ---
MODEL_NAME = "benchmark_mlp.pt"
INPUT_DIM = 13

class PlantDataset(Dataset):
    """
    Standard PyTorch Dataset for plant parameters and costs.
    Expects CSV with columns: [ID, Cost, Param1, Param2, ..., Param13]
    """
    def __init__(self, csv_file, root_dir=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # Params start at column 2 (Index 2)
        # Using 2: to capture all remaining columns as parameters, consistent with other models
        self.params = self.data.iloc[:, 2:].values.astype(np.float32)
        self.costs = self.data.iloc[:, 1].values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.params[idx]), torch.tensor(self.costs[idx])

class BenchmarkSurrogateNet(nn.Module):
    """
    Standard 3-Layer MLP Architecture (Benchmark).
    
    Architecture:
        Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
        
    Note:
        Does not use hierarchical structure or optimal transport (Sinkhorn) loss.
        Intended as a baseline for regression performance comparison.
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, 
                 input_mean=None, input_std=None, 
                 output_mean=None, output_std=None):
        super().__init__()
        
        # Simple Regression Network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Register normalization statistics as buffers (persistent state)
        self._register_stats('input_mean', input_mean, input_dim, 0.0)
        self._register_stats('input_std', input_std, input_dim, 1.0)
        self._register_stats('output_mean', output_mean, 1, 0.0)
        self._register_stats('output_std', output_std, 1, 1.0)
    
    def _register_stats(self, name, value, size, default):
        """Helper to register buffers for normalization stats."""
        if value is None:
            if size == 1:
                tensor = torch.tensor(default, dtype=torch.float32)
            else:
                tensor = torch.full((size,), default, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        
        self.register_buffer(name, tensor)

    def forward(self, x, real_bp=None, real_ep=None):
        """
        Forward pass with normalization.
        
        Args:
            x (Tensor): Input parameters.
            real_bp, real_ep: Ignored (compatibility args for shared training loop).
        """
        # Normalize Inputs
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        
        # Inference
        out_norm = self.net(x_norm)
        
        # De-normalize Output
        out = out_norm * self.output_std + self.output_mean
        
        return out.squeeze()

def benchmark_loss_function(pred, target, *args):
    """
    Standard MSE Loss for the Benchmark.
    
    Args:
        pred (Tensor): Predicted cost (raw).
        target (Tensor): True cost (raw).
        *args: Catch-all for structure args (bp_syn, ep_syn, etc.) used by other models.
        
    Returns:
        tuple: (total_loss, mse_component, structure_component, alignment_component)
    """
    loss = F.mse_loss(pred.squeeze(), target.squeeze())
    # Return tuple matching the signature of complex losses for compatibility
    return loss, loss, 0.0, 0.0

def get_scheduler(optimizer):
    """
    Returns the learning rate scheduler for the benchmark model.
    Uses ReduceLROnPlateau to lower LR when validation loss plateaus.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
