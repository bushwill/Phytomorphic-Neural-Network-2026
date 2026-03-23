"""
Benchmark Standard MLP Surrogate Model.

Provides a standard fully-connected baseline for comparison against
Phytomorphic architectures. Includes dataset handling and model definition.
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

# Dataset column indices (0-based)
COL_COST = 1
COL_PARAMS_START = 2
COL_PARAMS_END = 15  # Exclusive (slices are start:end)

class PlantDataset(Dataset):
    """
    Standard PyTorch Dataset for plant parameters and costs.
    
    Expected CSV Structure:
        [ID, Cost, Param1, ..., Param13, ...]
        - Column 1: Cost (Target)
        - Columns 2-14: Parameters (Features)
    """
    def __init__(self, csv_file, root_dir=None):
        """
        Args:
            csv_file (str): Path to the CSV file.
            root_dir (str, optional): Root directory for images (unused, kept for compatibility).
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        # Extract features (Params) and targets (Costs)
        self.params = self.data.iloc[:, COL_PARAMS_START:COL_PARAMS_END].values.astype(np.float32)
        self.costs = self.data.iloc[:, COL_COST].values.astype(np.float32)
        
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
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64, 
                 input_mean=None, input_std=None, 
                 output_mean=None, output_std=None):
        super().__init__()
        
        # Regression Network
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
            real_bp, real_ep: Ignored (compatibility args).
        """
        # Normalize -> Infer -> Denormalize
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        out_norm = self.net(x_norm)
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
    return loss, loss, 0.0, 0.0

def get_scheduler(optimizer):
    """
    Returns the learning rate scheduler for the benchmark model.
    Uses ReduceLROnPlateau to lower LR when validation loss plateaus.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
