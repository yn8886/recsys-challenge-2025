import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig


class PLEConfig(PretrainedConfig):
    model_type = "ple"

    def __init__(self, bin_edges: list[float] = []):
        self.bin_edges = bin_edges
        self.num_bins = len(self.bin_edges) - 1
        super().__init__()

    @classmethod
    def from_values(cls, values: np.ndarray, num_bins: int = 10) -> "PLEConfig":
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(values, quantiles)

        # Remove duplicates (e.g., when the number of unique values in discrete data is less than the specified number of bins)
        bin_edges = np.unique(bin_edges).astype(np.float32)
        if len(bin_edges) == 1:
            bin_edges = np.array([bin_edges[0], bin_edges[0] + 1e-6])

        return cls(bin_edges.tolist())


class PiecewiseLinearEncoding(nn.Module):
    """
    Piecewise Linear Encoding (PLE) implemented as a torch.nn.Module

    This module transforms a sequence of numerical values into higher-dimensional
    representations using piecewise linear encoding as described in the paper:
    "On Embeddings for Numerical Features in Tabular Deep Learning"

    Input: (batch_size, seq_length)
    Output: (batch_size, seq_length, num_bins)
    """

    def __init__(self, bin_edges):
        """
        Initialize the PLE module

        Parameters:
        - bin_edges: tensor or list of bin boundaries [b_0, b_1, ..., b_T]
        """
        super().__init__()

        if not torch.is_tensor(bin_edges):
            bin_edges = torch.tensor(bin_edges, dtype=torch.float32)

        self.register_buffer("bin_edges", bin_edges)
        self.num_bins = len(bin_edges) - 1

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode input sequence using PLE

        Parameters:
        - x: input tensor of shape (batch_size, seq_length)

        Returns:
        - encoding: tensor of shape (batch_size, seq_length, num_bins)
        """
        batch_size, seq_length = x.shape
        x_flat = x.reshape(-1, 1)

        lower_bounds = self.bin_edges[:-1]  # [b_0, b_1, b_2, b_3, ..., b_T-1]
        upper_bounds = self.bin_edges[1:]  # [b_1, b_2, b_3, b_4, ..., b_T]

        # Calculate relationships between values and bin boundaries
        # Broadcasting: (B*L, 1) vs (num_bins,) -> (B*L, num_bins)
        # e.g. b_2 < x < b_3
        # is_above_lower = [T, T, T, F, ..., F]
        # is_below_upper = [F, F, T, T, ..., T]
        # is_in_bin      = [F, F, T, F, ..., F]
        # is_above_upper = [T, T, F, F, ..., F]
        is_above_lower = x_flat >= lower_bounds
        is_below_upper = x_flat < upper_bounds
        is_in_bin = is_above_lower & is_below_upper
        is_above_upper = x_flat >= upper_bounds

        # Calculate bin position for values within bins
        bin_width = upper_bounds - lower_bounds
        bin_position = (x_flat - lower_bounds) / bin_width

        # Calculate PLE
        # e.g.
        # [0, 0, 0, ..., 0]
        encoding = torch.zeros_like(is_in_bin, dtype=x.dtype)
        # [0, z, 0, ..., 0]
        encoding = torch.where(is_in_bin, bin_position, encoding)
        # [1, z, 0, ..., 0]
        encoding = torch.where(is_above_upper, torch.tensor(1.0, device=x.device), encoding)

        encoding = encoding.reshape(batch_size, seq_length, self.num_bins)

        return encoding
