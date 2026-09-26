"""
Continuous-Filter Convolution Layer (cfconv) for 1D Point Clouds
Implements continuous coordinate convolution with dynamic filter networks, Triton acceleration, and FiLM.
"""

import torch
import torch.nn as nn
from ..kernels.triton_continuous_conv import triton_continuous_rbf_conv
from .film_energy_modulator import FiLMEnergyModulator

class ContinuousFilterConvolutionBlock(nn.Module):
    def __init__(self, d_model: int, cond_dim: int, sigma: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.norm1 = nn.LayerNorm(d_model)
        
        # Continuous filter generating network
        self.filter_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # FiLM energy modulator
        self.film = FiLMEnergyModulator(cond_dim=cond_dim, d_model=d_model)
        
        # Feed-forward block
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, P, D] peak features
            m: [B, P] continuous m/z coordinates
            cond: [B, cond_dim] energy & precursor conditioning
            mask: [B, P] boolean mask of valid peaks
        Returns:
            x: [B, P, D] updated features
        """
        residual = x
        normed_x = self.norm1(x)
        
        # Modulate features by collision energy
        modulated_x = self.film(normed_x, cond)
        
        # Apply continuous coordinate convolution (Triton or PyTorch fallback)
        conv_out = triton_continuous_rbf_conv(modulated_x, m, sigma=self.sigma)
        
        # Pass through filter transformation
        conv_out = self.filter_net(conv_out)
        
        if mask is not None:
            conv_out = conv_out * mask.unsqueeze(-1).float()
            
        x = residual + conv_out
        
        # Residual Feed-Forward
        x = x + self.ffn(self.norm2(x))
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            
        return x
