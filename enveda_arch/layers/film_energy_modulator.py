"""
Feature-wise Linear Modulation (FiLM) for Energy and Instrument Conditioning
Modulates continuous spectral activations conditioned on collision energy (eV) and ionization mode.
"""

import torch
import torch.nn as nn

class FiLMEnergyModulator(nn.Module):
    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model)
        )
        # Initialize gamma to 0 (so 1 + gamma starts as identity) and beta to 0
        nn.init.normal_(self.cond_net[-1].weight, std=0.01)
        nn.init.zeros_(self.cond_net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, P, d_model] features
            cond: [B, cond_dim] conditioning vector (e.g. collision energy, precursor mass, polarity)
        Returns:
            modulated_x: [B, P, d_model]
        """
        film_params = self.cond_net(cond) # [B, 2 * d_model]
        gamma, beta = torch.chunk(film_params, 2, dim=-1) # [B, d_model] each
        
        # Reshape for broadcasting over peaks P: [B, 1, d_model]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        return (1.0 + gamma) * x + beta
