"""
Continuous Fourier Coordinate Embeddings (Random / Harmonic Fourier Features)
Maps continuous 1D mass-to-charge (m/z) coordinates to high-dimensional metric embeddings.
"""

import math
import torch
import torch.nn as nn

class ContinuousFourierCoordinateEmbedding(nn.Module):
    def __init__(self, d_model: int = 128, min_scale: float = 1e-3, max_scale: float = 2000.0, learnable: bool = True):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sin/cos pairs"
        self.num_frequencies = d_model // 2
        
        # Log-spaced frequencies from 2*pi / max_scale to 2*pi / min_scale
        freqs = torch.exp(
            torch.linspace(
                math.log(2 * math.pi / max_scale),
                math.log(2 * math.pi / min_scale),
                self.num_frequencies
            )
        )
        if learnable:
            self.frequencies = nn.Parameter(freqs)
        else:
            self.register_buffer("frequencies", freqs)
            
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [..., 1] or [...] continuous mass coordinates in Da
        Returns:
            embedded: [..., d_model] continuous harmonic coordinate representations
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(-1) # [B, P, 1]
            
        # [B, P, 1] * [F] -> [B, P, F]
        angles = coords * self.frequencies
        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)
        
        fourier_features = torch.cat([sin_part, cos_part], dim=-1)
        return self.proj(fourier_features)
