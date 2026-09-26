"""
Multiscale Continuous Kernel Neural Operator (Kovachki-Stuart 2021, Mallat 2011)
Applies a parallel bank of continuous Gaussian RBF kernels across physical mass scales:
- Micro-scale (sigma = 0.02 Da): Elemental isotopic mass defects (13C, 18O, 34S)
- Meso-scale (sigma = 0.5 Da): Hydrogen and radical shifts
- Fragment-scale (sigma = 5.0 Da): Small neutral losses (H2O, CO, NH3)
- Macro-scale (sigma = 28.0 Da): Functional group and ring cleavages
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from enveda_arch.kernels.triton_continuous_conv import triton_continuous_rbf_conv, HAS_TRITON

class MultiscaleContinuousKernelOperator(nn.Module):
    """
    Multiscale Continuous Filter Integral Kernel Operator on 1D continuous measures.
    
    v_{t+1}(m_i) = W v_t(m_i) + sum_{s} sum_{j} kappa^{(s)}(m_i, m_j) v_t(m_j) I_j
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        sigmas: Tuple[float, ...] = (0.02, 0.5, 5.0, 28.0),
        use_mlp: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigmas = sigmas
        self.num_scales = len(sigmas)

        # Scale weighting projection
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
        # Linear transform for direct channel mapping
        self.w_direct = nn.Linear(in_dim, out_dim)

        # Transforms for each continuous kernel scale
        self.kernel_projections = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in self.sigmas
        ])

        # Pointwise feedforward post-convolution
        if use_mlp:
            self.post_mlp = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.GELU(),
                nn.Linear(out_dim * 2, out_dim)
            )
        else:
            self.post_mlp = nn.Identity()

        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        intensities: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D_in] continuous point features
            coords: [B, N] continuous coordinates (m/z or delta m/z)
            intensities: [B, N] physical peak intensities (default 1.0)
            mask: [B, N] boolean mask (True for real peaks)
        Returns:
            out: [B, N, D_out] updated point features
        """
        B, N, D = x.shape
        w_scales = F.softmax(self.scale_weights, dim=0)

        # Direct linear mapping
        out = self.w_direct(x)

        # Apply continuous kernel operator across each physical scale
        for idx, (sigma, proj) in enumerate(zip(self.sigmas, self.kernel_projections)):
            x_proj = proj(x)
            
            # If intensities are provided, weight the source features by quadrature weights
            if intensities is not None:
                x_weighted = x_proj * intensities.unsqueeze(-1)
            else:
                x_weighted = x_proj

            # Execute continuous convolution at scale sigma
            # Uses fast Triton kernel on GPU if available, else PyTorch fallback
            conv_out = triton_continuous_rbf_conv(x_weighted, coords, sigma=sigma)

            # Accumulate scale contributions
            out = out + w_scales[idx] * conv_out

        if mask is not None:
            out = out * mask.unsqueeze(-1)

        # Residual connection + LayerNorm + MLP
        out = self.norm(out)
        out = out + self.post_mlp(out)

        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out
