"""
Forward Spectral Loss: Dual-Sided MS/MS Training Objective
Trains the forward predictor to reconstruct physical collision-induced dissociation (CID)
fragment spectra given molecular chemical structure and collision energy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_spectral_density_target(
    mzs: torch.Tensor,
    intensities: torch.Tensor,
    mask: torch.Tensor,
    num_bins: int = 512,
    max_mz: float = 500.0
) -> torch.Tensor:
    """
    Projects discrete empirical Radon measure peaks (m_i, I_i) into a continuous
    normalized spectral abundance density vector over num_bins mass channels.
    """
    B, N = mzs.shape
    device = mzs.device
    target_density = torch.zeros((B, num_bins), dtype=torch.float32, device=device)

    bin_scale = num_bins / max_mz
    bin_indices = torch.clamp((mzs * bin_scale).long(), 0, num_bins - 1)

    # Accumulate intensities into mass bins
    valid_mask = mask & (mzs > 0.0)
    for b in range(B):
        b_idx = bin_indices[b][valid_mask[b]]
        b_ints = intensities[b][valid_mask[b]]
        if len(b_idx) > 0:
            target_density[b].scatter_add_(0, b_idx, b_ints)

    # L1 normalization (abundance conservation)
    sums = target_density.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return target_density / sums

class ForwardSpectralLoss(nn.Module):
    """
    Computes spectral cosine and Jensen-Shannon / KL divergence between
    forward predicted mass spectral density and experimental spectrum.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_density: torch.Tensor, target_density: torch.Tensor) -> torch.Tensor:
        # 1. Cosine spectral similarity loss: 1 - cos(pred, target)
        pred_norm = F.normalize(pred_density, p=2, dim=-1)
        target_norm = F.normalize(target_density, p=2, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        cos_loss = 1.0 - cos_sim.mean()

        # 2. Entropy / Cross-entropy divergence
        p = pred_density.clamp(min=self.eps)
        q = target_density.clamp(min=self.eps)
        kl = torch.sum(q * torch.log(q / p), dim=-1).mean()

        return cos_loss + 0.2 * kl
