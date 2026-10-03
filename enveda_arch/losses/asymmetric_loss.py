"""
Asymmetric Loss (ASL) for Multi-Label Classification
Reference: Emanuel Ben-Baruch et al., arXiv:2009.14119
Designed for sparse binary targets (e.g. 10,226-bit or 2048-bit Morgan Fingerprints).
Guaranteed numerically stable in FP16 / AMP autocast.
Supports both unpacked float targets and high-performance packed uint8 bit arrays with fused Triton autodiff.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enveda_arch.kernels.triton_packed_ops import triton_packed_asl_loss, unpack_bits_torch

class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-6,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, K] unnormalized logits
            targets: [B, K] float binary targets OR [B, K/8] packed uint8 bit targets
        Returns:
            loss: scalar or [B] depending on reduction
        """
        if targets.dtype == torch.uint8:
            return triton_packed_asl_loss(
                logits, targets,
                gamma_pos=self.gamma_pos,
                gamma_neg=self.gamma_neg,
                margin=self.clip,
                eps=self.eps
            )

        logits = logits.float()
        targets = targets.float()

        p = torch.sigmoid(logits)

        # Positive targets: -targets * (1 - p)^gamma_pos * log(p)
        p_pos = p.clamp(min=self.eps, max=1.0 - self.eps)
        loss_pos = -targets * torch.pow(1.0 - p_pos, self.gamma_pos) * torch.log(p_pos)

        # Negative targets with asymmetric probability margin shifting:
        p_neg = (p - self.clip).clamp(min=0.0, max=1.0 - self.eps)
        loss_neg = -(1.0 - targets) * torch.pow(p_neg, self.gamma_neg) * torch.log((1.0 - p_neg).clamp(min=self.eps))

        loss = loss_pos + loss_neg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class BalancedSubstructureLoss(nn.Module):
    """
    Balanced Multilabel Binary Cross-Entropy Loss for Sparse Molecular Fingerprints.
    Solves gradient vanishing on sparse binary targets.
    Supports packed uint8 bit arrays via vectorized unpacking.
    """
    def __init__(self, neg_weight: float = 0.25, eps: float = 1e-7):
        super().__init__()
        self.neg_weight = neg_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dtype == torch.uint8:
            targets = unpack_bits_torch(targets, n_bits=logits.shape[1])

        logits = logits.float()
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        n_pos = pos_mask.sum().clamp(min=1.0)
        n_neg = neg_mask.sum().clamp(min=1.0)
        pos_loss = (bce * pos_mask.float()).sum() / n_pos
        neg_loss = (bce * neg_mask.float()).sum() / n_neg
        return pos_loss + self.neg_weight * neg_loss
