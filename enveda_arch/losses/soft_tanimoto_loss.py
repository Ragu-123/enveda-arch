"""
Differentiable Soft Tanimoto Loss
Directly optimizes the intersection-over-union of continuous predicted fingerprint probabilities with binary ground truth.
Supports both unpacked float targets and high-performance packed uint8 bit arrays with fused Triton autodiff.
"""

import torch
import torch.nn as nn
from enveda_arch.kernels.triton_packed_ops import triton_packed_soft_tanimoto_loss

class SoftTanimotoLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, K] unnormalized logits
            targets: [B, K] float binary targets OR [B, K/8] packed uint8 bit targets
        Returns:
            loss: scalar or [B]
        """
        if targets.dtype == torch.uint8:
            return triton_packed_soft_tanimoto_loss(logits, targets, eps=self.eps)

        probs = torch.sigmoid(logits.float())
        targets = targets.float()

        intersection = torch.sum(probs * targets, dim=-1)
        union = torch.sum(probs + targets - (probs * targets), dim=-1)

        tanimoto = (intersection + self.eps) / (union + self.eps)
        loss = 1.0 - tanimoto

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
