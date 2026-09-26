"""
Differentiable Soft Tanimoto Loss
Directly optimizes the intersection-over-union of continuous predicted fingerprint probabilities with binary ground truth.
Guaranteed numerically stable under AMP / fp16 autocast.
"""

import torch
import torch.nn as nn

class SoftTanimotoLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, K] unnormalized logits
            targets: [B, K] binary ground truth {0, 1}
        Returns:
            loss: scalar or [B]
        """
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
