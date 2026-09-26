"""
Asymmetric Loss (ASL) for Multi-Label Classification
Reference: Emanuel Ben-Baruch et al., arXiv:2009.14119
Designed for sparse binary targets (e.g. 2048-bit Morgan Fingerprints where 98% of bits are 0).
"""

import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8, reduction: str = 'mean'):
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
            targets: [B, K] binary ground truth {0, 1}
        Returns:
            loss: scalar or [B] depending on reduction
        """
        # Probabilities
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Asymmetric clipping / probability shifting for negative samples
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic Cross Entropy logs
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.gamma_pos > 0:
                pt0 = xs_pos * targets
                pt1 = (1.0 - xs_pos) * (1.0 - targets)
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
                one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
                loss *= one_sided_w

        loss = -loss.sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
