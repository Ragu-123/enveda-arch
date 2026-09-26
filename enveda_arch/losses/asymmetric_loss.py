"""
Asymmetric Loss (ASL) for Multi-Label Classification
Reference: Emanuel Ben-Baruch et al., arXiv:2009.14119
Designed for sparse binary targets (e.g. 2048-bit Morgan Fingerprints where 98% of bits are 0).
Guaranteed numerically stable in FP16 / AMP autocast.
"""

import torch
import torch.nn as nn

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
            targets: [B, K] binary ground truth {0, 1}
        Returns:
            loss: scalar or [B] depending on reduction
        """
        # Always compute loss in float32 to prevent AMP / fp16 underflows/overflows
        logits = logits.float()
        targets = targets.float()

        p = torch.sigmoid(logits)

        # Positive targets: -targets * (1 - p)^gamma_pos * log(p)
        p_pos = p.clamp(min=self.eps, max=1.0 - self.eps)
        loss_pos = -targets * torch.pow(1.0 - p_pos, self.gamma_pos) * torch.log(p_pos)

        # Negative targets with asymmetric probability margin shifting:
        # p_neg = max(p - m, 0)
        p_neg = (p - self.clip).clamp(min=0.0, max=1.0 - self.eps)
        loss_neg = -(1.0 - targets) * torch.pow(p_neg, self.gamma_neg) * torch.log((1.0 - p_neg).clamp(min=self.eps))

        loss = loss_pos + loss_neg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
