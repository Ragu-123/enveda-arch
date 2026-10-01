"""
Asymmetric Loss (ASL) for Multi-Label Classification
Reference: Emanuel Ben-Baruch et al., arXiv:2009.14119
Designed for sparse binary targets (e.g. 2048-bit Morgan Fingerprints where 98% of bits are 0).
Guaranteed numerically stable in FP16 / AMP autocast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class BalancedSubstructureLoss(nn.Module):
    """
    Balanced Multilabel Binary Cross-Entropy Loss for Sparse Molecular Fingerprints.
    Solves gradient vanishing on sparse binary targets (e.g. 2048-bit Morgan vectors where 98% of bits are 0).
    Averages loss over positive bits and negative bits separately, ensuring positive substructure bits receive
    strong, undiluted gradients of O(1) magnitude instead of being divided by 2048.
    """
    def __init__(self, neg_weight: float = 0.25, eps: float = 1e-7):
        super().__init__()
        self.neg_weight = neg_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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

